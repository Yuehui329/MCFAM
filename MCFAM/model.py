'''
noisy_rate=0.9
self.noisy_rate=noisy_rate
self.gene_batch_normal=nn.BatchNorm1d(output_dim)
self.weight = nn.Parameter(torch.zeros(size=(gene_num, output_dim)))
nn.init.xavier_uniform_(self.weight.data, gain=1.414)
x=self.gene_batch_normal(x)
x=x+self.noisy_rate*self.weight
'''
from typing import Dict, Optional
#import torch

import numpy as np
#import tensorflow
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import sys

sys.path.append('..')
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
# from utils.lookahead import Lookahead
# from transformers import AdamW
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import random
positive_total_num=9
random.seed(1)
# co-attention
def create_src_lengths_mask(
        batch_size: int, src_lengths: Tensor, max_src_len: Optional[int] = None
):
    """
    Generate boolean mask to prevent attention beyond the end of source
    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths
      max_src_len: Optionally override max_src_len for the mask
    Outputs:
      [batch_size, max_src_len]
    """
    if max_src_len is None:
        max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)

    return (src_indices < src_lengths).int().detach()


def masked_softmax(scores, src_lengths, src_length_masking=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if src_length_masking:
        bsz, max_src_len = scores.size()
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        src_mask = src_mask.to(device)
        scores = scores.masked_fill(src_mask == 0, -np.inf)

    return F.softmax(scores.float(), dim=-1).type_as(scores)


class ParallelCoAttentionNetwork(nn.Module):

    def __init__(self, hidden_dim, co_attention_dim, device, src_length_masking=True):
        super(ParallelCoAttentionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.src_length_masking = src_length_masking

        self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))

    def forward(self, V, Q, Q_lengths, device):
        """
        :param V: batch_size * hidden_dim * region_num, eg B x 512 x 196
        :param Q: batch_size * seq_len * hidden_dim, eg B x L x 512
        :param Q_lengths: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """
        V = V.to(device)
        Q = Q.to(device)
        self.W_b = self.W_b.to(device)
        self.W_v = self.W_v.to(device)
        self.W_q = self.W_q.to(device)
        V = V.to(torch.bfloat16)
        Q = Q.to(torch.bfloat16)
        C = torch.matmul(Q, torch.matmul(self.W_b.to(torch.bfloat16), V))
        H_v = nn.Tanh()(torch.matmul(self.W_v.to(torch.bfloat16), V) + torch.matmul(torch.matmul(self.W_q.to(torch.bfloat16), Q.permute(0, 2, 1)), C))
        H_q = nn.Tanh()(
            torch.matmul(self.W_q.to(torch.bfloat16), Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v.to(torch.bfloat16), V), C.permute(0, 2, 1)))

        a_v = F.softmax(torch.matmul(torch.t(self.w_hv.to(torch.bfloat16)), H_v.to(torch.bfloat16)), dim=2)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq.to(torch.bfloat16)), H_q.to(torch.bfloat16)), dim=2)

        masked_a_q = masked_softmax(
            a_q.squeeze(1), Q_lengths, self.src_length_masking
        ).unsqueeze(1)

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)), dim=1)
        q = torch.squeeze(torch.matmul(masked_a_q, Q), dim=1)

        return a_v, masked_a_q, v, q


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        torch.cuda.empty_cache()
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3,
                                           2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x

class WSelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)

        return x

class AttentionFeatureFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFeatureFusion, self).__init__()
        self.query = nn.Parameter(torch.randn(feature_dim))
        self.key = nn.Linear(feature_dim, feature_dim)

    def forward(self, x1, x2, x3):
        features = torch.stack([x1, x2, x3], dim=1)
        keys = self.key(features)
        scores = torch.einsum('d,bijd->bij', self.query, keys)
        weights = F.softmax(scores, dim=1)
        fused = torch.einsum('bij,bijd->bjd', weights, features)

        return fused

class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout, device):
        super().__init__()
        assert kernel_size % 2 == 1
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid_dim, 2 * hid_dim, kernel_size, padding=(kernel_size - 1) // 2) for _ in
             range(self.n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved,
                           dim=1)
            conved = (
                                 conved + conv_input) * self.scale
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg


class Decoder(nn.Module):
    def __init__(self, embed_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super(Decoder, self).__init__()
        self.ft = nn.Linear(embed_dim, hid_dim)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.dropout = dropout
        self.device = device

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = F.dropout(self.ft(trg), p=self.dropout)
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        return trg  # [bs, seq_len, hid_dim]

class TextCNN(nn.Module):
    def __init__(self, embed_dim, hid_dim, kernels=[3, 5, 7],
                 dropout_rate=0.5):
        super(TextCNN, self).__init__()
        padding1 = (kernels[0] - 1) // 2
        padding2 = (kernels[1] - 1) // 2
        padding3 = (kernels[2] - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv = nn.Sequential(
            nn.Linear(hid_dim * len(kernels), hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )

    def forward(self, protein):
        protein = protein.permute([0, 2, 1])  # [bs, hid_dim, seq_len]维度变换
        features1 = self.conv1(protein)
        features2 = self.conv2(protein)
        features3 = self.conv3(protein)
        features = torch.cat((features1, features2, features3), 1)  # [bs, hid_dim*3, seq_len]
        features = features.max(dim=-1)[0]  # [bs, hid_dim*3]
        return self.conv(features)


class Predictor(nn.Module):
    def __init__(self, hid_dim, n_layers, kernel_size, n_heads, pf_dim, dropout, device, atom_dim,protein_dim):
        super(Predictor, self).__init__()
        protein_dim = 100
        atom_dim = 34
        id2smi, smi2id, smi_embed = np.load('./data/drugbank/pretrain_embed/smi2vec.npy', allow_pickle=True)
        id2prot, prot2id, prot_embed = np.load('./data/drugbank/pretrain_embed/prot2vec.npy', allow_pickle=True)
        gene_num=positive_total_num
        self.noisy_rate =0.2
        self.weight = nn.Parameter(torch.zeros(size=(gene_num, hid_dim * 4)))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.dropout = dropout
        self.device = device
        self.prot_embed = nn.Embedding(len(prot_embed) + 1, len(prot_embed[0]), padding_idx=0)
        self.prot_embed.data = prot_embed
        for param in self.prot_embed.parameters():
            param.requires_grad = False
        self.smi_embed = nn.Embedding(len(smi_embed) + 1, len(smi_embed[0]), padding_idx=0)
        self.smi_embed.data = smi_embed
        for param in self.smi_embed.parameters():
            param.requires_grad = False
        print(f'prot Embed: {len(prot_embed)},  smi Embed: {len(smi_embed)}')

        self.enc_prot = Encoder(len(prot_embed[0]), hid_dim, n_layers, kernel_size, dropout, device)
        self.dec_smi = Decoder(len(smi_embed[0]), hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                               PositionwiseFeedforward, dropout, device)

        self.enc_smi = Encoder(len(smi_embed[0]), hid_dim, n_layers, kernel_size, dropout, device)
        self.dec_prot = Decoder(len(prot_embed[0]), hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention,
                                PositionwiseFeedforward, dropout, device)

        self.co_attention = ParallelCoAttentionNetwork(hid_dim, n_heads, device)
        self.prot_textcnn = TextCNN(100, hid_dim)
        self.W_gnn = nn.ModuleList([nn.Linear(atom_dim, atom_dim),
                                    nn.Linear(atom_dim, atom_dim),
                                    nn.Linear(atom_dim, atom_dim)])
        self.W_gnn_trans = nn.Linear(atom_dim, hid_dim)

        self.out = nn.Sequential(

            # nn.Linear(hid_dim * 3, 1024),
            nn.Linear(hid_dim * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.do = nn.Dropout(dropout)
        self.atom_dim = atom_dim
        self.compound_attn = nn.ParameterList(
            [nn.Parameter(torch.randn(size=(2 * atom_dim, 1))) for _ in range(len(self.W_gnn))])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.LGMFAtt = WSelfAttention(hid_dim=hid_dim, n_heads=1, dropout=0.2, device=device)

    def gnn(self, xs, A):
        for i in range(len(self.W_gnn)):
            h = torch.relu(
                self.W_gnn[i](xs))
            size = h.size()[0]
            N = h.size()[1]
            a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1,
                                                                                                          2 * self.atom_dim)

            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(
                3))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(A > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h)
            xs = xs + h_prime
        xs = self.do(F.relu(self.W_gnn_trans(xs)))
        return xs

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask

    def mcf_LGMFd(self, d1, d2):
        q = d1
        k = d2
        v = d2
        batchs = q.shape[0]
        q_lens = torch.zeros(batchs).long()
        q_lens[:] = q.shape[1]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        xd = self.LGMFAtt(q, k, v)
        return xd

    def mcf_LGMFp(self, p1, p2):
        q = p1
        k = p2
        v = p2
        batchs = q.shape[0]
        q_lens = torch.zeros(batchs).long()
        q_lens[:] = q.shape[1]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        xp = self.LGMFAtt(q, k, v)
        return xp

    def mcf_CMIGAd(self, q, v, xd):
        batchs = q.shape[0]
        smi_d = xd
        q_lens = torch.zeros(batchs).long()
        q_lens[:] = q.shape[1]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _, _, FDv, DFq = self.co_attention(v, q, q_lens, device)
        return FDv, DFq

    def mcf_CMIGAp(self, q, v, xp):
        batchs = v.shape[0]
        y_lens = xp
        q_lens = torch.zeros(batchs).long()
        q_lens[:] = q.shape[1]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _, _, FPv, PFq = self.co_attention(v, q, q_lens, device)
        return FPv, PFq

    def mcf(self, g_smi, out_dec_ld, out_dec_lp, p_cnn):
        xd = self.mcf_LGMFd(g_smi, out_dec_ld)

        xp = self.mcf_LGMFp(p_cnn, out_dec_lp)
        v = out_dec_ld.permute(0, 2, 1)
        q = g_smi
        FDv, DFq = self.mcf_CMIGAd(q, v, xd)

        v = out_dec_lp.permute(0, 2, 1)
        if p_cnn.ndim == 2: p_cnn = torch.unsqueeze(p_cnn, dim=1)
        q = p_cnn
        FPv, PFq = self.mcf_CMIGAp(q, v, xp)
        return FDv, DFq, FPv, PFq

    def forward(self, compound, adj, protein, smi_ids, prot_ids, smi_num, prot_num,padding_label,train_index,mode):
        Gdrug = self.gnn(compound, adj)
        Pcnn = self.prot_textcnn(protein)

        smi_max_len = smi_ids.shape[1]
        prot_max_len = prot_ids.shape[1]

        smi_mask, prot_mask = self.make_masks(smi_num, prot_num, smi_max_len, prot_max_len)
        out_enc_prot = self.enc_prot(self.prot_embed(prot_ids))
        out_dec_ld = self.dec_smi(self.smi_embed(smi_ids), out_enc_prot, smi_mask, prot_mask)

        prot_mask, smi_mask = self.make_masks(prot_num, smi_num, prot_max_len, smi_max_len)
        out_enc_smi = self.enc_smi(self.smi_embed(smi_ids))
        out_dec_lp = self.dec_prot(self.prot_embed(prot_ids), out_enc_smi, prot_mask, smi_mask)

        FDv, DFq, FPv, PFq = self.mcf(Gdrug, out_dec_ld, out_dec_lp, Pcnn)
        is_max = False
        if is_max:
            Gdrug = Gdrug.max(dim=1)[0]
            if Pcnn.ndim >= 3: Pcnn = Pcnn.max(dim=1)[0]
            out_dec_ld = out_dec_ld.max(dim=1)[0]
            out_dec_lp = out_dec_lp.max(dim=1)[0]
        else:
            Gdrug = Gdrug.mean(dim=1)
            if Pcnn.ndim>=3: Pcnn = Pcnn.mean(dim=1)
            out_dec_ld = out_dec_ld.mean(dim=1)
            out_dec_lp = out_dec_lp.mean(dim=1)

        out_fc = torch.cat([DFq, FDv, PFq, FPv], dim=-1)
        if mode==True:
            train_index1 = random.choices(train_index, k=positive_total_num - len(train_index))
            x2 = out_fc[train_index]
            x1 = out_fc[train_index1]
            x_new = torch.cat([x2, x1], dim=0)
            x_new = (1-self.noisy_rate)*x_new + self.noisy_rate * self.weight
            if len(padding_label)!=0:
                generated_data=[index for index in range(x_new.shape[0])]
                random_index=random.choices(generated_data, k=2)
                x_new=x_new[random_index]
            out_fc=torch.cat([out_fc, x_new], dim=0)

        out_fc = out_fc.to(self.out[-1].weight.dtype)
        return self.out(out_fc)

    def __call__(self, data, train=True):
        compound, adj, protein, correct_interaction, smi_ids, prot_ids, atom_num, protein_num,padding_label,train_index,mode = data
        Loss = nn.CrossEntropyLoss()
        if train:
            predicted_interaction = self.forward(compound, adj, protein, smi_ids, prot_ids, atom_num, protein_num,padding_label,train_index,True)
            padding_label = torch.tensor([1,1])
            padding_label = padding_label.to(correct_interaction.device)
            correct_interaction = torch.cat((correct_interaction, padding_label))
            loss = Loss(predicted_interaction, correct_interaction)
            return loss
        else:
            predicted_interaction = self.forward(compound, adj, protein, smi_ids, prot_ids, atom_num,
                                                 protein_num,padding_label,train_index,False)
            correct_interaction = correct_interaction
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores


MAX_PROTEIN_LEN = 1500
MAX_DRUG_LEN = 200


def pack(atoms, adjs, proteins, labels, smi_ids, prot_ids, device,padding_label,train_index,mode):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num, protein_num = [], []

    for atom in atoms:
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    for protein in proteins:
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]

    if atoms_len > MAX_DRUG_LEN: atoms_len = MAX_DRUG_LEN
    atoms_new = torch.zeros((N, atoms_len, 34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        if a_len > atoms_len: a_len = atoms_len
        atoms_new[i, :a_len, :] = atom[:a_len, :]
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        if a_len > atoms_len: a_len = atoms_len
        adjs_new[i, :a_len, :a_len] = adj[:a_len, :a_len]
        i += 1

    if proteins_len > MAX_PROTEIN_LEN: proteins_len = MAX_PROTEIN_LEN
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        if a_len > proteins_len: a_len = proteins_len
        proteins_new[i, :a_len, :] = protein[:a_len, :]
        i += 1
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    smi_id_len = 0
    for smi_id in smi_ids:
        atom_num.append(len(smi_id))
        if len(smi_id) >= smi_id_len:
            smi_id_len = len(smi_id)

    if smi_id_len > MAX_DRUG_LEN: smi_id_len = MAX_DRUG_LEN
    smi_ids_new = torch.zeros([N, smi_id_len], dtype=torch.long, device=device)
    for i, smi_id in enumerate(smi_ids):
        t_len = len(smi_id)
        if t_len > smi_id_len: t_len = smi_id_len
        smi_ids_new[i, :t_len] = smi_id[:t_len]
    ##########################################################
    prot_id_len = 0
    for prot_id in prot_ids:
        protein_num.append(len(prot_id))
        if len(prot_id) >= prot_id_len: prot_id_len = len(prot_id)

    if prot_id_len > MAX_PROTEIN_LEN: prot_id_len = MAX_PROTEIN_LEN
    prot_ids_new = torch.zeros([N, prot_id_len], dtype=torch.long, device=device)
    for i, prot_id in enumerate(prot_ids):
        t_len = len(prot_id)
        if t_len > prot_id_len: t_len = prot_id_len
        prot_ids_new[i, :t_len] = prot_id[:t_len]
    return (atoms_new, adjs_new, proteins_new, labels_new, smi_ids_new, prot_ids_new, atom_num, protein_num,padding_label,train_index,mode)


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch, n_sample, epochs):
        self.model = model
        weight_p, bias_p = [], []
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer = AdamW(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=10,
                                                            num_training_steps=(n_sample // batch) * epochs)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        # my_test = 0
        i = 0
        self.optimizer.zero_grad()
        adjs, atoms, proteins, labels, smi_ids, prot_ids = [], [], [], [], [], []
        adjs1, atoms1, proteins1, labels1, smi_ids1, prot_ids1 = [], [], [], [], [], []
        previous_train_positive_samples=[]
        for data in dataset:
            print(i)
            i = i + 1
            atom, adj, protein, label, smi_id, prot_id = data
            adjs.append(adj)
            atoms.append(atom)
            proteins.append(protein)
            labels.append(label)
            smi_ids.append(smi_id)
            prot_ids.append(prot_id)
            train_index = [index for index in range(len(labels)) if labels[index] == 1]

            if i % 8 == 0:
                if len(train_index)!=0:
                    train_index = [index for index in range(len(labels)) if labels[index] == 1]
                    adjs1=[adjs[index] for index in train_index]
                    atoms1 = [atoms[index] for index in train_index]
                    proteins1= [proteins[index] for index in train_index]
                    labels1= [labels[index] for index in train_index]
                    smi_ids1= [smi_ids[index] for index in train_index]
                    prot_ids1 = [prot_ids[index] for index in train_index]
                    padding_labels = [1 for i in range(positive_total_num)]
                    data_pack = pack(atoms, adjs, proteins, labels, smi_ids, prot_ids, device, padding_labels,
                                     train_index,True)
                    loss = self.model(data_pack, train=True)
                    loss.backward()

                    adjs, atoms, proteins, labels, smi_ids, prot_ids = [], [], [], [], [], []
                else:
                    adjs=adjs1+adjs
                    atoms=atoms1+atoms
                    proteins=proteins1+proteins
                    labels=labels1+labels
                    smi_ids=smi_ids1+smi_ids
                    prot_ids=prot_ids1+prot_ids
                    train_index_1 = [index for index in range(len(labels)) if labels[index] == 1]
                    padding_labels = [1 for i in range(positive_total_num)]
                    data_pack = pack(atoms, adjs, proteins, labels, smi_ids, prot_ids, device, padding_labels,
                                     train_index_1,True)
                    loss = self.model(data_pack, train=True)
                    loss.backward()

                    adjs, atoms, proteins, labels, smi_ids, prot_ids = [], [], [], [], [], []


            else:
                continue
            if i % self.batch == 0 or i == N:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model,padding_labels,train_index_1):
        self.model = model
        self.padding_labels=padding_labels
        self.train_index_1=train_index_1

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins, labels, smi_ids, prot_ids = [], [], [], [], [], []
                atom, adj, protein, label, smi_id, prot_id = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                smi_ids.append(smi_id)
                prot_ids.append(prot_id)

                data = pack(atoms, adjs, proteins, labels, smi_ids, prot_ids, self.model.device,self.padding_labels,
                                     self.train_index_1,False)
                correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, PRC, precision, recall

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

