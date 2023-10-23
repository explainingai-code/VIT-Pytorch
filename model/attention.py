import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config['n_heads']
        self.head_dim = config['head_dim']
        self.emb_dim = config['emb_dim']
        self.drop_prob = config['dropout'] if 'dropout' in config else 0.0
        self.att_dim = self.n_heads * self.head_dim
        
        self.qkv_proj = nn.Linear(self.emb_dim, 3 * self.att_dim, bias=False)
        self.output_proj = nn.Sequential(
            nn.Linear(self.att_dim, self.emb_dim),
            nn.Dropout(self.drop_prob))

        self.attn_dropout = nn.Dropout(self.drop_prob)

    def forward(self, x):
        
        #  Converting to Attention Dimension
        ######################################################
        # Batch Size x Number of Patches x Dimension
        B, N = x.shape[:2]
        # Projecting to 3*att_dim and then splitting to get q, k v(each of att_dim)
        # qkv -> Batch Size x Number of Patches x (3* Attention Dimension)
        # q(as well as k and v) -> Batch Size x Number of Patches x Attention Dimension
        q, k ,v = self.qkv_proj(x).split(self.att_dim, dim=-1)
        # Batch Size x Number of Patches x Attention Dimension
        # -> Batch Size x Number of Patches x (Heads * Head Dimension)
        # -> Batch Size x Number of Patches x (Heads * Head Dimension)
        # -> Batch Size x Heads x Number of Patches x Head Dimension
        # -> B x H x N x Head Dimension
        q = rearrange(q, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        k = rearrange(k, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        v = rearrange(v, 'b n (n_h h_dim) -> b n_h n h_dim',
                      n_h=self.n_heads, h_dim=self.head_dim)
        #########################################################
        
        # Compute Attention Weights
        #########################################################
        # B x H x N x Head Dimension @ B x H x Head Dimension x N
        # -> B x H x N x N
        att = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**(-0.5))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        #########################################################
        
        # Weighted Value Computation
        #########################################################
        #  B x H x N x N @ B x H x N x Head Dimension
        # -> B x H x N x Head Dimension
        out = torch.matmul(att, v)
        #########################################################
        
        # Converting to Transformer Dimension
        #########################################################
        # B x N x (Heads * Head Dimension) -> B x N x (Attention Dimension)
        out = rearrange(out, 'b n_h n h_dim -> b n (n_h h_dim)')
        #  B x N x Dimension
        out = self.output_proj(out)
        ##########################################################
        
        return out
    
