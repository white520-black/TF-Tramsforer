"""Feature tokenizer + PreNorm Transformer, following paper section 3.3."""
import torch
from torch import nn


class ReGLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * torch.relu(b)


class Block(nn.Module):
    def __init__(self, d, heads, hidden, attn_dropout, ffn_dropout, first):
        super().__init__()
        self.norm1 = nn.Identity() if first else nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(d, heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, hidden * 2), ReGLU(),
                                 nn.Dropout(ffn_dropout), nn.Linear(hidden, d))

    def forward(self, x):
        q = self.norm1(x)
        x = x + self.attention(q, q, q, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class FTTransformer(nn.Module):
    def __init__(self, n_numeric, cardinalities, d_token=192, layers=3, heads=8,
                 attention_dropout=.2, ffn_dropout=.1):
        super().__init__()
        if d_token % heads:
            raise ValueError('d_token must be divisible by heads')
        self.num_weight = nn.Parameter(torch.empty(n_numeric, d_token))
        self.feature_bias = nn.Parameter(torch.empty(n_numeric + len(cardinalities), d_token))
        self.embeddings = nn.ModuleList([nn.Embedding(n, d_token) for n in cardinalities])
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        for w in [self.num_weight, self.feature_bias, self.cls]:
            nn.init.uniform_(w, -d_token ** -.5, d_token ** -.5)
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -d_token ** -.5, d_token ** -.5)
        self.blocks = nn.ModuleList([Block(d_token, heads, int(d_token * 4 / 3),
            attention_dropout, ffn_dropout, first=i == 0) for i in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, 10))

    def forward(self, xnum, xcat):
        numeric = xnum.unsqueeze(-1) * self.num_weight.unsqueeze(0)
        categorical = torch.stack([e(xcat[:, i]) for i, e in enumerate(self.embeddings)], dim=1)
        tokens = torch.cat([numeric, categorical], dim=1) + self.feature_bias
        x = torch.cat([self.cls.expand(len(xnum), -1, -1), tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])  # logits, not softmax; CrossEntropyLoss handles it.
