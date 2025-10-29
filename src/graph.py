import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Optional
from tqdm import trange

# === GNN ===
from torch_geometric.nn import HGTConv

class HeteroGNN(nn.Module):
    def __init__(self, metadata, num_users: int, emb_dim=768, num_layers=2):
        super().__init__()
        # self.user_emb = nn.Parameter(torch.zeros(num_users, emb_dim))  # Learnable user emb
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)

        for _ in range(num_layers):
            self.convs.append(HGTConv(emb_dim, emb_dim, metadata=metadata, heads=4))
            self.norms.append(nn.LayerNorm(emb_dim))

    def forward(self, x_dict, edge_index_dict):
        h = x_dict.copy()
        # h["user"] = self.user_emb  # Inject learnable user emb

        for i, conv in enumerate(self.convs):
            out = conv(h, edge_index_dict)
            out = {k: self.norms[i](self.dropout(v)) for k, v in out.items()}
            h.update(out)
        return h

class PreferencePredictor(nn.Module):
    def __init__(self, dim=768, hidden_dim=256, dropout=0.1, num_heads=4):
        super().__init__()
        self.user_proj = nn.Linear(dim, hidden_dim)
        self.query_proj = nn.Linear(dim, hidden_dim)
        self.llm_proj = nn.Linear(dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, user, query, llm):
        # Shape: [D] → [1, D]
        user = user.unsqueeze(0)
        query = query.unsqueeze(0)
        llm = llm.unsqueeze(0)

        u = self.user_proj(user)     # [1, H]
        q = self.query_proj(query)   # [1, H]
        l = self.llm_proj(llm)       # [1, H]

        # Combine user + query as context: [1, 2, H]
        context = torch.stack([u, q], dim=1)
        target = l.unsqueeze(1)  # [1, 1, H]

        # Apply attention: llm attends to user+query
        attended, _ = self.cross_attn(query=target, key=context, value=context)  # [1, 1, H]
        attended = attended.squeeze(1)  # [1, H]

        # Concatenate attended context and llm for final prediction
        x = torch.cat([attended, l], dim=-1)  # [1, 2H]
        return self.out(x).squeeze()          # scalar


