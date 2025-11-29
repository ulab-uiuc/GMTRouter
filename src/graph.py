import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import random
from typing import Dict, List, Tuple, Optional
from tqdm import trange

# === Graph Neural Network Components ===
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGNN(nn.Module):
    def __init__(self, metadata, num_users: int, emb_dim=768, num_layers=2):
        """
        Heterogeneous Graph Neural Network based on HeteroConv with SAGEConv
        for each edge type.

        Args:
            metadata: Tuple of (node_types, edge_types) as provided by
                PyTorch Geometric heterogeneous graph metadata.
            num_users: Number of user nodes (kept for compatibility; not used directly).
            emb_dim: Dimensionality of node embeddings.
            num_layers: Number of message-passing layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)

        node_types, edge_types = metadata

        for _ in range(num_layers):
            conv_dict = {
                edge_type: SAGEConv(emb_dim, emb_dim)
                for edge_type in edge_types
            }

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(nn.LayerNorm(emb_dim))

    def forward(self, x_dict, edge_index_dict):
        """
        Perform message passing on the heterogeneous graph.

        Args:
            x_dict: Dictionary mapping node types to feature tensors.
            edge_index_dict: Dictionary mapping edge types to edge index tensors.

        Returns:
            Updated dictionary of node-type-specific embeddings.
        """
        h = x_dict.copy()

        for i, conv in enumerate(self.convs):
            out = conv(h, edge_index_dict)
            out = {k: self.norms[i](self.dropout(F.relu(v))) for k, v in out.items()}
            h.update(out)
        return h


class PreferencePredictor(nn.Module):
    def __init__(self, dim=768, hidden_dim=256, dropout=0.1, num_heads=4):
        """
        Preference prediction head that combines user, query, and LLM
        embeddings via cross-attention.

        Args:
            dim: Input embedding dimensionality.
            hidden_dim: Hidden dimensionality used in projections and attention.
            dropout: Dropout probability applied before the final linear layer.
            num_heads: Number of attention heads in MultiheadAttention.
        """
        super().__init__()
        self.user_proj = nn.Linear(dim, hidden_dim)
        self.query_proj = nn.Linear(dim, hidden_dim)
        self.llm_proj = nn.Linear(dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1),
        )

    def forward(self, user, query, llm):
        """
        Compute a scalar preference score for a (user, query, llm) triple.

        All inputs are assumed to be 1D embedding vectors of shape [D].

        Args:
            user: User embedding tensor of shape [D].
            query: Query embedding tensor of shape [D].
            llm: LLM embedding tensor of shape [D].

        Returns:
            A scalar tensor representing the preference score.
        """
        # [1, D]
        user = user.unsqueeze(0)
        query = query.unsqueeze(0)
        llm = llm.unsqueeze(0)

        # [1, H]
        u = self.user_proj(user)   
        q = self.query_proj(query)
        l = self.llm_proj(llm)

        # [1, 2, H]
        context = torch.stack([u, q], dim=1)
        # [1, 1, H]
        target = l.unsqueeze(1)

        # [1, 1, H]
        attended, _ = self.cross_attn(query=target, key=context, value=context)
        attended = attended.squeeze(1)  # [1, H]

        x = torch.cat([attended, l], dim=-1)
        return self.out(x).squeeze()