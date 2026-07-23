"""
===========================================================================
Multimodal Encoder
Description: Sequence/2D-Graph adapter fused with a 3D structural EGNN
===========================================================================

Workflow:
1. Adapts frozen per-residue ESM2 embeddings via a small residual MLP
   (ProteinAdapterEncoder), then pools catalytic-site and substrate-
   conditioned representations directly from the adapted residue
   embeddings.
2. Embeds molecular graphs (ligand, co-substrate) using a Directed Message
   Passing Neural Network (D-MPNN) over atom-level structural features,
   fused with each molecule's whole-molecule ChemBERTa embedding
   (mol_combine).
3. Runs a heterogeneous, bipartite E(n)-Equivariant Graph Neural Network
   (StructuralEGNNEncoder) over the AlphaFold/ESMFold-derived protein
   pocket, and the same ligand/co-substrate atoms (now carrying 3D
   conformer coordinates), connected by interacts_with proximity edges
   (src/data/processors/geometry_processor.py). Reuses the ligand/co-
   substrate covalent_bond edges/features already built for the D-MPNN -
   no duplicate topology is stored on disk.
4. Concatenates the 2D pooled representations (protein/ligand/co-substrate)
   with the 3D EGNN pooled representations (protein pocket/ligand/co-
   substrate), pH, temperature, and the mutation descriptor into a single
   fused representation, then a unified kinetics head predicts k_1 and
   k_reverse. Returns these alongside the fused representation for the
   thermodynamics layer.

Known Caveats:
- Model capacity is kept deliberately small (see hidden_channels/
  egnn_hidden_channels defaults): this dataset has ~3,600 unique proteins
  and ~3,400 unique substrates, a fine-tuning-scale dataset for the frozen
  ESM2/ChemBERTa embeddings.
- If a record's tensors lack the `protein_pocket_atoms` node type (no 3D
  structural data available), `StructuralEGNNEncoder.forward` falls back
  to a zero vector rather than raising, so such records degrade
  gracefully instead of crashing training.

Author: ThermoKP Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import AttentionalAggregation, MessagePassing, global_mean_pool
from typing import Dict, List, Optional, Tuple, cast

from src.physics.thermodynamics import KB_SI, H_SI

# Duplicated from src/data/processors/pretrained_embeddings.py /
# generate_tensors.py / geometry_processor.py to avoid importing their
# transformers/rdkit/biopython/network-fetch dependencies into this module.
CHEMBERTA_EMBED_DIM = 384
MUTATION_FEATURE_DIM = 7
COVALENT_BOND_FEATURE_DIM = 5
PROTEIN_POCKET_ONEHOT_CHANNELS = 27  # element(4) + residue(20) + backbone(1) + catalytic(2)

# ═══════════════════════════════════════════════════════════════════════════
#  2D: Directed Message Passing (Ligand / Co-Substrate Graphs)
# ═══════════════════════════════════════════════════════════════════════════

class DMPNNLayer(nn.Module):
    """
    A single step of Directed Message Passing on edge states.
    For simplicity and PyG compatibility, this approximates a D-MPNN step
    by passing messages from nodes to edges and back, effectively propagating
    information along the molecular graph topology.

    Attributes
    ----------
    W : nn.Linear
        Projects the node-aggregated incoming message before the
        edge-state update.
    dropout : nn.Dropout
        Applied to the updated edge message before the residual add.
    norm : nn.LayerNorm
        Normalizes the residual sum of the previous and updated edge
        states.
    """
    def __init__(self, hidden_channels: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, h_edge: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h_edge : torch.Tensor
            Current edge hidden states, shape (num_edges, hidden_channels).
        edge_index : torch.Tensor
            Directed edge connectivity, shape (2, num_edges), with source
            and destination node indices as rows.

        Returns
        -------
        torch.Tensor
            Updated edge hidden states, same shape as `h_edge`. Returned
            unchanged if the graph has no edges.
        """
        src, dst = edge_index
        node_aggr = torch.zeros((int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0, h_edge.size(1)),
                                dtype=h_edge.dtype, device=h_edge.device)
        if edge_index.numel() > 0:
            node_aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(h_edge), h_edge)

        if edge_index.numel() > 0:
            msg = node_aggr[src]
            h_edge_new = F.gelu(self.W(msg))
            h_edge_new = self.dropout(h_edge_new)
            return self.norm(h_edge + h_edge_new)
        return h_edge

class DMPNNEncoder(nn.Module):
    """
    D-MPNN Encoder for 2D molecular graphs.

    Operates on pure atom-level structural features (element,
    hybridization, formal charge, aromaticity, Gasteiger charge, ring
    membership, degree, H-count, and an electrophilic/nucleophilic
    z-score pair - see `featurize_ligand` in
    src/data/processors/generate_tensors.py). The whole-molecule ChemBERTa
    embedding is fused in separately, once per graph, by
    `MultimodalEncoder.mol_combine` after this encoder's own pooling - not
    mixed into the per-atom input here.

    Attributes
    ----------
    W_i : nn.Linear
        Projects each directed edge's source-atom features concatenated
        with the bond features into the initial edge hidden state.
    layers : nn.ModuleList
        Stack of `DMPNNLayer` message-passing steps over edge states.
    W_o : nn.Linear
        Projects each atom's own features concatenated with its
        aggregated incoming edge messages into the final node
        representation.
    node_norm : nn.LayerNorm
        Normalizes the final per-atom representation.
    dropout : nn.Dropout
        Applied to the final per-atom representation before pooling.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        edge_attr_dim = COVALENT_BOND_FEATURE_DIM

        self.W_i = nn.Linear(in_channels + edge_attr_dim, hidden_channels)
        self.layers = nn.ModuleList([DMPNNLayer(hidden_channels, dropout) for _ in range(num_layers)])

        self.W_o = nn.Linear(in_channels + hidden_channels, hidden_channels)
        self.node_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor, num_graphs: Optional[int] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Per-atom structural features, shape (num_atoms, in_channels).
        edge_index : torch.Tensor
            Directed covalent-bond connectivity, shape (2, num_edges).
        edge_attr : torch.Tensor
            Per-edge bond features, shape (num_edges, COVALENT_BOND_FEATURE_DIM).
        batch : torch.Tensor
            Graph index for each atom, shape (num_atoms,), for pooling.
        num_graphs : int, optional
            Number of graphs in the batch. Passed to `global_mean_pool` so
            graphs with zero atoms still produce a pooled row of zeros.

        Returns
        -------
        torch.Tensor
            Mean-pooled per-graph molecular representation, shape
            (num_graphs, hidden_channels). Zero-filled for graphs with no
            atoms.
        """
        if x.size(0) == 0:
            return torch.empty((0, self.W_o.out_features), device=x.device)

        src, dst = edge_index
        if edge_index.numel() == 0:
            m_v_zeros = torch.zeros((x.size(0), self.W_o.out_features), dtype=x.dtype, device=x.device)
            h_node = F.gelu(self.W_o(torch.cat([x, m_v_zeros], dim=-1)))
            h_node = self.dropout(self.node_norm(h_node))
            return global_mean_pool(h_node, batch, size=num_graphs)

        h_edge = self.W_i(torch.cat([x[src], edge_attr], dim=-1))
        h_edge = F.gelu(h_edge)

        for layer in self.layers:
            h_edge = layer(h_edge, edge_index)

        m_v = torch.zeros((x.size(0), h_edge.size(1)), dtype=h_edge.dtype, device=h_edge.device)
        m_v.scatter_add_(0, dst.unsqueeze(-1).expand_as(h_edge), h_edge)

        h_node = F.gelu(self.W_o(torch.cat([x, m_v], dim=-1)))
        h_node = self.dropout(self.node_norm(h_node))

        return global_mean_pool(h_node, batch, size=num_graphs)

# ═══════════════════════════════════════════════════════════════════════════
#  2D: Protein Sequence Adapter & Pooling
# ═══════════════════════════════════════════════════════════════════════════

class ProteinAdapterEncoder(nn.Module):
    """
    Lightweight per-residue adapter over frozen ESM2 embeddings.

    ESM2 already encodes rich per-residue evolutionary and positional
    context via its own self-attention layers, so this only does a small
    residual refinement: project ESM2 down to hidden_channels, add a
    trainable amino-acid-identity embedding, and pass through a bottleneck
    MLP. Cross-residue reasoning is left to downstream task-specific
    mechanisms (SubstrateConditionedPooling, catalytic-site pooling)
    rather than a second self-attention stack here.

    Attributes
    ----------
    aa_embedding : nn.Embedding
        Trainable per-amino-acid-identity embedding, indexed by
        `aa_indices` (21 entries: 20 known amino acids plus 1
        unknown/'X' fallback).
    esm_proj : nn.Linear
        Projects the frozen ESM2 per-residue embedding down to
        `hidden_channels`.
    input_norm : nn.LayerNorm
        Normalizes the sum of the identity embedding and the projected
        ESM2 embedding.
    adapter : nn.Sequential
        Bottleneck residual MLP refining the combined per-residue
        representation.
    final_norm : nn.LayerNorm
        Normalizes the residual sum of the pre-adapter and adapter
        output representations.
    dropout : nn.Dropout
        Applied after `final_norm`.
    """
    def __init__(self, esm_dim: int = 1280, hidden_channels: int = 64, dropout: float = 0.1):
        """
        Parameters
        ----------
        esm_dim : int, optional
            Dimensionality of the frozen ESM2 per-residue embedding.
        hidden_channels : int, optional
            Width of the adapted per-residue representation.
        dropout : float, optional
            Dropout probability applied within the adapter MLP and to
            its residual output.
        """
        super().__init__()
        # 20 known amino acids + 1 unknown/'X' fallback index (sequence_to_indices, generate_tensors.py).
        self.aa_embedding = nn.Embedding(21, hidden_channels)
        self.esm_proj = nn.Linear(esm_dim, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)

        self.adapter = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.final_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, aa_indices: torch.Tensor, esm2_embedding: torch.Tensor, catalytic_mask: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        aa_indices : torch.Tensor
            Per-residue amino-acid identity index, shape (num_residues,).
        esm2_embedding : torch.Tensor
            Frozen per-residue ESM2 embedding, shape (num_residues, esm_dim).
        catalytic_mask : torch.Tensor
            Per-residue boolean flag marking curated catalytic-site
            residues, shape (num_residues,).
        batch : torch.Tensor
            Graph index for each residue, shape (num_residues,).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            catalytic_pooled : shape (B, hidden_channels), the
            catalytic-site-pooled representation (falling back to mean
            pooling over all residues when no catalytic site is
            annotated). h_combined : shape (B, L, hidden_channels), the
            dense per-residue adapted representation. mask : shape
            (B, L), the boolean validity mask for `h_combined`.
        """
        h_flat = self.aa_embedding(aa_indices) + self.esm_proj(esm2_embedding)
        h_flat = self.input_norm(h_flat)

        h_combined_flat = h_flat + self.adapter(h_flat)
        h_combined_flat = self.dropout(self.final_norm(h_combined_flat))

        from torch_geometric.utils import to_dense_batch

        h_combined, mask = to_dense_batch(h_combined_flat, batch)
        x_cat, _ = to_dense_batch(catalytic_mask, batch)

        is_catalytic = (x_cat.sum(dim=-1) > 0).float()
        has_catalytic = (is_catalytic.sum(dim=-1, keepdim=True) > 0).float()

        # Pool only over the active site; fall back to global mean pooling when none is annotated.
        pool_mask = is_catalytic * has_catalytic + mask.float() * (1.0 - has_catalytic)
        pool_mask = pool_mask.unsqueeze(-1)

        catalytic_pooled = (h_combined * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1e-6)

        return catalytic_pooled, h_combined, mask

class SubstrateConditionedPooling(nn.Module):
    """
    Attention pooling over protein residues, queried by a substrate's
    pooled representation - lets the bound substrate's identity influence
    which residues get pooled, complementing the static curated
    catalytic-site annotation (get_catalytic_site_mask,
    pretrained_embeddings.py), which is frequently absent for
    poorly-characterized enzymes.

    The protein side always has at least one valid residue for every
    graph, so this is well-defined (no fully-masked-key softmax) even
    when the substrate query is the zero placeholder used for an absent
    co-substrate.

    Attributes
    ----------
    attn : nn.MultiheadAttention
        Cross-attention module with the substrate representation as
        query and the per-residue protein representations as key/value.
    norm : nn.LayerNorm
        Normalizes the pooled attention output.
    """

    def __init__(self, hidden_channels: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Parameters
        ----------
        hidden_channels : int
            Width of the protein and substrate representations.
        num_heads : int, optional
            Number of attention heads.
        dropout : float, optional
            Dropout probability within the attention module.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_channels, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, protein_h: torch.Tensor, protein_mask: torch.Tensor, substrate_query: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        protein_h : torch.Tensor
            Per-residue protein representations, shape (B, L, hidden_channels).
        protein_mask : torch.Tensor
            Boolean validity mask, shape (B, L). True marks a real (non-padding) residue.
        substrate_query : torch.Tensor
            Pooled substrate representation, shape (B, hidden_channels).

        Returns
        -------
        torch.Tensor
            Substrate-conditioned pooled protein representation, shape (B, hidden_channels).
        """
        query = substrate_query.unsqueeze(1)
        key_padding_mask = ~protein_mask
        attn_out, _ = self.attn(query=query, key=protein_h, value=protein_h, key_padding_mask=key_padding_mask)
        return self.norm(attn_out.squeeze(1))

# ═══════════════════════════════════════════════════════════════════════════
#  3D: Heterogeneous Equivariant Graph Neural Network
# ═══════════════════════════════════════════════════════════════════════════

class HeteroEGNNMessagePassing(MessagePassing):
    """
    Bipartite Equivariant Graph Convolutional layer for heterogeneous graphs.

    Computes equivariant messages between source and destination node types
    and derives coordinate updates.

    Attributes
    ----------
    edge_mlp : nn.Sequential
        Multi-Layer Perceptron to compute the invariant message m_ij.
    coord_mlp : nn.Sequential or None
        Multi-Layer Perceptron to compute the weight for the coordinate
        update, or None when `update_pos=False`.
    dist_scale_sq : torch.Tensor
        Non-trainable buffer holding the squared distance (Angstrom^2)
        used to normalize the raw squared pairwise distance before it
        enters edge_mlp.
    edge_attr_dim : int
        Width of the edge-feature tensor this instance expects (0 for
        `interacts_with`, `COVALENT_BOND_FEATURE_DIM` for `covalent_bond`).
    update_pos : bool
        Whether this instance computes a coordinate update at all. False
        for the final EGNN layer, whose coordinate output is never read.
    """

    def __init__(
        self,
        hidden_channels: int,
        dropout: float = 0.3,
        dist_scale_sq: float = 36.0,
        edge_attr_dim: int = 0,
        update_pos: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        hidden_channels : int
            Width of the node representations and the invariant message.
        dropout : float, optional
            Dropout probability within `edge_mlp`.
        dist_scale_sq : float, optional
            Squared distance (Angstrom^2) used to normalize the raw
            squared pairwise distance before it enters `edge_mlp`.
        edge_attr_dim : int, optional
            Width of the edge-feature tensor this instance expects (0 if
            the edge type carries no edge attributes).
        update_pos : bool, optional
            Whether to build `coord_mlp` and compute a coordinate update.
        """
        super().__init__(aggr='mean')
        self.register_buffer("dist_scale_sq", torch.tensor(dist_scale_sq, dtype=torch.float32))
        self.edge_attr_dim = edge_attr_dim
        self.update_pos = update_pos
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1 + edge_attr_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1, bias=False)
        ) if update_pos else None

    def forward(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        pos: Tuple[torch.Tensor, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor]
            (source, destination) node features for this bipartite edge
            type, each shape (num_nodes_of_type, hidden_channels).
        pos : Tuple[torch.Tensor, torch.Tensor]
            (source, destination) node 3D coordinates, each shape
            (num_nodes_of_type, 3).
        edge_index : torch.Tensor
            Source-to-destination connectivity for this edge type, shape
            (2, num_edges).
        edge_attr : torch.Tensor, optional
            Per-edge features, shape (num_edges, edge_attr_dim), when
            `edge_attr_dim > 0`.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            h_msg : shape (num_dst_nodes, hidden_channels), the
            aggregated invariant message per destination node. pos_msg :
            shape (num_dst_nodes, 3), the aggregated coordinate update,
            or None when `update_pos=False`.
        """
        msg_out = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)

        if not self.update_pos:
            return msg_out, None

        h_msg = msg_out[:, :-3]
        pos_msg = msg_out[:, -3:]
        return h_msg, pos_msg

    def message(  # type: ignore[override]
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_i : torch.Tensor
            Destination-node features for each edge, shape (num_edges, hidden_channels).
        x_j : torch.Tensor
            Source-node features for each edge, shape (num_edges, hidden_channels).
        pos_i : torch.Tensor
            Destination-node coordinates for each edge, shape (num_edges, 3).
        pos_j : torch.Tensor
            Source-node coordinates for each edge, shape (num_edges, 3).
        edge_attr : torch.Tensor, optional
            Per-edge features, shape (num_edges, edge_attr_dim), when
            `edge_attr_dim > 0`.

        Returns
        -------
        torch.Tensor
            Per-edge message, shape (num_edges, hidden_channels) when
            `update_pos=False`, or (num_edges, hidden_channels + 3) with
            the coordinate displacement message appended otherwise.
        """
        dist_sq = torch.sum((pos_i - pos_j) ** 2, dim=-1, keepdim=True)
        # pyrefly: ignore [unsupported-operation]
        dist_sq_norm = dist_sq / self.dist_scale_sq

        edge_inputs = [x_i, x_j, dist_sq_norm]
        if edge_attr is not None:
            edge_inputs.append(edge_attr)
        m_ij = self.edge_mlp(torch.cat(edge_inputs, dim=-1))

        if not self.update_pos:
            return m_ij

        assert self.coord_mlp is not None
        coord_weight = self.coord_mlp(m_ij)
        coord_weight = 10.0 * torch.tanh(coord_weight / 10.0)
        pos_msg = (pos_i - pos_j) * coord_weight

        return torch.cat([m_ij, pos_msg], dim=-1)


class HeteroEGNNLayer(nn.Module):
    """
    A full heterogeneous EGNN layer managing multiple bipartite edge types.

    Aggregates messages from all incoming edge types for each node type
    before applying the node-wise update MLP.

    Attributes
    ----------
    edge_types : List[Tuple[str, str, str]]
        The list of bipartite interaction routes.
    mp_layers : nn.ModuleDict
        Dictionary containing `HeteroEGNNMessagePassing` instances per edge type.
    node_mlps : nn.ModuleDict
        Dictionary containing update MLPs per node type.
    update_pos : bool
        Whether this layer computes coordinate updates at all.
    """

    def __init__(
        self,
        hidden_channels: int,
        edge_types: List[Tuple[str, str, str]],
        dropout: float = 0.3,
        dist_scale_sq: float = 36.0,
        edge_attr_dims: Optional[Dict[Tuple[str, str, str], int]] = None,
        update_pos: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        hidden_channels : int
            Width of the node representations passed to every edge type's
            message-passing module and node-update MLP.
        edge_types : List[Tuple[str, str, str]]
            Bipartite (source, relation, destination) routes this layer
            manages.
        dropout : float, optional
            Dropout probability within the message-passing and node-update
            MLPs.
        dist_scale_sq : float, optional
            Squared distance (Angstrom^2) used to normalize pairwise
            distances in every edge type's message-passing module.
        edge_attr_dims : Dict[Tuple[str, str, str], int], optional
            Per-edge-type width of edge features (defaults to 0, i.e. no
            edge attributes, for any edge type not present).
        update_pos : bool, optional
            Whether every edge type's message-passing module computes a
            coordinate update.
        """
        super().__init__()
        self.edge_types = edge_types
        self.update_pos = update_pos
        edge_attr_dims = edge_attr_dims or {}

        self.mp_layers = nn.ModuleDict({
            "__".join(edge_type): HeteroEGNNMessagePassing(
                hidden_channels, dropout, dist_scale_sq, edge_attr_dims.get(edge_type, 0), update_pos
            )
            for edge_type in edge_types
        })

        node_types = set([src for src, _, _ in edge_types] + [dst for _, _, dst in edge_types])

        self.node_mlps = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout)
            ) for ntype in node_types
        })

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        pos_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        x_dict : Dict[str, torch.Tensor]
            Per-node-type feature tensors, each shape
            (num_nodes_of_type, hidden_channels).
        pos_dict : Dict[str, torch.Tensor]
            Per-node-type 3D coordinate tensors, each shape
            (num_nodes_of_type, 3).
        edge_index_dict : Dict[Tuple[str, str, str], torch.Tensor]
            Per-edge-type connectivity, each shape (2, num_edges).
        edge_attr_dict : Dict[Tuple[str, str, str], torch.Tensor], optional
            Per-edge-type edge features, for edge types with
            `edge_attr_dim > 0`.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            Updated (x_dict, pos_dict) with the same keys and shapes as
            the inputs. A node type with no incoming edges this layer is
            passed through unchanged.
        """
        h_msgs: Dict[str, List[torch.Tensor]] = {ntype: [] for ntype in self.node_mlps.keys()}
        pos_msgs: Dict[str, List[torch.Tensor]] = {ntype: [] for ntype in self.node_mlps.keys()}
        edge_attr_dict = edge_attr_dict or {}

        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue

            src, _, dst = edge_type
            key = "__".join(edge_type)

            if src not in x_dict or dst not in x_dict:
                continue
            if src not in pos_dict or dst not in pos_dict:
                continue

            x_tuple = (x_dict[src], x_dict[dst])
            pos_tuple = (pos_dict[src], pos_dict[dst])

            h_msg, pos_msg = self.mp_layers[key](
                x_tuple, pos_tuple, edge_index_dict[edge_type], edge_attr_dict.get(edge_type)
            )

            h_msgs[dst].append(h_msg)
            if pos_msg is not None:
                pos_msgs[dst].append(pos_msg)

        out_x_dict: Dict[str, torch.Tensor] = {}
        out_pos_dict: Dict[str, torch.Tensor] = {}

        for ntype in x_dict.keys():
            if ntype in h_msgs and len(h_msgs[ntype]) > 0:
                agg_h_msg = h_msgs[ntype][0]
                for msg in h_msgs[ntype][1:]:
                    agg_h_msg = agg_h_msg + msg

                out_x_dict[ntype] = (
                    x_dict[ntype]
                    + self.node_mlps[ntype](torch.cat([x_dict[ntype], agg_h_msg], dim=-1))
                )
            else:
                out_x_dict[ntype] = x_dict[ntype]

            if ntype in pos_dict:
                if ntype in pos_msgs and len(pos_msgs[ntype]) > 0:
                    agg_pos_msg = pos_msgs[ntype][0]
                    for msg in pos_msgs[ntype][1:]:
                        agg_pos_msg = agg_pos_msg + msg
                    out_pos_dict[ntype] = pos_dict[ntype] + agg_pos_msg
                else:
                    out_pos_dict[ntype] = pos_dict[ntype]

        return out_x_dict, out_pos_dict


class StructuralEGNNEncoder(nn.Module):
    """
    Bipartite heterogeneous EGNN over the AlphaFold/ESMFold protein pocket
    and the ligand/co-substrate 3D conformers.

    Attributes
    ----------
    protein_pocket_adapter : nn.Sequential
        Projects the ESM2 embedding - gathered at forward time from
        `protein_sequence.esm2_embedding` via the `residue_of` edge, not
        stored per-atom on disk (a residue's embedding would otherwise be
        duplicated across its ~5-15 atoms) - down to `adapter_dim` before
        it rejoins the one-hot/catalytic slice. Keeps the 1280-dim
        pretrained embedding from numerically dominating the small
        structural one-hot once concatenated.
    layers : nn.ModuleList
        `HeteroEGNNLayer` stack. The final layer skips coordinate updates
        (`update_pos=False`): pooling reads only node features, and there
        is no subsequent layer to consume a final coordinate update.
    pool_layers : nn.ModuleDict
        Per-node-type gated-attention readout (`AttentionalAggregation`),
        replacing uniform mean pooling so catalytically relevant pocket
        atoms can dominate the pooled vector.
    """

    def __init__(
        self,
        protein_pocket_onehot_channels: int = PROTEIN_POCKET_ONEHOT_CHANNELS,
        ligand_in_channels: int = 22,
        co_substrate_in_channels: int = 22,
        esm_dim: int = 1280,
        hidden_channels: int = 32,
        num_layers: int = 3,
        dropout: float = 0.3,
        dist_scale_sq: float = 36.0,
        adapter_dim: int = 16,
    ) -> None:
        """
        Parameters
        ----------
        protein_pocket_onehot_channels : int, optional
            Width of the per-atom structural one-hot feature (element,
            residue, backbone, catalytic-site flags) for
            `protein_pocket_atoms`.
        ligand_in_channels : int, optional
            Width of the per-atom structural feature for `ligand_atoms`.
        co_substrate_in_channels : int, optional
            Width of the per-atom structural feature for
            `co_substrate_atoms`.
        esm_dim : int, optional
            Dimensionality of the frozen ESM2 per-residue embedding
            gathered onto pocket atoms via the `residue_of` edge.
        hidden_channels : int, optional
            Width of the per-atom representation used throughout the
            EGNN stack.
        num_layers : int, optional
            Number of `HeteroEGNNLayer` message-passing steps. The final
            layer skips coordinate updates since no subsequent layer
            would consume them.
        dropout : float, optional
            Dropout probability within the projections, EGNN layers, and
            the pocket ESM2 adapter.
        dist_scale_sq : float, optional
            Squared distance (Angstrom^2) used to normalize pairwise
            distances within every `HeteroEGNNLayer`.
        adapter_dim : int, optional
            Width the per-residue ESM2 embedding is projected down to
            before it rejoins the pocket-atom one-hot slice.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.esm_dim = esm_dim
        self._NODE_TYPES: Tuple[str, str, str] = ('protein_pocket_atoms', 'ligand_atoms', 'co_substrate_atoms')

        self.protein_pocket_adapter = nn.Sequential(
            nn.Linear(esm_dim, adapter_dim),
            nn.Dropout(dropout),
        )

        self.projections = nn.ModuleDict({
            'protein_pocket_atoms': nn.Sequential(
                nn.Linear(protein_pocket_onehot_channels + adapter_dim, hidden_channels), nn.Dropout(dropout)
            ),
            'ligand_atoms': nn.Sequential(nn.Linear(ligand_in_channels, hidden_channels), nn.Dropout(dropout)),
            'co_substrate_atoms': nn.Sequential(nn.Linear(co_substrate_in_channels, hidden_channels), nn.Dropout(dropout)),
        })

        edge_routes: List[Tuple[str, str, str]] = [
            ('ligand_atoms', 'interacts_with', 'protein_pocket_atoms'),
            ('protein_pocket_atoms', 'interacts_with', 'ligand_atoms'),
            ('ligand_atoms', 'interacts_with', 'co_substrate_atoms'),
            ('co_substrate_atoms', 'interacts_with', 'ligand_atoms'),
            ('ligand_atoms', 'covalent_bond', 'ligand_atoms'),
            ('co_substrate_atoms', 'covalent_bond', 'co_substrate_atoms'),
        ]
        edge_attr_dims: Dict[Tuple[str, str, str], int] = {
            edge_type: (COVALENT_BOND_FEATURE_DIM if edge_type[1] == 'covalent_bond' else 0)
            for edge_type in edge_routes
        }

        self.layers = nn.ModuleList([
            HeteroEGNNLayer(
                hidden_channels, edge_routes, dropout, dist_scale_sq, edge_attr_dims,
                update_pos=(i < num_layers - 1),
            )
            for i in range(num_layers)
        ])

        self.pool_layers = nn.ModuleDict({
            ntype: AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_channels // 2, 1)
                )
            )
            for ntype in self._NODE_TYPES
        })

    def forward(self, data: HeteroData, num_graphs: int, device: torch.device) -> torch.Tensor:
        """
        Parameters
        ----------
        data : HeteroData
            Batched heterogeneous graph, expected to carry
            `protein_pocket_atoms`/`ligand_atoms`/`co_substrate_atoms` with
            `x`/`pos`, and `interacts_with`/`covalent_bond` edges.
        num_graphs : int
            Number of graphs in the batch, for pooling and the zero-fallback shape.
        device : torch.device
            Device for the zero-fallback tensor.

        Returns
        -------
        torch.Tensor
            Concatenated per-node-type pooled representations, shape
            (num_graphs, hidden_channels * 3). All-zero if
            `protein_pocket_atoms` is absent from the input (e.g. a record
            without 3D structural data).
        """
        if 'protein_pocket_atoms' not in data.node_types:
            return torch.zeros((num_graphs, self.hidden_channels * 3), device=device)

        onehot_part = data['protein_pocket_atoms'].x
        residue_edge = data['protein_pocket_atoms', 'residue_of', 'protein_sequence'].edge_index
        atom_idx, residue_idx = residue_edge[0], residue_edge[1]
        embed_part = onehot_part.new_zeros((onehot_part.size(0), self.esm_dim))
        embed_part[atom_idx] = data['protein_sequence'].esm2_embedding[residue_idx]
        adapted_embed = self.protein_pocket_adapter(embed_part)

        x_dict: Dict[str, torch.Tensor] = {
            'protein_pocket_atoms': self.projections['protein_pocket_atoms'](
                torch.cat([onehot_part, adapted_embed], dim=-1)
            ),
            'ligand_atoms': self.projections['ligand_atoms'](data['ligand_atoms'].x),
            'co_substrate_atoms': self.projections['co_substrate_atoms'](data['co_substrate_atoms'].x),
        }
        pos_dict: Dict[str, torch.Tensor] = {ntype: data[ntype].pos for ntype in x_dict}

        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        for layer in self.layers:
            x_dict, pos_dict = layer(x_dict, pos_dict, edge_index_dict, edge_attr_dict)

        pooled_components = []
        for ntype in self._NODE_TYPES:
            x_n = x_dict[ntype]
            batch_idx = data[ntype].batch if hasattr(data[ntype], 'batch') else torch.zeros(
                x_n.size(0), dtype=torch.long, device=x_n.device
            )
            pooled = self.pool_layers[ntype](x_n, index=batch_idx, dim_size=num_graphs)
            pooled_components.append(pooled)

        return torch.cat(pooled_components, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
#  Multimodal Fusion
# ═══════════════════════════════════════════════════════════════════════════

class MultimodalEncoder(nn.Module):
    """
    Main encoder fusing the sequence/2D-graph adapter with the 3D structural EGNN.

    Combines a frozen-embedding sequence/2D-graph branch (protein
    adapter, ligand/co-substrate D-MPNNs) with a 3D structural branch
    (StructuralEGNNEncoder) over the same molecules' AlphaFold/ESMFold
    conformers, then predicts the forward and reverse rate constants
    from the fused representation. Outputs k_1, k_reverse, and the
    concatenated representation for the thermodynamics head.

    Attributes
    ----------
    protein_encoder : ProteinAdapterEncoder
        Adapts frozen per-residue ESM2 embeddings and pools a
        catalytic-site representation.
    ligand_encoder : DMPNNEncoder
        D-MPNN over the ligand's 2D molecular graph.
    co_sub_encoder : DMPNNEncoder
        D-MPNN over the co-substrate's 2D molecular graph (shares
        architecture, not weights, with `ligand_encoder`).
    mol_proj : nn.Linear
        Projects a molecule's whole-molecule ChemBERTa embedding down to
        `hidden_channels`.
    mol_combine : nn.Sequential
        Fuses a molecule's D-MPNN atom-pooled representation with its
        projected ChemBERTa embedding. Shared between the ligand and
        co-substrate branches.
    substrate_pool : SubstrateConditionedPooling
        Attention-pools protein residues conditioned on a substrate's
        pooled representation. Shared between the ligand and
        co-substrate branches.
    protein_pool_combine : nn.Sequential
        Fuses the catalytic-site-pooled and both substrate-conditioned
        protein representations into a single protein representation.
    structural_encoder : StructuralEGNNEncoder
        3D heterogeneous EGNN over the protein pocket and ligand/
        co-substrate conformers.
    kinetics_head : nn.Sequential
        Maps the fused representation to the two raw logits underlying
        k_1 and k_reverse.
    mutation_feature_scale : torch.Tensor
        Non-trainable buffer of per-channel scale factors bringing the
        mutation descriptor's heterogeneous units onto a comparable
        range before concatenation.
    concat_dim : int
        Width of the fused representation passed to `kinetics_head`;
        the single source of truth for downstream head widths
        (EyringArrheniusLayer, BaselineNNModel).
    hidden_channels : int
        Width of the 2D-branch representations.
    """
    def __init__(
        self,
        ligand_in_channels: int = 22,
        co_substrate_in_channels: int = 22,
        esm_dim: int = 1280,
        hidden_channels: int = 64,
        dropout: float = 0.3,
        head_dropout: float = 0.1,
        protein_pocket_onehot_channels: int = PROTEIN_POCKET_ONEHOT_CHANNELS,
        egnn_hidden_channels: int = 32,
        egnn_num_layers: int = 3,
        egnn_adapter_dim: int = 16,
        egnn_dist_scale_sq: float = 36.0,
    ):
        """
        Parameters
        ----------
        ligand_in_channels : int, optional
            Width of the ligand atom-level structural feature.
        co_substrate_in_channels : int, optional
            Width of the co-substrate atom-level structural feature.
        esm_dim : int, optional
            Dimensionality of the frozen ESM2 per-residue embedding.
        hidden_channels : int, optional
            Width of the 2D-branch representations (protein, ligand,
            co-substrate).
        dropout : float, optional
            Dropout probability applied throughout the 2D and 3D
            branches.
        head_dropout : float, optional
            Dropout probability within `kinetics_head`.
        protein_pocket_onehot_channels : int, optional
            Width of the 3D-branch protein-pocket atom one-hot feature,
            forwarded to `StructuralEGNNEncoder`.
        egnn_hidden_channels : int, optional
            Width of the 3D-branch (StructuralEGNNEncoder) representations.
        egnn_num_layers : int, optional
            Number of `HeteroEGNNLayer` steps in `structural_encoder`.
        egnn_adapter_dim : int, optional
            Width the 3D branch projects the pocket ESM2 embedding down
            to, forwarded to `StructuralEGNNEncoder`.
        egnn_dist_scale_sq : float, optional
            Squared distance (Angstrom^2) normalizing pairwise distances
            within `structural_encoder`.
        """
        super().__init__()

        # --- 2D: sequence adapter + D-MPNN ---
        self.protein_encoder = ProteinAdapterEncoder(
            esm_dim=esm_dim, hidden_channels=hidden_channels, dropout=dropout
        )
        self.ligand_encoder = DMPNNEncoder(
            in_channels=ligand_in_channels, hidden_channels=hidden_channels, dropout=dropout
        )
        self.co_sub_encoder = DMPNNEncoder(
            in_channels=co_substrate_in_channels, hidden_channels=hidden_channels, dropout=dropout
        )

        self.mol_proj = nn.Linear(CHEMBERTA_EMBED_DIM, hidden_channels)
        self.mol_combine = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.substrate_pool = SubstrateConditionedPooling(hidden_channels, dropout=dropout)
        self.protein_pool_combine = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # --- 3D: structural EGNN ---
        self.structural_encoder = StructuralEGNNEncoder(
            protein_pocket_onehot_channels=protein_pocket_onehot_channels,
            ligand_in_channels=ligand_in_channels,
            co_substrate_in_channels=co_substrate_in_channels,
            esm_dim=esm_dim,
            hidden_channels=egnn_hidden_channels,
            num_layers=egnn_num_layers,
            dropout=dropout,
            dist_scale_sq=egnn_dist_scale_sq,
            adapter_dim=egnn_adapter_dim,
        )

        # concat_rep = 2D pooled reps (hidden_channels * 3) + 3D pooled reps
        # (egnn_hidden_channels * 3) + pH + temperature + mutation descriptor.
        # Single source of truth for every downstream head width
        # (EyringArrheniusLayer, BaselineNNModel).
        self.concat_dim = hidden_channels * 3 + egnn_hidden_channels * 3 + 2 + MUTATION_FEATURE_DIM
        self.kinetics_head = nn.Sequential(
            nn.Linear(self.concat_dim, hidden_channels),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_channels, 2)  # k_1, k_reverse
        )

        self.register_buffer(
            "mutation_feature_scale",
            torch.tensor([1/3, 1/5, 1/5, 1/100, 1/2, 1/5, 1/3], dtype=torch.float),
        )

        # sigmoid(b_0) * 1e10 ~ 1e6 (k_1); sigmoid(b_1) * 6.21e12 ~ 100 (k_reverse @ 298K).
        with torch.no_grad():
            final_linear = cast(nn.Linear, self.kinetics_head[-1])
            if final_linear.bias is not None:
                final_linear.bias[0] = -9.21
                final_linear.bias[1] = -24.84

        self.hidden_channels = hidden_channels

    def forward(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        data : HeteroData
            Batched heterogeneous graph carrying:
            `protein_sequence` (`aa_indices`, `esm2_embedding`,
            `catalytic_mask`); `ligand_atoms`/`co_substrate_atoms` (`x`,
            `pos`) with `covalent_bond` self-edges; `ligand_embedding`/
            `co_substrate_embedding` whole-molecule ChemBERTa vectors
            (the co-substrate embedding is the zero vector when absent);
            optionally `protein_pocket_atoms` and `interacts_with` edges
            for the 3D branch; optional `pH`, `temperature` (K), and
            `mutation_features` (defaulting to pH 7.0, 298.15 K, and an
            all-zero descriptor respectively when absent).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            k_1 : shape (num_graphs,), the forward rate constant
            (M^-1 s^-1), sigmoid-bounded by the diffusion limit
            (1e10 M^-1 s^-1). k_reverse : shape (num_graphs,), the
            reverse rate constant (s^-1), sigmoid-bounded by the
            Eyring/transition-state-theory limit (k_B * T / h).
            concat_rep : shape (num_graphs, concat_dim), the fused
            representation consumed by the thermodynamics layer.
        """
        # --- 2D branch ---
        p_seq = data["protein_sequence"]
        p_batch = p_seq.batch if hasattr(p_seq, "batch") else torch.zeros(p_seq.aa_indices.size(0), dtype=torch.long, device=p_seq.aa_indices.device)
        p_catalytic_pooled, p_h, p_mask = self.protein_encoder(p_seq.aa_indices, p_seq.esm2_embedding, p_seq.catalytic_mask, p_batch)
        num_graphs = p_catalytic_pooled.size(0)

        l_nodes = data["ligand_atoms"]
        l_batch = l_nodes.batch if hasattr(l_nodes, "batch") else torch.zeros(l_nodes.x.size(0), dtype=torch.long, device=l_nodes.x.device)
        l_edges = data["ligand_atoms", "covalent_bond", "ligand_atoms"]
        l_rep_atoms = self.ligand_encoder(l_nodes.x, l_edges.edge_index, l_edges.edge_attr, l_batch, num_graphs=num_graphs)
        l_rep = self.mol_combine(torch.cat([l_rep_atoms, self.mol_proj(data.ligand_embedding)], dim=-1))

        c_nodes = data["co_substrate_atoms"]
        c_batch = c_nodes.batch if hasattr(c_nodes, "batch") else torch.zeros(c_nodes.x.size(0), dtype=torch.long, device=c_nodes.x.device)
        if c_nodes.x.size(0) > 0:
            c_edges = data["co_substrate_atoms", "covalent_bond", "co_substrate_atoms"]
            c_rep_atoms = self.co_sub_encoder(c_nodes.x, c_edges.edge_index, c_edges.edge_attr, c_batch, num_graphs=num_graphs)
        else:
            c_rep_atoms = torch.zeros((num_graphs, self.hidden_channels), dtype=torch.float, device=p_catalytic_pooled.device)

        co_sub_embedding = data.co_substrate_embedding
        has_co_sub = (co_sub_embedding.abs().sum(dim=-1, keepdim=True) > 0).float()
        c_rep = self.mol_combine(torch.cat([c_rep_atoms, self.mol_proj(co_sub_embedding)], dim=-1)) * has_co_sub

        p_ligand_cond = self.substrate_pool(p_h, p_mask, l_rep)
        p_co_sub_cond = self.substrate_pool(p_h, p_mask, c_rep)
        p_rep = self.protein_pool_combine(torch.cat([p_catalytic_pooled, p_ligand_cond, p_co_sub_cond], dim=-1))

        # --- 3D branch ---
        structural_rep = self.structural_encoder(data, num_graphs=num_graphs, device=p_rep.device)

        # --- Metadata ---
        if hasattr(data, 'pH'):
            pH = data.pH.view(-1, 1)
        else:
            pH = torch.full((p_rep.size(0), 1), 7.0, device=p_rep.device)

        if hasattr(data, 'temperature'):
            T = data.temperature.view(-1, 1)
        else:
            T = torch.full((p_rep.size(0), 1), 298.15, device=p_rep.device)

        T_scaled = T / 298.15
        pH_scaled = (pH - 7.0) / 7.0

        if hasattr(data, 'mutation_features'):
            mutation_features = data.mutation_features.view(-1, MUTATION_FEATURE_DIM)
        else:
            mutation_features = torch.zeros((p_rep.size(0), MUTATION_FEATURE_DIM), device=p_rep.device)

        mutation_features_scaled = mutation_features * cast(torch.Tensor, self.mutation_feature_scale)

        concat_rep = torch.cat(
            [p_rep, l_rep, c_rep, structural_rep, pH_scaled, T_scaled, mutation_features_scaled], dim=-1
        )

        kinetics_out = self.kinetics_head(concat_rep)

        # Diffusion limit on k_1: 10^10 M^-1 s^-1.
        k_1 = 1e10 * torch.sigmoid(kinetics_out[:, 0])

        # Eyring/TST limit on k_reverse: (k_B * T) / h.
        kb_over_h = KB_SI / H_SI
        T_val = T.squeeze(-1)
        k_reverse = (kb_over_h * T_val) * torch.sigmoid(kinetics_out[:, 1])

        return k_1, k_reverse, concat_rep
