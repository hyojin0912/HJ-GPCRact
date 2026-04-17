import torch
import torch.nn as nn
from torch_geometric.utils import subgraph, to_dense_batch
from torch_scatter import scatter_mean
from .layers import create_encoder

class GPCRact_Model(nn.Module):
    """
    GPCRact Hierarchical Model.
    Features:
    1. Interaction Module (Cross-Attention)
    2. Local Propagation (EGNN with Gating)
    3. Global Integration (Transformer)
    4. Dual Heads (Binding & Activity)
    """
    def __init__(self, protein_in_dim_clean, protein_in_dim_full, ligand_in_dim, hidden_dim,
                 protein_config, ligand_config, element_embedding_dim,
                 n_attn_heads, dropout, propagation_attention_layers,
                 use_gpcr_cf_embed=True, num_gpcr_classes=0, num_gpcr_families=0
                 ):
        super().__init__()
        
        # --- Module 1: Interaction ---
        self.element_embedding = nn.Embedding(num_embeddings=6, embedding_dim=element_embedding_dim)
        self.bs_encoder = create_encoder(protein_config, protein_in_dim_clean, hidden_dim)
        self.ligand_encoder = create_encoder(ligand_config, ligand_in_dim, hidden_dim)
        self.p_to_l_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_attn_heads, dropout=dropout, batch_first=True)
        self.l_to_p_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_attn_heads, dropout=dropout, batch_first=True)

        # --- Module 2: Local Propagation via EGNN ---
        propagation_config = {"type": protein_config['type'], "n_layers": protein_config['n_layers']}
        self.local_propagation_encoder = create_encoder(propagation_config, hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        # Role-specific embeddings
        self.protein_embedding_ca = nn.Linear(protein_in_dim_full, hidden_dim)
        self.protein_embedding_sc = nn.Linear(protein_in_dim_full, hidden_dim)

        # --- Module 3: Global Propagation (Transformer) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_attn_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.global_integration_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=propagation_attention_layers
        )
        self.use_gpcr_cf_embed = use_gpcr_cf_embed
        if self.use_gpcr_cf_embed:
            # +1 slot for "unknown" (-1) mapped to padding_idx
            self.gpcr_class_embedding = nn.Embedding(
                num_embeddings=num_gpcr_classes + 1,
                embedding_dim=hidden_dim,
                padding_idx=num_gpcr_classes
            )
            self.gpcr_family_embedding = nn.Embedding(
                num_embeddings=num_gpcr_families + 1,
                embedding_dim=hidden_dim,
                padding_idx=num_gpcr_families
            )
        else:
            self.gpcr_class_embedding = None
            self.gpcr_family_embedding = None

        self.final_norm = nn.LayerNorm(hidden_dim)

        # --- Module 4: Prediction Heads (hierarchical two-stage) ---
        # Stage 1: binary binding (non-binder vs binder); used as the
        # gatekeeper for Stage 2 and as a gate on allosteric propagation.
        self.binding_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        # Stage 2: binary activity type (antagonist vs agonist);
        # trained on binder-labeled samples only.
        self.activity_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, protein_batch, ligand_batch):
        # 1. Interaction Stage
        bs_mask = protein_batch.bs_mask
        bs_edge_index, _ = subgraph(bs_mask, protein_batch.edge_index, relabel_nodes=True, num_nodes=protein_batch.num_nodes)
        bs_pos = protein_batch.pos[bs_mask]

        p_features_bs_clean = torch.cat([
            protein_batch.x_float_clean[bs_mask],
            self.element_embedding(protein_batch.x_elem[bs_mask])
        ], dim=1)
        
        h_p_bs, _ = self.bs_encoder(p_features_bs_clean, bs_pos, bs_edge_index)
        h_l, _ = self.ligand_encoder(ligand_batch.x, ligand_batch.pos, ligand_batch.edge_index)

        padded_bs_h, bs_padding_mask = to_dense_batch(h_p_bs, protein_batch.batch[bs_mask])
        padded_l_h, l_padding_mask = to_dense_batch(h_l, ligand_batch.batch)

        p_updated_padded_h, _ = self.p_to_l_attention(query=padded_bs_h, key=padded_l_h, value=padded_l_h, key_padding_mask=~l_padding_mask)
        l_updated_padded_h, _ = self.l_to_p_attention(query=padded_l_h, key=padded_bs_h, value=padded_bs_h, key_padding_mask=~bs_padding_mask)

        p_updated_padded_h[~bs_padding_mask] = 0
        l_updated_padded_h[~l_padding_mask] = 0

        protein_interaction_vector = p_updated_padded_h.sum(dim=1) / bs_padding_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        ligand_interaction_vector = l_updated_padded_h.sum(dim=1) / l_padding_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        
        interaction_vector = torch.cat([protein_interaction_vector, ligand_interaction_vector], dim=1)
        binding_logit = self.binding_head(interaction_vector)
        
        ligand_signal = p_updated_padded_h[bs_padding_mask]

        # 2. Propagation Stage
        with torch.no_grad():
            # reshape(-1) is robust for B=1 (squeeze(-1) would 0-dim collapse an already-1D tensor).
            gate_weight = torch.sigmoid(binding_logit).reshape(-1)
            node_gate_weight = gate_weight[protein_batch.batch].unsqueeze(-1)
        
        gated_ligand_signal = ligand_signal * node_gate_weight[bs_mask]
        
        p_features_full = torch.cat([protein_batch.x_float_full, self.element_embedding(protein_batch.x_elem)], dim=1)
        h = torch.zeros(p_features_full.size(0), self.hidden_dim, device=p_features_full.device)
        
        ca_mask = (protein_batch.node_roles == 0)
        sc_mask = (protein_batch.node_roles == 1)
        h[ca_mask] = self.protein_embedding_ca(p_features_full[ca_mask]).to(h.dtype)
        h[sc_mask] = self.protein_embedding_sc(p_features_full[sc_mask]).to(h.dtype)

        coord = protein_batch.pos.clone()
        h_initial_for_residual = h.clone()

        for i in range(self.local_propagation_encoder.n_layers):
            h[bs_mask] = h[bs_mask] + gated_ligand_signal
            h, coord, _ = self.local_propagation_encoder._modules[f"gcl_{i}"](h, protein_batch.edge_index, coord)
        
        h = h + h_initial_for_residual
        h_after_egnn = self.local_propagation_encoder.embedding_out(h)
        h_after_egnn = self.final_norm(h_after_egnn)

        # 3. Global Integration Stage (Transformer)
        padded_h, padding_mask = to_dense_batch(h_after_egnn, protein_batch.batch)
        transformer_mask = ~padding_mask
        final_padded_h = self.global_integration_transformer(padded_h, src_key_padding_mask=transformer_mask)
        final_h_p = final_padded_h[padding_mask]

        # 4. Activity Prediction
        pooled_protein_vector = scatter_mean(final_h_p, protein_batch.batch, dim=0)

        if self.use_gpcr_cf_embed:
            # Expect graph-level tensors shaped [batch_size, 1] after batching
            class_ids = protein_batch.gpcr_class_id.view(-1).clone()
            family_ids = protein_batch.gpcr_family_id.view(-1).clone()

            # Map -1 -> padding idx
            class_ids[class_ids == -1] = self.gpcr_class_embedding.padding_idx
            family_ids[family_ids == -1] = self.gpcr_family_embedding.padding_idx

            class_emb = self.gpcr_class_embedding(class_ids)
            family_emb = self.gpcr_family_embedding(family_ids)

            pooled_protein_vector = pooled_protein_vector + class_emb + family_emb

        final_combined_vector = torch.cat([pooled_protein_vector, ligand_interaction_vector], dim=1)
        activity_type_logit = self.activity_type_head(final_combined_vector)
        

        return binding_logit, activity_type_logit



