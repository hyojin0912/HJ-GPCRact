# src/model.py
import torch
import torch.nn as nn
from torch_geometric.utils import subgraph, to_dense_batch
from torch_scatter import scatter_mean

from .modules import EGNN, EGNN_Gated, EGNN_GlobalResidual, EGNN_Gated_GlobalResidual


# Helper function to create an encoder based on config
def create_encoder(config, in_dim, hidden_dim):
    encoder_type = config['type']
    n_layers = config['n_layers']
    
    # Common kwargs for all encoder types
    kwargs = {
        'in_node_nf': in_dim,
        'hidden_nf': hidden_dim,
        'out_node_nf': hidden_dim,
        'n_layers': n_layers,
        'attention': True,
        'tanh': True
    }

    if encoder_type == 'base':
        # NOTE: The original EGNN does not have a global residual connection
        return EGNN(**kwargs)
    elif encoder_type == 'residual':
        return EGNN_GlobalResidual(**kwargs)
    elif encoder_type == 'gated':
        # NOTE: The gated EGNN does not have a global residual connection by default
        return EGNN_Gated(**kwargs)
    elif encoder_type == 'gated_residual':
        return EGNN_Gated_GlobalResidual(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

class DAGN_HybridModel(nn.Module):
    def __init__(self, protein_in_dim_clean, protein_in_dim_full, ligand_in_dim, hidden_dim,
                 protein_config, ligand_config, element_embedding_dim,
                 n_attn_heads, dropout, propagation_attention_layers,
                 num_gpcr_classes, num_gpcr_families):
        super().__init__()
        # --- Module 1: Interaction ---
        self.element_embedding = nn.Embedding(num_embeddings=6, embedding_dim=element_embedding_dim)
        # Separate encoder for the 'clean' binding site subgraph
        self.bs_encoder = create_encoder(protein_config, protein_in_dim_clean, hidden_dim)
        self.ligand_encoder = create_encoder(ligand_config, ligand_in_dim, hidden_dim)
        self.p_to_l_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_attn_heads, dropout=dropout, batch_first=True)
        self.l_to_p_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_attn_heads, dropout=dropout, batch_first=True)

        # --- ✅ Module 2: Local Propagation via EGNN ---
        propagation_config = {"type": protein_config['type'], "n_layers": protein_config['n_layers']} # Use PROTEIN_LAYERS for this
        self.local_propagation_encoder = create_encoder(propagation_config, hidden_dim, hidden_dim)

        # ✅ SAVE hidden_dim AS INSTANCE VARIABLE (prevents NameError)
        self.hidden_dim = hidden_dim

        # ✅ REPLACE single embedding layer with role-specific layers
        self.protein_embedding_ca = nn.Linear(protein_in_dim_full, hidden_dim)
        self.protein_embedding_sc = nn.Linear(protein_in_dim_full, hidden_dim)

        # --- Module 3: Global Propagation (Transformer) ---
        # This is the core of Experiment 1, replacing the propagation_encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_attn_heads,
            dim_feedforward=hidden_dim * 4, # Standard practice
            dropout=dropout,
            batch_first=True
        )
        self.global_integration_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=propagation_attention_layers
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        # --- Module 4: Prediction Heads ---
        # --- ADDED: Binding Head (Auxiliary task) ---
        self.binding_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Takes concatenated protein-ligand vector
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        # Primary task head: MODIFIED: Activity head now outputs 3 classes (Non-binder, Agonist, Antagonist)
        self.activity_type_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2) # Outputs 2 logits: [Antagonist, Agonist]
        )

    def forward(self, protein_batch, ligand_batch):
        # =========================================================================
        # STAGE 1: BINDING INTERACTION & SIGNAL GENERATION (Unchanged)
        # Correctly computes the ligand's "signal" in the 128-dim latent space.
        # =========================================================================
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

        # --- MODIFICATION START: Calculate binding prediction ---
        p_updated_padded_h, _ = self.p_to_l_attention(query=padded_bs_h, key=padded_l_h, value=padded_l_h, key_padding_mask=~l_padding_mask)
        l_updated_padded_h, _ = self.l_to_p_attention(query=padded_l_h, key=padded_bs_h, value=padded_bs_h, key_padding_mask=~bs_padding_mask)

        p_updated_padded_h[~bs_padding_mask] = 0
        l_updated_padded_h[~l_padding_mask] = 0

        # Pool protein and ligand vectors from the interaction
        protein_interaction_vector = p_updated_padded_h.sum(dim=1) / bs_padding_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        ligand_interaction_vector = l_updated_padded_h.sum(dim=1) / l_padding_mask.sum(dim=1).unsqueeze(-1).clamp(min=1)
        
        # Concatenate and predict binding
        interaction_vector = torch.cat([protein_interaction_vector, ligand_interaction_vector], dim=1)
        binding_logit = self.binding_head(interaction_vector) # Use a consistent vector
        
        # Ligand signal for Stage 2 remains the same
        ligand_signal = p_updated_padded_h[bs_padding_mask]

        # =========================================================================
        # STAGE 2: ALLOSTERIC PROPAGATION (✅ REVISED LOGIC FOR CORRECT INFO FLOW)
        # =========================================================================
        with torch.no_grad():
            gate_weight = torch.sigmoid(binding_logit).squeeze(-1)
            node_gate_weight = gate_weight[protein_batch.batch].unsqueeze(-1)
        
        gated_ligand_signal = ligand_signal * node_gate_weight[bs_mask]
        
        p_features_full = torch.cat([protein_batch.x_float_full, self.element_embedding(protein_batch.x_elem)], dim=1)
        h = torch.zeros(p_features_full.size(0), self.hidden_dim, device=p_features_full.device)
        
        ca_mask = (protein_batch.node_roles == 0)
        sc_mask = (protein_batch.node_roles == 1)
        h[ca_mask] = self.protein_embedding_ca(p_features_full[ca_mask]).to(h.dtype)
        h[sc_mask] = self.protein_embedding_sc(p_features_full[sc_mask]).to(h.dtype)

        predicted_coords = protein_batch.pos.clone() # Start with initial coords
        h_initial_for_residual = h.clone()
        
        for i in range(self.local_propagation_encoder.n_layers):
            h[bs_mask] = h[bs_mask] + gated_ligand_signal
            h, predicted_coords, _ = self.local_propagation_encoder._modules[f"gcl_{i}"](
                h, protein_batch.edge_index, predicted_coords
            )
        
        h = h + h_initial_for_residual
        h_after_egnn = self.local_propagation_encoder.embedding_out(h)
        h_after_egnn = self.final_norm(h_after_egnn)

        # =========================================================================
        # STAGE 3: GLOBAL INTEGRATION VIA TRANSFORMER
        # =========================================================================
        # The Transformer takes the EGNN's output to find long-range dependencies.
        padded_h, padding_mask = to_dense_batch(h_after_egnn, protein_batch.batch)
        transformer_mask = ~padding_mask
        final_padded_h = self.global_integration_transformer(padded_h, src_key_padding_mask=transformer_mask)
        final_h_p = final_padded_h[padding_mask]

        # =========================================================================
        # STAGE 3: ACTIVITY PREDICTION
        # =========================================================================
        # 1. Pool the node features to get a graph-level protein representation
        pooled_protein_vector = scatter_mean(final_h_p, protein_batch.batch, dim=0)

        # 2. Pool the final ligand representation from Stage 1
        final_combined_vector = torch.cat([pooled_protein_vector, ligand_interaction_vector], dim=1)
        # 3. Feed the combined vector to the modified activity head
        activity_type_logit = self.activity_type_head(final_combined_vector)
        

        return binding_logit, activity_type_logit, predicted_coords
