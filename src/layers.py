import torch
import torch.nn as nn
from torch_scatter import scatter_add

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom scatter add operation."""
    out = data.new_zeros((num_segments, data.size(1)))
    scatter_add(data, segment_ids, out=out, dim=0)
    return out

class E_GCL(nn.Module):
    """
    Equivariant Graph Convolutional Layer (E_GCL).
    Includes stabilization modification in coord_model.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), 
                 residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
            
        self.node_norm = nn.LayerNorm(output_nf)
        
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        
        coord_mlp_list = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer]
        if self.tanh:
            coord_mlp_list.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp_list)
        
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)
        if self.normalize:
            norm = torch.sqrt(radial + self.epsilon)
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def edge_model(self, h_row, h_col, radial, edge_attr):
        if edge_attr is not None:
            out = torch.cat([h_row, h_col, radial, edge_attr], dim=1)
        else:
            out = torch.cat([h_row, h_col, radial], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_feat, node_attr=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        # --- STABILIZATION MODIFICATION ---
        coord_diff_normalized = coord_diff / (torch.norm(coord_diff, dim=-1, keepdim=True) + self.epsilon)
        trans = coord_diff_normalized * self.coord_mlp(edge_feat)
        # ----------------------------------
        
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0)) / (unsorted_segment_sum(torch.ones_like(trans), row, num_segments=coord.size(0)) + 1e-8)
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        h_row, h_col = h[row], h[col]
        e_ij = self.edge_model(h_row, h_col, radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, e_ij)
        h = self.node_model(h, edge_index, e_ij, node_attr)
        h = self.node_norm(h)
        return h, coord, e_ij

class E_GCL_Gated(E_GCL):
    """E_GCL with Gating mechanism."""
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), 
                 residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL_Gated, self).__init__(
            input_nf, output_nf, hidden_nf, edges_in_d, act_fn,
            residual, attention, normalize, coords_agg, tanh
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
            nn.Sigmoid()
        )
        self.node_norm = nn.LayerNorm(output_nf)

    def node_model(self, x, edge_index, edge_feat, node_attr=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=x.size(0))
        
        if node_attr is not None:
            agg_cat = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg_cat = torch.cat([x, agg], dim=1)

        update_val = self.node_mlp(agg_cat)
        gate_val = self.gate_mlp(agg_cat)
        
        if self.residual:
            out = x * gate_val + (x + update_val) * (1 - gate_val)
        else:
            out = x * gate_val + update_val * (1 - gate_val)
        out = self.node_norm(out)
        return out

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, n_layers=4, 
                 residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(hidden_nf, hidden_nf, hidden_nf, edges_in_d=in_edge_nf,
                                                residual=residual, attention=attention,
                                                normalize=normalize, coords_agg=coords_agg, tanh=tanh))
    
    def forward(self, h, coord, edge_index, edge_attr=None):
        device = self.embedding_in.weight.device
        # h, coord, edge_index = h.to(device), coord.to(device), edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, coord, _ = self._modules["gcl_%d" % i](h, edge_index, coord, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, coord

class EGNN_Gated(EGNN):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, n_layers=4, 
                 residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_out = nn.Linear(hidden_nf, out_node_nf)
        
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_Gated(
                input_nf=hidden_nf, output_nf=hidden_nf, hidden_nf=hidden_nf, 
                edges_in_d=in_edge_nf, residual=residual, attention=attention,
                normalize=normalize, coords_agg=coords_agg, tanh=tanh
            ))

class EGNN_GlobalResidual(EGNN):
    def forward(self, h, coord, edge_index, edge_attr=None):
        h_initial = self.embedding_in(h)
        h = h_initial
        for i in range(self.n_layers):
            h, coord, _ = self._modules[f"gcl_{i}"](h, edge_index, coord, edge_attr=edge_attr)
        h = h + h_initial
        h = self.embedding_out(h)
        return h, coord

class EGNN_Gated_GlobalResidual(EGNN_Gated):
    def forward(self, h, coord, edge_index, edge_attr=None):
        device = self.embedding_in.weight.device
        # h, coord, edge_index = h.to(device), coord.to(device), edge_index.to(device)
        if edge_attr is not None: edge_attr = edge_attr.to(device)

        h_initial = self.embedding_in(h)
        h = h_initial

        for i in range(0, self.n_layers):
            h, coord, _ = self._modules[f"gcl_{i}"](h, edge_index, coord, edge_attr=edge_attr)
        
        h = h + h_initial
        h = self.embedding_out(h)
        return h, coord

def create_encoder(config, in_dim, hidden_dim):
    """Factory function for Encoders."""
    encoder_type = config['type']
    n_layers = config['n_layers']
    kwargs = {
        'in_node_nf': in_dim, 'hidden_nf': hidden_dim, 'out_node_nf': hidden_dim,
        'n_layers': n_layers, 'attention': True, 'tanh': True
    }

    if encoder_type == 'base': return EGNN(**kwargs)
    elif encoder_type == 'residual': return EGNN_GlobalResidual(**kwargs)
    elif encoder_type == 'gated': return EGNN_Gated(**kwargs)
    elif encoder_type == 'gated_residual': return EGNN_Gated_GlobalResidual(**kwargs)
    else: raise ValueError(f"Unknown encoder type: {encoder_type}")
