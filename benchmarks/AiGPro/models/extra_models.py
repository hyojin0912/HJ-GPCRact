import lightning as L
import torch
import torch.nn as nn
from rich.console import Console
from aigpro.models import component
from aigpro.models.gpcr_mambed import MixerModel
from aigpro.utils import logger
console = Console()
log = logger.get_logger()
log.setLevel("DEBUG")

class BestGPCR(L.LightningModule):
    """Transformer model for kinase data."""
    def __init__(self, test=None):  
        super().__init__()
        self.LAYER_MULTIPLIER = 2
        down_sample = False
        conv_stride = 1
        print(test)
        self.complete_conv = component.Conv1DBlock(
            1,
            [
                2,
                8,
                16,
                32,
            ],
            stride=conv_stride,
            dilation=[1, 2, 4, 8],
            dropout=0.1,
            down_smaple=down_sample,
            normalization_end=True,
        )
        self.complete_conv1 = component.Conv1DBlock(
            1,
            [
                8,
            ],
            stride=conv_stride,
            dilation=[
                1,
            ],
            dropout=0.1,
            down_smaple=True,
            normalization_end=True,
        )
        
        self.prot_emb = 32
        self.prot_dim_dmodel = 1024
        self.prot_dim_layer = 4  
        self.prot_nheads = 16
        self.prot_dim_dmffeat = self.LAYER_MULTIPLIER * self.prot_dim_dmodel
        self.prot_dim_dp = 0.1
        self.prot_ff_in = 2560  
        self.prot_ff_dmffeat = self.LAYER_MULTIPLIER * self.prot_ff_in
        self.prot_ff_fout = 2048
        self.pdb_sequence_one_hot_conv = component.Conv1DBlock(
            self.prot_emb,
            [8, 16, 32, 64, 128, 32],
            stride=conv_stride,
            dropout=0.1,
            dilation=[1, 2, 4, 8, 16, 32, 32],
            down_smaple=down_sample,
            normalization_end=True,
        )
        self.emb_prot = nn.Embedding(25, self.prot_emb)
        self.prot_mha = MixerModel(
            self.prot_emb,
            n_layer=16,
            rms_norm=True,
            fused_add_norm=True,
            
            
            
        )
        self.pos_prot = component.PositionalEncoding1D(self.prot_emb)
        self.smi_emb = 32
        self.smi_dim_dmodel = 100
        self.smi_dim_layer = 4
        self.smi_nheads = 16
        self.smi_dim_dmffeat = self.LAYER_MULTIPLIER * self.prot_dim_dmodel
        self.smi_dim_dp = 0.1
        
        self.smi_conv = component.Conv1DBlock(
            self.smi_emb,
            [16, 32, 64, 32],
            stride=conv_stride,
            dropout=0.1,
            dilation=[1, 2, 4, 8],
            down_smaple=False,
        )
        self.emb_smi = nn.Embedding(575, self.smi_emb)
        self.pos_smi = component.PositionalEncoding1D(self.smi_emb)
        self.smile_mha = MixerModel(
            self.smi_emb,
            n_layer=12,
            rms_norm=True,
            fused_add_norm=True,
            
            
            
        )
        self.normalize_smile_desc = torch.nn.BatchNorm1d(170)
        self.smi_act = torch.nn.LeakyReLU(0.3)
        num = 2730 + 366  
        num = 2048
        self.complete_layer_norm = nn.LayerNorm(num)
        num = num + 1024
        self.complete_dropout = nn.Dropout(0.1)
        
        self.ff = nn.Sequential(
            nn.LayerNorm(num),
            nn.Linear(num, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        reg_in = 2368  
        self.regresor = nn.Sequential(
            nn.LayerNorm(reg_in),
            nn.Linear(reg_in, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            
        )
        
        
        self.last_mha = MixerModel(
            reg_in,
            n_layer=4,
            rms_norm=True,
            fused_add_norm=True,
        )
        self.fp_projection = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 366),
            
            
        )
        bi_out = 128
        self.constant_label = nn.Embedding(2, 32)
        self.bilinear = nn.Bilinear(64, 32, bi_out)
        num_input_channels: int = bi_out
        base_channel_size: int = 32
        act_fn: object = nn.GELU
        c_hid = base_channel_size
        self.cond_encoder = nn.Sequential(
            
            nn.Conv1d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(c_hid, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(),
            nn.LayerNorm(4 * c_hid),
            
        )
    def forward(self, x):  
        protein_sequence, ligand_smile, smi_desc, charge_fp, constant_label, emb = (
            x[0],
            x[1].long().squeeze(1),
            
            x[2].float(),
            x[3],
            x[4],
            x[5],
        )
        log.debug(f"{protein_sequence.shape},{ligand_smile.shape}, <=== shapwe ")
        log.debug(f"{protein_sequence.dtype},{ligand_smile.dtype}, ")
        batch_size = ligand_smile.size(0)
        log.debug(f"{charge_fp.shape}:charge_fp")
        smi = self.emb_smi(ligand_smile)
        log.debug(f" {smi.shape}: smi")
        smi = self.pos_smi(smi)
        smi = smi.permute(0, 2, 1)
        smi = self.smi_conv(smi)
        prot = self.emb_prot(protein_sequence)
        log.debug(f"{prot.shape}:prot")
        prot = self.pos_prot(prot)
        prot = prot.permute(0, 2, 1)
        prot = self.pdb_sequence_one_hot_conv(prot)
        log.debug(f"{prot.shape}:prot")
        log.debug(f"smae {smi.shape} {prot.shape}")
        prot = self.prot_mha(prot)
        log.debug(f"{prot.shape}:prosdst")
        log.debug(f" {smi.shape}: smi")
        smi = self.smile_mha(smi)
        smi = smi.view(batch_size, 1, -1)
        prot = prot.view(batch_size, 1, -1)
        log.debug(f"{prot.shape}:protddsasa")
        log.debug(f"{prot.shape}:prot asdasd")
        log.debug(f"{prot.shape}:prot asd")
        smi_desc = self.normalize_smile_desc(smi_desc.view(batch_size, -1)).view(batch_size, 1, -1)
        smi_desc = self.smi_act(smi_desc)
        complete = torch.cat(
            [
                smi,
                prot,
            ],
            dim=-1,
        ).view(batch_size, 1, -1)
        log.debug(f"{charge_fp.shape}:charge_fp")
        mfl_projection = self.fp_projection(charge_fp)
        log.debug(f"{mfl_projection.shape}:mfl_projection")
        log.debug(f"{complete.shape}:complete")
        label = self.constant_label(constant_label).view(batch_size, -1, 1)
        log.debug(f"{constant_label.shape}:constant_label")
        complete = self.complete_dropout(complete)
        log.debug(f"{complete.shape}:complete")
        fin = torch.cat([complete.flatten(start_dim=1), emb.flatten(start_dim=1)], dim=-1)
        out = self.ff(fin)
        out_cond = self.bilinear(out, label.flatten(start_dim=1))
        log.debug(f"{out_cond.shape}:out_cond")
        fouter = torch.matmul(out_cond.view(batch_size, -1, 1), emb.view(batch_size, 1, -1))
        
        log.debug(f"{fouter.shape}:fouter")
        conv_conv = self.cond_encoder(fouter)
        log.debug(f"{conv_conv.shape}:conv_conv")
        out_cond = torch.cat([complete.flatten(start_dim=1), out_cond, out, conv_conv], dim=-1)
        
        
        
        
        
        
        
        
        
        
        out_cond = self.last_mha(out_cond.view(batch_size, 1, -1)).view(batch_size, -1)
        regressor = self.regresor(out_cond)
        out = regressor
        
        
        
        log.debug(f"{out.shape}:out")
        return out


class BestGPCR2(L.LightningModule):
    """Transformer model for kinase data."""
    def __init__(self, test=None):  
        super().__init__()
        self.LAYER_MULTIPLIER = 2
        down_sample = False
        conv_stride = 1
        print(test)
        self.complete_conv = component.Conv1DBlock(
            1,
            [
                2,
                8,
                16,
                32,
            ],
            stride=conv_stride,
            dilation=[1, 2, 4, 8],
            dropout=0.1,
            down_smaple=down_sample,
            normalization_end=True,
        )
        self.complete_conv1 = component.Conv1DBlock(
            1,
            [
                8,
            ],
            stride=conv_stride,
            dilation=[
                1,
            ],
            dropout=0.1,
            down_smaple=True,
            normalization_end=True,
        )
        
        self.prot_emb = 32
        self.prot_dim_dmodel = 1024
        self.prot_dim_layer = 4  
        self.prot_nheads = 16
        self.prot_dim_dmffeat = self.LAYER_MULTIPLIER * self.prot_dim_dmodel
        self.prot_dim_dp = 0.1
        self.prot_ff_in = 2560  
        self.prot_ff_dmffeat = self.LAYER_MULTIPLIER * self.prot_ff_in
        self.prot_ff_fout = 2048
        self.pdb_sequence_one_hot_conv = component.Conv1DBlock(
            self.prot_emb,
            [8, 16, 32, 64, 128, 32],
            stride=conv_stride,
            dropout=0.1,
            dilation=[1, 2, 4, 8, 16, 32, 32],
            down_smaple=down_sample,
            normalization_end=True,
        )
        self.emb_prot = nn.Embedding(25, self.prot_emb)
        
        self.prot_mha = MixerModel(
            self.prot_emb,
            n_layer=36,
            rms_norm=True,
            fused_add_norm=True,
        )
        
        self.pos_prot = component.PositionalEncoding1D(self.prot_emb)
        
        self.smi_emb = 32
        self.smi_dim_dmodel = 100
        self.smi_dim_layer = 4
        self.smi_nheads = 16
        self.smi_dim_dmffeat = self.LAYER_MULTIPLIER * self.prot_dim_dmodel
        self.smi_dim_dp = 0.1
        
        
        
        self.smi_conv = component.Conv1DBlock(
            self.smi_emb,
            [16, 32, 64, 32],
            stride=conv_stride,
            dropout=0.1,
            dilation=[1, 2, 4, 8],
            down_smaple=False,
        )
        self.emb_smi = nn.Embedding(575, self.smi_emb)
        self.pos_smi = component.PositionalEncoding1D(self.smi_emb)
        
        
        self.smile_mha = MixerModel(
            self.smi_emb,
            n_layer=32,
            rms_norm=True,
            fused_add_norm=True,
        )
        
        self.normalize_smile_desc = torch.nn.BatchNorm1d(170)
        self.smi_act = torch.nn.LeakyReLU(0.3)
        
        num = 2730 + 366  
        num = 2048
        self.complete_mha = MixerModel(
            num,
            n_layer=24,
            rms_norm=True,
            fused_add_norm=True,
        )
        
        self.complete_layer_norm = nn.LayerNorm(num)
        num = num + 1024
        self.complete_dropout = nn.Dropout(0.1)
        
        self.ff = nn.Sequential(
            nn.LayerNorm(num),
            nn.Linear(num, 1024),
            
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            
            nn.ReLU(),
            
            nn.Linear(128, 64),
        )
        
        reg_in = 2368  
        self.regresor = nn.Sequential(
            nn.LayerNorm(reg_in),
            nn.Linear(reg_in, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 128),
            
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            
        )
        self.last_mha = MixerModel(
            reg_in,
            n_layer=12,
            rms_norm=True,
            fused_add_norm=True,
        )
        self.fp_projection = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 366),
        )

        bi_out = 128
        self.constant_label = nn.Embedding(2, 32)
        self.bilinear = nn.Bilinear(64, 32, bi_out)
        num_input_channels: int = bi_out
        base_channel_size: int = 32
        act_fn: object = nn.GELU
        c_hid = base_channel_size
        self.cond_encoder = nn.Sequential(
            
            nn.Conv1d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(c_hid, c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Conv1d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(),
            nn.LayerNorm(4 * c_hid),
            
        )
    def forward(self, x):  
        protein_sequence, ligand_smile, smi_desc, charge_fp, constant_label, emb = (
            x[0],
            x[1].long().squeeze(1),
            
            x[2].float(),
            x[3],
            x[4],
            x[5],
        )
        log.debug(f"{protein_sequence.shape},{ligand_smile.shape}, <=== shapwe ")
        log.debug(f"{protein_sequence.dtype},{ligand_smile.dtype}, ")
        batch_size = ligand_smile.size(0)
        log.debug(f"{charge_fp.shape}:charge_fp")
        smi = self.emb_smi(ligand_smile)
        log.debug(f" {smi.shape}: smi")
        smi = smi.permute(0, 2, 1)
        smi = self.smi_conv(smi)
        prot = self.emb_prot(protein_sequence)
        log.debug(f"{prot.shape}:prot")
        prot = prot.permute(0, 2, 1)
        prot = self.pdb_sequence_one_hot_conv(prot)
        log.debug(f"{prot.shape}:prot")
        log.debug(f"smae {smi.shape} {prot.shape}")
        prot = self.prot_mha(prot)
        log.debug(f"{prot.shape}:prosdst")
        log.debug(f" {smi.shape}: smi")
        smi = self.pos_smi(smi)
        smi = self.smile_mha(smi)
        smi = smi.view(batch_size, 1, -1)
        prot = prot.view(batch_size, 1, -1)
        smi_desc = self.normalize_smile_desc(smi_desc.view(batch_size, -1)).view(batch_size, 1, -1)
        smi_desc = self.smi_act(smi_desc)
        complete = torch.cat(
            [
                smi,
                prot,
            ],
            dim=-1,
        ).view(batch_size, 1, -1)
        mfl_projection = self.fp_projection(charge_fp)
        label = self.constant_label(constant_label).view(batch_size, -1, 1)
        complete = self.complete_mha(complete).view(batch_size, -1)
        complete = self.complete_dropout(complete)
        log.debug(f"{complete.shape}:complete")
        fin = torch.cat([complete, emb.flatten(start_dim=1)], dim=-1)
        out = self.ff(fin)
        out_cond = self.bilinear(out, label.flatten(start_dim=1))
        fouter = torch.matmul(out_cond.view(batch_size, -1, 1), emb.view(batch_size, 1, -1))
        conv_conv = self.cond_encoder(fouter)
        out_cond = torch.cat([complete, out_cond, out, conv_conv], dim=-1)
        out_cond = self.last_mha(out_cond.view(batch_size, 1, -1)).view(batch_size, -1)
        regressor = self.regresor(out_cond)
        out = regressor
        log.debug(f"{out.shape}:out")
        return out
