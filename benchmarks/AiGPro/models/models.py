import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich.console import Console
from torch import nn
from aigpro.models import component
from aigpro.utils import logger

console = Console()
log = logger.get_logger()

class BestGPCR(L.LightningModule):
    def __init__(self, test=None):
        super().__init__()
        self.LAYER_MULTIPLIER = 2
        down_sample = False
        conv_stride = 1
        print(test)

        self.complete_conv = component.Conv1DBlock(
            1,
            [2, 8, 16, 32],
            stride=conv_stride,
            dilation=[1, 2, 4, 8],
            dropout=0.1,
            down_smaple=down_sample,
        )

        self.complete_conv1 = component.Conv1DBlock(
            1,
            [8],
            stride=conv_stride,
            dilation=[1],
            dropout=0.1,
            down_smaple=True,
        )

        self.prot_emb = 32
        self.prot_dim_dmodel = 1024
        self.prot_dim_layer = 6
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
        )

        self.emb_prot = nn.Embedding(25, self.prot_emb)
        self.prot_mha = component.MultiLayerTransformer(
            d_model=self.prot_emb,
            n_heads=self.prot_nheads,
            dropout=self.prot_dim_dp,
            n_layers=self.prot_dim_layer,
        )

        self.pos_prot = component.PositionalEncoding1D(self.prot_emb, max_len=700)
        self.smi_emb = 32
        self.smi_dim_dmodel = 100
        self.smi_dim_layer = 6
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
        self.pos_smi = component.PositionalEncoding1D(self.smi_emb, max_len=60)
        self.smile_mha = component.MultiLayerTransformer(
            d_model=self.smi_emb,
            n_heads=self.smi_nheads,
            dropout=self.smi_dim_dp,
            n_layers=self.smi_dim_layer,
        )

        self.normalize_smile_desc = torch.nn.BatchNorm1d(170)
        self.smi_act = torch.nn.LeakyReLU(0.3)
        num = 2250
        self.bidirectional = component.CCrossMultiLayerTransformer(
            d_model=32,
            v_d_model=32,
            k_d_model=32,
            n_heads=16,
            n_layers=2,
            dropout=0.1,
            out_dim=16,
        )
        self.otherbidirectional = component.CCrossMultiLayerTransformer(
            d_model=32,
            v_d_model=32,
            k_d_model=32,
            n_heads=16,
            n_layers=2,
            dropout=0.1,
            out_dim=16,
        )

        self.complete_projection = nn.Linear(1738, num)
        self.complete_mha = component.MultiLayerTransformer(
            d_model=num, n_heads=18, n_layers=4, dropout=0.1, out_dim=num
        )

        self.complete_dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(
            nn.Linear(num, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 2), # 3->2
        )

        self.constant_embedding = nn.Embedding(2, self.smi_emb)

    def forward(self, x):
        protein_sequence, ligand_smile, smi_desc, charge_fp, constant_label = (
            x[0],
            x[1].long().squeeze(1),
            x[2].float(),
            x[3],
            x[4],
        )
        log.debug(f"{protein_sequence.shape},{ligand_smile.shape}, <=== shape ")
        log.debug(f"{protein_sequence.dtype},{ligand_smile.dtype}, ")
        batch_size = ligand_smile.size(0)
        # 디버깅: 디바이스 확인
        log.debug(f"constant_label device: {constant_label.device}")
        log.debug(f"constant_embedding weight device: {self.constant_embedding.weight.device}")
        constant_label = constant_label.to(self.constant_embedding.weight.device)
        constant_emb = self.constant_embedding(constant_label)

        device = self.constant_embedding.weight.device

        # Ensure all inputs are on the correct device
        protein_sequence = protein_sequence.to(device)
        ligand_smile = ligand_smile.to(device)
        smi_desc = smi_desc.to(device)
        charge_fp = charge_fp.to(device)
        constant_label = constant_label.to(device)

        smi = self.emb_smi(ligand_smile)
        smi = torch.concat([constant_emb.unsqueeze(1), smi], dim=1)
        log.debug(f" {smi.shape}: smi")
        smi = smi.permute(0, 2, 1)
        smi = self.smi_conv(smi)
        prot = self.emb_prot(protein_sequence)
        prot = torch.concat([constant_emb.unsqueeze(1), prot], dim=1)
        log.debug(f"{prot.shape}:prot")
        prot = prot.permute(0, 2, 1)
        prot = self.pdb_sequence_one_hot_conv(prot)
        bi = self.bidirectional(prot, smi, smi).view(batch_size, 1, -1)
        di = self.otherbidirectional(smi, prot, prot).view(batch_size, 1, -1)
        prot = self.pos_prot(prot)
        prot = self.prot_mha(prot)
        smi = self.pos_smi(smi)
        smi = self.smile_mha(smi)
        smi = smi.view(batch_size, 1, -1)
        prot = prot.view(batch_size, 1, -1)
        smi_desc = self.normalize_smile_desc(smi_desc.view(batch_size, -1)).view(batch_size, 1, -1)
        smi_desc = self.smi_act(smi_desc)
        log.debug(f"{smi_desc.shape}:smi_desc, smi : {smi.shape}, prot : {prot.shape}")
        complete = torch.cat([smi_desc, bi, di], dim=-1).view(batch_size, 1, -1)
        mfl_projection = charge_fp.view(batch_size, 1, -1)
        constant_emb = constant_emb.view(batch_size, 1, -1)
        complete = torch.cat([mfl_projection, complete, constant_emb], dim=-1)
        log.debug(f"{complete.shape}:complete")
        complete = self.complete_projection(complete)
        complete = self.complete_mha(complete).view(batch_size, -1)
        complete = self.complete_dropout(complete)
        out = self.ff(complete)
        return out