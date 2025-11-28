# ruff: noqa: D102 D103 D107 D101 F841 D105 E501

from dataclasses import dataclass
from typing import Any
from typing import Iterator
from typing import Optional
from typing import Union
import numpy as np
import torch


@dataclass
class SeqInfo:  # noqa: D101
    id: str
    description: str
    seq: str
    splitter: str = "_"

    def __post_init__(self):
        self.length = len(self.seq)
        try:
            self.uniprot, self.species = self.id.split(self.splitter, 1)
        except ValueError:
            self.uniprot = self.id
            self.species = "unknown"

    def __str__(self):
        return f"{self.id} ({self.length}aa) from {self.species}\nuniprot: {self.uniprot}\nDesc: {self.description}\nSeq: {self.seq[:10]}...{self.seq[-10:]}\n"  # noqa: E501


@dataclass
class DM:
    dm_filename: str
    dm_col_iden_name: str = "entry_name"


@dataclass
class GetDM:
    # uniport_entry: str
    loaded_dm: Any
    dm_col_iden_name: str = "entry_name"


@dataclass
class GPCRTargetConfig:
    state: str = "train"
    protein_tokenizer_filename: str = "GPCR_prot_tokenizer.json"
    ligand_tokenizer_filename: str = "GPCR_smi_tokenizer.json"
    desc_col_name: str = "desc"
    canonical_smiles_col_name: str = "canonical_smiles"
    charge_fp_col_name: str = "charge_fp"
    selfies_col_name: str = "selfies"
    msa_col_name: str = "Align_Sequence"
    y_col_name: str = "pEndPoint"
    morgan_fp_col_name: str = "morgan_fingerprint"
    dm: Union[DM, None, GetDM] = None


@dataclass
class Metrics:
    mse: float
    rmse: float
    mae: float
    r2: float
    pearson: float
    spearman: float
    ci: float

    def __str__(self):
        return f"MSE: {self.mse}\nRMSE: {self.rmse}\nMAE: {self.mae}\nR2: {self.r2}\nPearson: {self.pearson}\nSpearman: {self.spearman}\nCI: {self.ci}\n"

    # post init converty type
    def __post_init__(self):
        self.mse = float(self.mse)
        self.rmse = float(self.rmse)
        self.mae = float(self.mae)
        self.r2 = float(self.r2)
        self.pearson = float(self.pearson)
        self.spearman = float(self.spearman)
        self.ci = float(self.ci)


@dataclass
class SeqDict:
    seq_dict: dict[str, SeqInfo]

    def __post_init__(self):
        self.seq_dict = self.seq_dict

    def __len__(self):
        return len(self.seq_dict)

    def __getitem__(self, key) -> SeqInfo:
        return self.seq_dict[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.seq_dict)

    def __str__(self):
        return f"SeqDict with {len(self.seq_dict)} sequences"

    def __repr__(self):
        return f"SeqDict with {len(self.seq_dict)} sequences"

    def __contains__(self, key):
        return key in self.seq_dict

    def __setitem__(self, key, value):
        self.seq_dict[key] = value

    def __delitem__(self, key):
        del self.seq_dict[key]

    def __eq__(self, other):
        return self.seq_dict == other.seq_dict

    def __ne__(self, other):
        return self.seq_dict != other.seq_dict

    def __lt__(self, other):
        return self.seq_dict < other.seq_dict

    def __le__(self, other):
        return self.seq_dict <= other.seq_dict

    def __gt__(self, other):
        return self.seq_dict > other.seq_dict

    def __ge__(self, other):
        return self.seq_dict >= other.seq_dict

    def __hash__(self):
        return hash(self.seq_dict)

    def __getstate__(self) -> dict[str, SeqInfo]:
        return self.seq_dict

    @property
    def uniprots(self) -> list[str]:
        return list(self.seq_dict.keys())

    @property
    def species(self) -> list[str]:
        return list(self.seq_dict[i].species for i in self.seq_dict.keys())

    @property
    def ids(self) -> list[str]:
        return list(self.seq_dict[i].id for i in self.seq_dict.keys())

    @property
    def seqs(self) -> list[str]:
        return list(self.seq_dict[i].seq for i in self.seq_dict.keys())

    @property
    def lengths(self) -> list[int]:
        return list(self.seq_dict[i].length for i in self.seq_dict.keys())

    def view(self, index: int) -> SeqInfo:
        return self.seq_dict[self.uniprots[index]]

    def from_json(self, json_file: str):
        """Load SeqDict from json file.

        Args:
            json_file (str): path to json file
        """
        import json

        with open(json_file, "r") as f:
            self.seq_dict = json.load(f)


@dataclass
class XYZ:
    coords: Union[list[list[float]], np.array]
    elements: Union[list[str], np.array]
    residue: Union[str, np.array]
    name: Optional[str] = None
    chain: Optional[str] = None
    c_alpha: Union[list[list[float]], np.array] = None


@dataclass
class MetricData:
    loss: torch.Tensor
    y_pred: torch.Tensor
    y_true: torch.Tensor
    y_pred_class: Optional[torch.Tensor] = None
    y_label: Optional[torch.Tensor] = None
