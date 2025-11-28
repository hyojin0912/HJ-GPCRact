import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from transformers import PreTrainedTokenizerFast
from aigpro.chem.old_tokenizer import tokenize_smiles
import ast
from typing import Tuple, List

class GPCRDataset(Dataset):
    def __init__(self, data_file: str):
        self.df = pd.read_csv(data_file)
        self.df["desc"] = self.df["desc"].apply(ast.literal_eval)
        self.df["charge_fp"] = self.df["charge_fp"].apply(ast.literal_eval)
        self.df["morgan_fp"] = self.df["morgan_fp"].apply(ast.literal_eval)
        self.prot_tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/aigpro_api/aigpro/references/GPCR_prot_tokenizer.json")
        if self.prot_tokenizer.pad_token is None:
            self.prot_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.prot_tokenizer.pad_token = '[PAD]'
        self.max_seq_len = 1500
        self.max_smiles_len = 100

    def __len__(self) -> int:
        return len(self.df)

    def tokenize_sequence(self, sequence: str) -> torch.Tensor:
        encoding = self.prot_tokenizer(
            sequence,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoding["input_ids"].squeeze(0)  # CPU에서 생성

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        row = self.df.iloc[idx]
        protein_sequence = self.tokenize_sequence(row["Align_Sequence"])
        ligand_smile = torch.tensor(tokenize_smiles(row["canonical_smiles"]), dtype=torch.long)
        if ligand_smile.size(0) < self.max_smiles_len:
            ligand_smile = torch.nn.functional.pad(ligand_smile, (0, self.max_smiles_len - ligand_smile.size(0)), value=0)
        ligand_smile = ligand_smile[:self.max_smiles_len].unsqueeze(0)
        smi_desc = torch.tensor(row["desc"][:170], dtype=torch.float32).unsqueeze(0)
        charge_fp = torch.tensor(row["charge_fp"][:512], dtype=torch.float32).unsqueeze(0)
        constant_label = torch.tensor(0, dtype=torch.long)
        label = torch.tensor(row["pEndPoint"], dtype=torch.long)
        x = [protein_sequence, ligand_smile, smi_desc, charge_fp, constant_label]
        y = (torch.tensor(0.0), label)
        return x, y

class GPCRDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 0, train_file: str = None, test_file: str = None, val_file: str = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers  # 멀티프로세싱 비활성화
        self.train_file = train_file
        self.test_file = test_file
        self.val_file = val_file
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        if self.train_file:
            self.train_dataset = GPCRDataset(self.train_file)
        if self.val_file:
            self.val_dataset = GPCRDataset(self.val_file)
        if self.test_file:
            self.test_dataset = GPCRDataset(self.test_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False) if self.val_dataset else None

    def predict_dataloader(self):
        return self.test_dataloader()

    def predict_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        