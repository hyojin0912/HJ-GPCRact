import json
import os
from pathlib import Path
import numpy as np
from SmilesPE.pretokenizer import atomwise_tokenizer

smile_json_filename = "smiles_vocab.json"
smile_json = Path("/home/aigpro_api/aigpro/references") / smile_json_filename

with open(smile_json, "r") as JDATA:
    # load json
    SMILES_VOCAB = json.load(JDATA)
assert SMILES_VOCAB is not None, "SMILES_VOCAB is None"


def tokenize_smiles(smi, max_length=100):
    """Tokenize SMILES string.

    Args:
        smi (_type_): _description_
        max_length (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    tokens = atomwise_tokenizer(smi)
    tokens = [SMILES_VOCAB.get(i, 0) for i in tokens]
    if len(tokens) < max_length:
        tokens += [0] * (max_length - len(tokens))
    return tokens[:max_length]


def get_pad_array(cmap, pad_size=(600, 600), constant=-1):
    """Get padding array.

    Args:
        cmap (_type_): _description_
        pad_size (tuple, optional): _description_. Defaults to (600, 600).
        constant (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """
    assert len(pad_size) == 2, "Padding size bust be 2D"
    try:
        if cmap.shape[0] > pad_size[0]:
            cmap = cmap[: pad_size[0], :]

        if cmap.shape[1] > pad_size[1]:
            cmap = cmap[:, : pad_size[1]]
        cmap = np.pad(
            cmap,
            ((0, pad_size[0] - cmap.shape[0]), (0, pad_size[1] - cmap.shape[1])),
            "constant",
            constant_values=constant,
        )
        return cmap
    except Exception as e:
        print(e)
        return None
