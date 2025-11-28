import os
from typing import Any
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerFast


def tokenize_smiles(smiles) -> list[Any]:
    """Tokenize SMILES string into a list of tokens.

    Parameters
    ----------
    smiles : str
        SMILES string.

    Returns
    -------
    list
        List of tokens.

    """
    return list(smiles)


def batch_preprocess_func(examples, tokenier, target_feature) -> list[Any]:
    """Return a list of tokenized SMILES strings.

    Args:
        examples (_type_): _description_
        tokenier (_type_): _description_
        target_feature (_type_): _description_

    Returns:
        _type_: _description_
    """
    return tokenier([" ".join(x) for x in examples[target_feature]])


def tokenize_data(  # noqa: D103
    tokenizer, examples, target_feature, max_length=128, padding="max_length", truncation=True, **kwargs
) -> BatchEncoding:
    return tokenizer(
        examples[target_feature],
        add_special_tokens=True,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        **kwargs,
    )


def tokenize_dataset(dataset, target_feature, tokenizer, num_proc=-1) -> list[Any]:  # noqa: D103
    if num_proc == -1:
        num_proc = os.cpu_count() - 1

    return dataset.map(
        tokenize_data,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer, "target_feature": target_feature},
    )


def load_custom_tokenizer(
    tokenizer_config_file, BOS_TOKEN="[CLS]", EOS_TOKEN="[END]", PAD_TOKEN="[PAD]", UNK_TOKEN="[UNK]"
) -> PreTrainedTokenizerFast:
    """Load tokenizer from config file.

    Args:
        tokenizer_config_file (_type_): _description_
        BOS_TOKEN (str, optional): _description_. Defaults to "[CLS]".
        EOS_TOKEN (str, optional): _description_. Defaults to "[END]".
        PAD_TOKEN (str, optional): _description_. Defaults to "[PAD]".
        UNK_TOKEN (str, optional): _description_. Defaults to "[UNK]".

    Returns:
        _type_: _description_
    """
    assert os.path.exists(tokenizer_config_file), f"{tokenizer_config_file} config file does not exist"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_config_file))
    tokenizer.add_special_tokens(
        {
            "bos_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "pad_token": PAD_TOKEN,
            "unk_token": UNK_TOKEN,
        }
    )
    return tokenizer
