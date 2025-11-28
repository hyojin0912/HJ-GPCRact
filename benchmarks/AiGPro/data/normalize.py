import numpy as np


def sigmoid(x):
    """Sigmoid function.

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """ ""
    return 1 / (1 + np.exp(-x))


def normalize_ic50(ic50_value, min_ic50_mm=1, max_ic50_nm=1e6):
    """Normalize IC50 values on a scale from 0 to 1.

    Args:
        ic50_value (_type_): _description_
        min_ic50_mm (int, optional): _description_. Defaults to 1.
        max_ic50_nm (_type_, optional): _description_. Defaults to 1e6.

    Returns:
        _type_: _description_
    """
    normalized_value = (ic50_value - min_ic50_mm) / (max_ic50_nm - min_ic50_mm)
    return normalized_value


def normalized_func(x):
    """Normalize IC50 values on a scale from 0 to 1.

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    return 1 - np.log10(x) / np.log10(1e6 - 1)
