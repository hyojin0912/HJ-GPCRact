import numpy as np
import pandas as pd
from sklearn import model_selection


def create_folds(data: pd.DataFrame, num_splits: int, targget_colum: str) -> pd.DataFrame:
    """Split data into folds.

    Args:
        data (pd.DataFrame): Dataframe to split.
        num_splits (int): Number of folds.
        targget_colum (str): Column to stratify.

    Returns:
        pd.DataFrame: Dataframe with kfold column.
    # we create a new column called kfold and fill it with -1
    """
    data["kfold"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    num_bins = int(np.floor(1 + np.log2(len(data))))
    data.loc[:, "bins"] = pd.cut(data[targget_colum], bins=num_bins, labels=False)
    kf = model_selection.StratifiedKFold(n_splits=num_splits)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):  # type: ignore
        data.loc[v_, "kfold"] = f
    data = data.drop("bins", axis=1)
    return data
