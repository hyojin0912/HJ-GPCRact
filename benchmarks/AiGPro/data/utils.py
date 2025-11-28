import h5py


def get_dm_from_h5(uniprot_entry, save_file):
    """Gets the dm from h5.

    Args:
        uniprot_entry (_type_): _description_
        save_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    with h5py.File(save_file, "r") as hf:
        adm = hf[uniprot_entry + "_active"][:]
        idm = hf[uniprot_entry + "_inactive"][:]
    return adm, idm


def get_dm(uniprot_entry, hf):
    """Gets the dm.

    Args:
        uniprot_entry (_type_): _description_
        hf (_type_): _description_

    Returns:
        _type_: _description_
    """
    if "_human" in uniprot_entry:
        uniprot_entry = uniprot_entry.split("_")[0]
    adm = hf[uniprot_entry + "_active"][:]
    idm = hf[uniprot_entry + "_inactive"][:]
    return adm, idm
