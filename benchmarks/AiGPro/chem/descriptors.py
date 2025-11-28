from typing import Any
import numpy as np
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdMolDescriptors  # type: ignore
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
#from aigpro.core.metrics import ic50_to_pic50
from aigpro.data.normalize import normalized_func


def morgan_fingerprint(smiles: str, radius: int = 3, num_bits: int = 512, use_counts: bool = False) -> np.ndarray:
    """Generates a morgan fingerprint for a smiles string.

    :param smiles: A smiles string for a molecule.
    :param radius: The radius of the fingerprint.
    :param num_bits: The number of bits to use in the fingerprint.
    :param use_counts: Whether to use counts or just a bit vector for the fingerprint
    :return: A 1-D numpy array containing the morgan fingerprint.
    """
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    else:
        mol = smiles
    if use_counts:
        fp_vect = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits, useChirality=True)
    else:
        fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits, useChirality=True)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)

    return fp


def macc_fp(smiles: str) -> np.ndarray:  # noqa: D103
    return MACCSkeys.GenMACCSKeys(smiles)


def mqn_fp(smiles: str) -> np.ndarray:  # noqa: D103
    return rdMolDescriptors.MQNs_(smiles)


def generate_charge_fingerprint(smiles, n_bits=512, bin_min=-1.0, bin_max=1.0, nbins=32):  # noqa: D103
    """Generates a charge fingerprint for a smiles string.

    Args:
        smiles (_type_): _description_
        n_bits (int, optional): _description_. Defaults to 512.
        bin_min (float, optional): _description_. Defaults to -1.0.
        bin_max (float, optional): _description_. Defaults to 1.0.
        nbins (int, optional): _description_. Defaults to 32.

    Returns:
        _type_: _description_
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Chem.AllChem.ComputeGasteigerCharges(mol)
    charges = [mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge") for i in range(mol.GetNumAtoms())]
    fingerprint = np.zeros(n_bits, dtype=np.uint8)
    bins = np.linspace(bin_min, bin_max, nbins + 1)
    for idx, charge in enumerate(charges):
        bin_index = np.digitize(charge, bins) - 1
        bit_index = idx * nbins + bin_index
        if bit_index < n_bits:
            fingerprint[bit_index] = 1
    return fingerprint


LigandDescriptors = [
    "MaxEStateIndex",
    "MinEStateIndex",
    "MaxAbsEStateIndex",
    "MinAbsEStateIndex",
    "qed",
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "NumValenceElectrons",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "PEOE_VSA14",
    "SMR_VSA1",
    "SMR_VSA10",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "TPSA",
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA11",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "VSA_EState1",
    "VSA_EState10",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "RingCount",
    "MolLogP",
    "MolMR",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_bicyclic",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_urea",
]
DescCalc = MolecularDescriptorCalculator(LigandDescriptors)


def compute_ligand_descriptors(smiles: str) -> tuple[int, ...]:
    """Computes the ligand descriptors for a smiles string.

    :param smiles: A smiles string for a molecule.
    :return: A 1-D numpy array containing the ligand descriptors.
    """
    if isinstance(smiles, str):
        mol = Chem.MolFromSmiles(smiles, sanitize=True)  # type: ignore
    else:
        mol = smiles
    fp_vect: tuple[int, ...] = DescCalc.CalcDescriptors(mol)
    return fp_vect


def add_desc(row, col="canonical_smi", verbose=False) -> tuple[int, ...] | None:  # noqa: D103
    """Add ligand descriptors to a dataframe.

    Args:
        row (_type_): _description_
        col (str, optional): _description_. Defaults to "canonical_smi".
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: tuple[int, ...]
    """
    try:
        return compute_ligand_descriptors(row[col])
    except Exception as e:
        if verbose:
            print(e)
        return None


def add_charge_fp(row, col="canonical_smi", verbose=False) -> list[Any] | None:  # noqa: D103
    """Add charge fingerprint to a dataframe.

    Args:
        row (_type_): _description_
        col (str, optional): _description_. Defaults to "canonical_smi".
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: list[int]
    """
    try:
        return list(generate_charge_fingerprint(row[col]))
    except Exception as e:
        if verbose:
            print(e)
        return None


def add_canonical_smi(row, target_col="smiles", verbose=False, sanitize=True):
    """Add canonical smiles to a dataframe.

    Args:
        row (_type_): _description_
        target_col (str, optional): _description_. Defaults to "smiles".
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    try:
        x = Chem.MolFromSmiles(  # type: ignore
            row[target_col], sanitize=sanitize
        )
        return Chem.MolToSmiles(x, canonical=True, isomericSmiles=True)  # type: ignore
    except Exception as e:
        if verbose:
            print(e)
        return None


def add_selfies(row, target_col="canoncail_smiles", verbose=False) -> str | None:
    """Add selfies to a dataframe.

    Args:
        row (_type_): _description_
        target_col (str, optional): _description_. Defaults to "canoncail_smiles".
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        str | None: _description_
    """
    try:
        return sf.encoder(row[target_col], strict=False)
    except Exception as e:
        if verbose:
            print(e)
        return None


def add_endpoint(row, target_col="standard_value", unit_col="standard_units", verbose=False) -> float | None:
    """Add endpoint to a dataframe.

    Args:
        row (_type_): _description_
        target_col (str, optional): _description_. Defaults to "standard_value".
        unit_col (str, optional): _description_. Defaults to "standard_units".
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        float | None: _description_
    """
    try:
        return ic50_to_pic50(float(row[target_col]), row[unit_col])
    except Exception as e:
        if verbose:
            print(e)
        return None


def custom_add_endpoint(row, target_col="standard_value", verbose=False) -> float | None:
    """Add endpoint to a dataframe.

    Args:
        row (_type_): _description_
        target_col (str, optional): _description_. Defaults to "standard_value".
        unit_col (str, optional): _description_. Defaults to "standard_units".
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        float | None: _description_
    """
    try:
        return compute_scaled_endpoint(float(row[target_col]))
    except Exception as e:
        if verbose:
            print(e)
        return None


def compute_scaled_endpoint(value, k=1):
    """compute_scaled_endpoint.

    Args:
        value (_type_): _description_
        k (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """  # value in nm
    if value <= 1: 
        scaled_value = 1
    elif value >= 1e6:  
        scaled_value = 0
    else:
        scaled_value = normalized_func(value)
    return scaled_value
