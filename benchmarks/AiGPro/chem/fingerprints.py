# ruff : noqa : F841

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs


def get_smiles_from_file(pdb_file, strict=False):  # noqa: D103
    if strict:
        removeHs = True
        sanitize = True
    else:
        removeHs = False
        sanitize = False
    try:
        mol = Chem.MolFromMol2File(pdb_file.replace("sdf", "mol2"), removeHs=removeHs, sanitize=sanitize)
        if not strict:
            mol.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(mol)
    except Exception:
        try:
            mol = Chem.MolFromMolFile(pdb_file, removeHs=removeHs, sanitize=sanitize)
            if not strict:
                mol.UpdatePropertyCache(strict=False)
            return Chem.MolToSmiles(mol)
        except Exception:
            try:
                OBABEL = "/share/anaconda3/bin/obabel"
                # !{OBABEL} -imol2 {pdb_file.replace("sdf", "mol2")} -osmi -O {pdb_file.replace("mol2", "smi")}
                with open(pdb_file.replace("mol2", "smi"), "r") as f:
                    smi = f.read()
                    # remove file
                    # !rm -rf {pdb_file.replace("mol2", "smi")}
                    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            except Exception:
                pass
            pass
    return None


def morgan_fingerprint(smiles, radius=3, nBits=512):
    """Generates a Morgan fingerprint for a SMILES string.

    Args:
        smiles (str): SMILES string.
        radius (int, optional): Morgan fingerprint radius. Defaults to 2.
        nBits (int, optional): Morgan fingerprint bit length. Defaults to 2048.

    Returns:
        np.ndarray: Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def assess_two_letter_elements(df):
    """Find the two letter elements in dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe which requires preprocessing.

    Returns
    -------
    two_letter_elements : list
        List with found two letter elements
    """
    # Search for unique characters in SMILES strings
    unique_chars = set(df.canonical_smiles.apply(list).sum())
    # Get upper and lower case letters only
    upper_chars = []
    lower_chars = []
    for entry in unique_chars:
        if entry.isalpha():
            if entry.isupper():
                upper_chars.append(entry)
            elif entry.islower():
                lower_chars.append(entry)
    print(f"Upper letter characters {sorted(upper_chars)}")
    print(f"Lower letter characters {sorted(lower_chars)}")

    # List of all possible periodic elements
    periodic_elements = [
        "Ac",
        "Al",
        "Am",
        "Sb",
        "Ar",
        "As",
        "At",
        "Ba",
        "Bk",
        "Be",
        "Bi",
        "Bh",
        "B",
        "Br",
        "Cd",
        "Ca",
        "Cf",
        "C",
        "Ce",
        "Cs",
        "Cl",
        "Cr",
        "Co",
        "Cn",
        "Cu",
        "Cm",
        "Ds",
        "Db",
        "Dy",
        "Es",
        "Er",
        "Eu",
        "Fm",
        "Fl",
        "F",
        "Fr",
        "Gd",
        "Ga",
        "Ge",
        "Au",
        "Hf",
        "Hs",
        "He",
        "Ho",
        "H",
        "In",
        "I",
        "Ir",
        "Fe",
        "Kr",
        "La",
        "Lr",
        "Pb",
        "Li",
        "Lv",
        "Lu",
        "Mg",
        "Mn",
        "Mt",
        "Md",
        "Hg",
        "Mo",
        "Mc",
        "Nd",
        "Ne",
        "Np",
        "Ni",
        "Nh",
        "Nb",
        "N",
        "No",
        "Og",
        "Os",
        "O",
        "Pd",
        "P",
        "Pt",
        "Pu",
        "Po",
        "K",
        "Pr",
        "Pm",
        "Pa",
        "Ra",
        "Rn",
        "Re",
        "Rh",
        "Rg",
        "Rb",
        "Ru",
        "Rf",
        "Sm",
        "Sc",
        "Sg",
        "Se",
        "Si",
        "Ag",
        "Na",
        "Sr",
        "S",
        "Ta",
        "Tc",
        "Te",
        "Ts",
        "Tb",
        "Tl",
        "Th",
        "Tm",
        "Sn",
        "Ti",
        "W",
        "U",
        "V",
        "Xe",
        "Yb",
        "Y",
        "Zn",
        "Zr",
    ]

    two_char_elements = []
    for upper in upper_chars:
        for lower in lower_chars:
            ch = upper + lower
            if ch in periodic_elements:
                two_char_elements.append(ch)

    two_char_elements_smiles = set()
    for char in two_char_elements:
        if df.canonical_smiles.str.contains(char).any():
            two_char_elements_smiles.add(char)

    return two_char_elements_smiles
