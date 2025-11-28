import Bio.PDB as bio
import numpy as np
import scipy
from aigpro.data.objects import XYZ

residue_letter_codes = {
    "GLY": "G",
    "PRO": "P",
    "ALA": "A",
    "VAL": "V",
    "LEU": "L",
    "ILE": "I",
    "MET": "M",
    "CYS": "C",
    "PHE": "F",
    "TYR": "Y",
    "TRP": "W",
    "HIS": "H",
    "LYS": "K",
    "ARG": "R",
    "GLN": "Q",
    "ASN": "N",
    "GLU": "E",
    "ASP": "D",
    "SER": "S",
    "THR": "T",
}

aa2ix = {
    "G": 0,
    "P": 1,
    "A": 2,
    "V": 3,
    "L": 4,
    "I": 5,
    "M": 6,
    "C": 7,
    "F": 8,
    "Y": 9,
    "W": 10,
    "H": 11,
    "K": 12,
    "R": 13,
    "Q": 14,
    "N": 15,
    "E": 16,
    "D": 17,
    "S": 18,
    "T": 19,
}


def cordinates(pdb_file) -> XYZ:
    """Returns the coordinates of the pdb file.

    Args:
        pdb_file (_type_): _description_

    Returns:
        XYZ: _description_
    """
    parser = bio.PDBParser()
    structure = parser.get_structure("X", pdb_file)
    coords = []
    atom_list = []
    res_list = []
    c_alpha = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "CA":
                        res_list.append(residue_letter_codes[residue.get_resname()])
                        c_alpha.append(atom.get_coord())

                    coords.append(atom.get_coord())
                    atom_list.append(atom.get_name())

    # always same order of atoms
    # order_atoms = ["N", "CA", "C", "O"]

    coords = np.array(coords)
    atom_list = np.array(atom_list)
    res_list = np.array(res_list)
    c_alpha = np.array(c_alpha)
    return XYZ(coords=coords, elements=atom_list, residue=res_list, c_alpha=c_alpha)


def distance_matrix(coords):
    """Calculates the distance matrix.

    Args:
        coords (_type_): _description_

    Returns:
        _type_: _description_
    """
    dist_mat = scipy.spatial.distance_matrix(coords, coords)
    return dist_mat
