import os
from rdkit import Chem

def bool_to_int(array):
    return [1 if cond else 0 for cond in array]

def atom_features(atom):
    return list(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                           'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                           'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                                           'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                           'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])) + \
                    list(one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])) + \
                    list(one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])) + \
                    list(one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])) + \
                    list(bool_to_int([atom.GetIsAromatic()])) +\
                    list(one_of_k_encoding_unk(str(atom.GetHybridization()),['SP','SP2','SP3','SP3D','SP3D2','UNK']))

def bond_features(bond):
    bt = bond.GetBondType()
    return bool_to_int([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])

def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))

def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return map(lambda s: 1 if x == s else 0, allowable_set)

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: 1 if x == s else 0, allowable_set)

if __name__=='__main__':
    la=num_atom_features()
    lb=num_bond_features()
    levo='CC1CC(=O)NN=C1C2=CC=C(C=C2)NN=C(C#N)C#N' #levosimendan
    print("{} atom features and {} bond features for CC".format(la,lb))
