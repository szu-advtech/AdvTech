import csv
import networkx as nx
import numpy as np
from rdkit import Chem



def atom_features(atom):
    HYB_list = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.OTHER]
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Sm', 'Tc', 'Gd', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetExplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding(atom.GetFormalCharge(), [-4, -3, -2, -1, 0, 1, 2, 3, 4]) +
                    one_of_k_encoding(atom.GetHybridization(), HYB_list) +
                    [atom.GetIsAromatic()])

def load_drug_smile(file):
    drug_dict = {}
    drug_smile = []
    reader = csv.reader(open(file))
    for item in reader:
        name = item[0]
        smile = item[1]
        if name in drug_dict:
            index = drug_smile[name]
        else:
            index = len(drug_dict)
            drug_dict[name] = index
        drug_smile[index] = smile
    return drug_dict, drug_smile

def smile_to_graph(smile):
    # ?????????mol??????rdkit?????????????????????????????????????????????????????????????????????smile(?????????????????????????????????)
    mol = Chem.MolFromSmiles(smile)
    # ??????????????????????????????
    features = []
    # ???????????????????????????
    atom_num = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature)
    features = np.array(features)

    # ????????????
    edges = []
    edge_type = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])  # ????????????????????????????????????????????????
        edge_type.append(bond.GetBondTypeAsDouble())  # ?????????????????????????????????
    # ??????????????????????????????
    # ????????????G ???????????????????????????????????????????????????u???v?????????????????????????????????u???v??????????????????v???u?????????????????????????????????
    g = nx.Graph(edges).to_directed()  # ???????????????
    edge_index = []  # ???????????????????????????????????????
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    if not edge_index:
        edge_index = []
    else:
        edge_index = np.array(edge_index).transpose(1, 0)  # ???????????????????????????????????????????????????

    return atom_num, features, edge_index, edge_type


def conver2graph(drug_smile):
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile) # ?????????????????????
        smile_graph[smile] = g
    return smile_graph


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    # lambda ????????????????????????
    # map ??????allowable_set????????????????????????lambda????????????????????????????????????????????????
    return list(map(lambda s: x == s, allowable_set))