import numpy as np
import os
from tqdm import tqdm
import biographs as bg
from bio_embeddings.embed import ProtTransBertBFDEmbedder, SeqVecEmbedder
import torch
import networkx as nx
from Bio.PDB.PDBParser import PDBParser
from torch_geometric.data import Dataset, Data

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.cuda("cpu")
print(device)


ftrs = np.load("data/pdb_to_seqvec_dict.npy", allow_pickle=True)

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

ressymbl = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K',
            'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
            'TRP': 'W', 'TYR': 'Y'}

pcp_dict = {'A': [0.62014, -0.18875, -1.2387, -0.083627, -1.3296, -1.3817, -0.44118],
            'C': [0.29007, -0.44041, -0.76847, -1.05, -0.4893, -0.77494, -1.1148],
            'D': [-0.9002, 1.5729, -0.89497, 1.7376, -0.72498, -0.50189, -0.91814],
            'E': [-0.74017, 1.5729, -0.28998, 1.4774, -0.25361, 0.094051, -0.4471],
            'F': [1.1903, -1.1954, 1.1812, -1.1615, 1.1707, 0.8872, 0.02584],
            'G': [0.48011, 0.062916, -1.9949, 0.25088, -1.8009, -2.0318, 2.2022],
            'H': [-0.40009, -0.18875, 0.17751, 0.77123, 0.5559, 0.44728, -0.71617],
            'I': [1.3803, -0.84308, 0.57625, -1.1615, 0.10503, -0.018637, -0.21903],
            'K': [-1.5003, 1.5729, 0.75499, 1.1057, 0.44318, 0.95221, -0.27937],
            'L': [1.0602, -0.84308, 0.57625, -1.273, 0.10503, 0.24358, 0.24301],
            'M': [0.64014, -0.59141, 0.59275, -0.97565, 0.46368, 0.46679, -0.51046],
            'N': [-0.78018, 1.0696, -0.38073, 1.2172, -0.42781, -0.35453, -0.46879],
            'P': [0.12003, 0.062916, -0.84272, -0.1208, -0.45855, -0.75977, 3.1323],
            'Q': [-0.85019, 0.16358, 0.22426, 0.8084, 0.04355, 0.24575, 0.20516],
            'R': [-2.5306, 1.5729, 0.89249, 0.8084, 1.181, 1.6067, 0.11866],
            'S': [-0.18004, 0.21392, -1.1892, 0.32522, -1.1656, -1.1282, -0.48056],
            'T': [-0.050011, -0.13842, -0.58422, 0.10221, -0.69424, -0.63625, -0.50017],
            'V': [1.0802, -0.69208, -0.028737, -0.90132, -0.36633, -0.3762, 0.32502],
            'W': [0.81018, -1.6484, 2.0062, -1.0872, 2.3901, 1.8299, 0.032377],
            'Y': [0.26006, -1.0947, 1.2307, -0.78981, 1.2527, 1.1906, -0.18876]}


class ProteinDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ProteinDataset, self).__init__(root, transform=None,
                                             pre_transform=None)
        self.data_prot = None
        self.data = self.processed_paths

    @property
    def raw_file_names(self):

        return [filename for filename in os.scandir(self.root + "/raw")]

    @property
    def processed_file_names(self):
        return [os.path.splitext(os.path.basename(file))[0] + '.pt' for file in self.raw_paths]

    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):

        return self.data_prot[idx]

    def get_adjacency(self, file):
        molecule = bg.Pmolecule(file)
        network = molecule.network()
        mat = nx.adjacency_matrix(network)
        m = mat.todense()
        return m

    def get_edgeindex(self, file):
        edge_ind = []
        m = self._get_adjacency(file)

        a = np.nonzero(m > 0)[0]
        b = np.nonzero(m > 0)[1]
        edge_ind.append(a)
        edge_ind.append(b)
        return torch.tensor(np.array(edge_ind), dtype=torch.long)

    def get_structure(self, file):
        parser = PDBParser()
        structure = parser.get_structure(id, file)
        return structure

    def get_sequence(self, structure):
        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in ressymbl.keys():
                        sequence = sequence + ressymbl[residue.get_resname()]
        return sequence

    def get_one_hot_symbftrs(self, sequence):
        one_hot_symb = np.zeros((len(sequence), len(pro_res_table)))
        row = 0
        for res in sequence:
            col = pro_res_table.index(res)
            one_hot_symb[row][col] = 1
            row += 1
        return torch.tensor(one_hot_symb, dtype=torch.float)

    def get_res_ftrs(self, sequence):
        res_ftrs_out = []
        for res in sequence:
            res_ftrs_out.append(pcp_dict[res])
        res_ftrs_out = np.array(res_ftrs_out)
        return torch.tensor(res_ftrs_out, dtype=torch.float)

    def get_node_ftrs(self, sequence):
        one_hot_symb = self.get_one_hot_symbftrs(sequence)
        res_ftrs_out = self.get_res_ftrs(sequence)
        return torch.tensor(np.hstack((one_hot_symb, res_ftrs_out)), dtype=torch.float)

    def bert_embedding(self, seq):
        embedder = ProtTransBertBFDEmbedder()
        embedding = embedder.embed(seq)
        protein_embd = torch.tensor(embedding).sum(dim=0)
        return protein_embd

    def seqVec_embedding(self, seq):
        embedder = SeqVecEmbedder()
        embedding = embedder.embed(seq)
        protein_embd = torch.tensor(embedding).sum(dim=0)
        return protein_embd

    def process(self):
        self.data = self.processed_paths

        data_list = []
        count = 0
        for file in tqdm(self.raw_paths):
            struct = self.get_structure(file)
            seq = self.get_sequence(struct)
            node_feats = self.bert_embedding(seq)
            # mat = self.get_adjacency(file)
            edge_index = self.get_edgeindex(file)
            data = Data(x=node_feats, edge_index=edge_index)
            count += 1
            data_list.append(data)
            torch.save(data, self.processed_dir + "/" + os.path.splitext(os.path.basename(file))[0] + '.pt')

        self.data_prot = data_list
        print(count)


prot_graphs = ProteinDataset("data/raw")
prot_graphs.process()
print("all proteins embeded !")
