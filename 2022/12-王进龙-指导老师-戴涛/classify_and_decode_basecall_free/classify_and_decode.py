#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import h5py
import os
import re
import pandas as pd
import numpy as np
import logging
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from subprocess import check_output

logger = logging.getLogger(__name__)
if logger.handlers:
    logger.handlers = []
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler()) 


# In[ ]:


# Input files

# Path to guppy basecaller output
basecall_dir = "path/to/guppy/basecaller/output"
try:
    os.makedirs(os.path.join(basecall_dir, "cnn"))
except:
    pass

# Path to fast5 files
f5_dir = "path/to/fast5/files"


# In[ ]:


# Output files

# Raw signal data will be extracted from all the fast5s and saved here as input to the classifier
molbit_data_file = os.path.join(basecall_dir, "molbit_extracted_data.hdf5")

# CNN classification results will be saved here
cnn_label_file = os.path.join(basecall_dir, "cnn", "labeled_reads.csv")


# In[2]:


device = torch.device('cuda')


# In[3]:


def med_mad(data, factor=1.4826):
    """Modified from Mako.
    Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median.

    :param data: A :class:`ndarray` object
    :param axis: For multidimensional arrays, which axis to calculate over 

    :returns: a tuple containing the median and MAD of the data

    .. note :: the default `factor` scales the MAD for asymptotically normal
        consistency as in R.

    """
    dmed = torch.median(data)
    dmad = factor * torch.median(torch.abs(data - dmed))
    return dmed, dmad

def _scale_data( data):
    '''Modified from Mako.'''
    med, mad = med_mad(data)
    data = (data - med) / mad
    return data

def trim_start_heuristic(signal, thresh=2, offset=20):
    try:
        above = np.where(signal[offset:] > thresh)[0][0] + offset
    except IndexError:
        above = 0
    return signal[above:]


# In[4]:


class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        O_1, O_2, O_3, O_4, O_5 = 64, 128, 256, 512, 1024
        K_1, K_2, K_3, K_4, K_5 = 15, 8, 6, 4, 2
        KP_1, KP_2, KP_3, KP_4, KP_5 = 6, 3, 2, 2, 1
        FN_1, FN_2 = 1000, 500

        self.conv1 = nn.Sequential(nn.Conv1d(1, O_1, K_1, stride=1), nn.ReLU(), nn.AvgPool1d(KP_1))
        self.conv1_bn = nn.BatchNorm1d(O_1)

        self.conv2 = nn.Sequential(nn.Conv1d(O_1, O_2, K_2), nn.ReLU(), nn.AvgPool1d(KP_2))
        self.conv2_bn = nn.BatchNorm1d(O_2)

        self.conv3 = nn.Sequential(nn.Conv1d(O_2, O_3, K_3), nn.ReLU(), nn.AvgPool1d(KP_3))
        self.conv3_bn = nn.BatchNorm1d(O_3)

        self.conv4 = nn.Sequential(nn.Conv1d(O_3, O_4, K_4), nn.ReLU(), nn.AvgPool1d(KP_4))
        self.conv4_bn = nn.BatchNorm1d(O_4)

        self.conv5 = nn.Sequential(nn.Conv1d(O_4, O_5, K_5), nn.ReLU(), nn.AvgPool1d(KP_5))
        self.conv5_bn = nn.BatchNorm1d(O_5)

        # not used, but is in the model file for some reason
        self.gru1 = nn.GRU(input_size=92160, hidden_size=10, num_layers=1)

        self.fc1 = nn.Linear(37888, FN_1, nn.Dropout(0.5))
        self.fc1_bn = nn.BatchNorm1d(FN_1)

        self.fc2 = nn.Linear(FN_1, FN_2, nn.Dropout(0.5))
        self.fc2_bn = nn.BatchNorm1d(FN_2)

        self.fc3 = nn.Linear(FN_2, 96)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_bn(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_bn(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_bn(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.relu(self.conv4_bn(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.relu(self.conv5_bn(x))
        x = x.view(len(x), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1_bn(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_bn(x))
        x = self.fc3(x)
        return x


# In[5]:


class MolbitDataset(Dataset):
    def __init__(self, data_file, unknown_labels=False):
        with h5py.File(data_file, "r") as f:
            self.data = torch.FloatTensor(f.get("data")[()])
            self.n_records = self.data.shape[0]
            self.max_len = self.data.shape[2]
            try:
                self.labels = torch.IntTensor(f.get("labels")[()])
                self.n_labels = len(np.unique(self.labels))
            except:
                self.labels = torch.IntTensor([-1 for _ in range(self.n_records)])
                self.n_labels = 0

        # Shuffle data
        self.shuffle_index = np.random.choice(range(self.n_records), replace=False, size=self.n_records)
        self.data = self.data[self.shuffle_index]       
        self.labels = self.labels[self.shuffle_index]        
        
    def _get_onehot(self, label):
        if self.labels is None:
            return None
        ix = self.labels.index(label)
        onehot = torch.zeros(self.n_labels)
        onehot[ix] = 1
        return onehot
    
    def __len__(self):
        return self.n_records
        
    def __getitem__(self, idx):
        return self.data[idx, :, :], self.labels[idx]


# In[6]:


def rescale_counts(counts, scaling_factors):
    n_reads = sum(counts)
    rescaled = np.multiply(np.array(counts), scaling_factors)
    return np.ceil(rescaled / sum(rescaled) * n_reads).astype(int)

def get_read_counts(labels, possible_labels=[]):
    labels = list(labels)

    for i, label in enumerate(labels):
        if label not in possible_labels:
            labels[i] = "-1"
    labels = np.array(labels, dtype=np.array(possible_labels).dtype)
    labels, counts = np.unique(labels, return_counts=True)
    ordered_counts = np.zeros(len(possible_labels), dtype=int)
    for i, possible_label in enumerate(possible_labels):
        ix = np.argwhere(labels == possible_label)
        if len(ix) > 0:
            ix = ix[0][0]
            ordered_counts[i] = counts[ix]
            assert labels[ix] == possible_label
            ordered_counts[i] = counts[ix]
    return ordered_counts

def get_tag(read_counts, t=0):
    read_counts = np.array(read_counts)
    tag = list(np.where(read_counts > t, 1, 0))
    return tag

def decode_c(received_codeword, decoder_binary):
    result = check_output([decoder_binary, received_codeword]).decode("utf-8").split("\n")
    codeword_distance = int(re.findall(r"\"distance\": ([\d]+)", result[2])[0])
    corrected_message = re.findall(r"\"message\": \"([\d]+)\"", result[1])[0]
    return corrected_message, codeword_distance

def compute_decoding_helper(counts, decoder_binary, stop_at_d=None):
    best_d, best_msg = 99999, None
    thresholds = list(np.sort(np.unique(counts)))
    for t in thresholds[::-1]:
        codeword_at_t = get_tag(counts, t=t)
        codeword_str = "".join([str(x) for x in codeword_at_t])
        closest_msg, closest_d = decode_c(codeword_str, decoder_binary)
        if stop_at_d is not None and closest_d <= stop_at_d:
            best_d = closest_d
            best_msg = closest_msg
            break
        if closest_d < best_d:
            best_d = closest_d
            best_msg = closest_msg
    return best_d, best_msg


# In[7]:


def decode_run(f5_dir, model_file, decoder_file, possible_labels, molbit_data_file, cnn_label_file,
               overwrite=False, conf_thresh=0.9, batch_size=500, n_workers_cnn=30,
               scaling_factors=None, decoding_guarantee=None):
    
    # Read in fast5 data & save to molbit dataset
    logger.info(f"Reading in fast5 data from {f5_dir}.")
    md, read_ids = fast5_to_molbit_dataset(f5_dir, molbit_data_file, overwrite=overwrite)
    
    # Load saved model
    logger.info(f"Reading in pretrained CNN.")
    model = load_model(model_file)
    
    # Run classification
    logger.info("Beginning classification.")
    preds, scores = classify(model, md, conf_thresh=conf_thresh,
                             batch_size=batch_size, n_workers=n_workers_cnn)
    
    # Save predictions to file
    logger.info(f"Saving classifications to {cnn_label_file}.")
    cnn_df = pd.DataFrame()
    cnn_df["read_id"] = read_ids
    cnn_df["cnn_label"] = preds
    cnn_df["cnn_score"] = scores
    cnn_df.to_csv(cnn_label_file, sep="\t", index=False)
    
    # Decode 
    logger.info("Beginning decoding.")
    d, msg = decode(preds, decoder_file,
                    possible_labels=possible_labels,
                    scaling_factors=scaling_factors,
                    stop_at_d=decoding_guarantee)
    logger.info(f"Decoded: {msg}, {d}")
    return msg, d
    
def fast5_to_molbit_dataset(f5_dir, molbit_data_file, overwrite=False):
    if not os.path.exists(molbit_data_file):
        f5_dirs = [f5_dir] + [os.path.join(f5_dir, x) for x in os.listdir(f5_dir) if "fast5" in x and not x.endswith(".fast5")]
        f5_fnames = []
        for f5_dir in f5_dirs:
            logger.debug(f"Searching dir: {f5_dir}")
            f5_fnames.extend([os.path.join(f5_dir, x) for x in os.listdir(f5_dir) if x.endswith(".fast5")])
        try:
            assert len(f5_fnames) > 0
        except:
            print(f"Checked these directories & found no fast5 files. {f5_dirs}")
        logger.info(f"Beginning data extraction ({len(f5_fnames)} fast5 files).")
        read_ids = []
        signal_data = []
        for f5_file in f5_fnames:
            with h5py.File(f5_file, "r") as f5:
                for group in f5.get("/").values():
                    read_id = str(dict(group.get("Raw").attrs).get("read_id"))[2:-1]
                    read_ids.append(read_id)
                    raw = torch.FloatTensor(list(group.get("Raw/Signal")[:15000]))  # extra for normalization
                    raw = trim_start_heuristic(_scale_data(raw))[:3000]
                    x = torch.zeros(3000)
                    x[:len(raw)] = raw
                    signal_data.append(x)
        signal_data_stacked = torch.stack(signal_data)
        signal_data_stacked = signal_data_stacked.unsqueeze(1)
        logger.info(f"Saving to file ({len(read_ids)} reads): {molbit_data_file}")
        with h5py.File(molbit_data_file, "w", swmr=True) as f:
            f.create_dataset("read_ids", shape=(len(read_ids), ), data=np.array(read_ids, dtype="S"))
            f.create_dataset("data", shape=signal_data_stacked.shape, dtype=np.float, data=signal_data_stacked)
    else:
        with h5py.File(molbit_data_file, "r") as f:
            read_ids = f.get("read_ids")[()]
    md = MolbitDataset(molbit_data_file, unknown_labels=True)
    return md, read_ids

def load_model(pretrained_model_file):
    model = CNN()
    model.load_state_dict(torch.load(pretrained_model_file))
    model.cuda()
    return model

def classify(model, md, conf_thresh=0.9, batch_size=500, n_workers=30):
    assert conf_thresh >= 0 and conf_thresh <= 1
    loader_params = {"batch_size": batch_size,
                     "num_workers": n_workers,
                     "shuffle": False}
    data_generator = DataLoader(md, **loader_params)
    preds = []
    scores = []
    for local_batch, local_labels in data_generator:
        local_batch = local_batch.to(device)
        pred = model(local_batch).to(dtype=torch.float64)
        softmax_score = torch.nn.functional.softmax(pred, dim=1)
        local_scores, local_preds = torch.max(softmax_score, dim=1)
        preds.extend([int(pred) if score > conf_thresh else -1 for score, pred in zip(local_scores.cpu(), local_preds.cpu())])
        scores.extend([float(score) for score in local_scores.cpu()])
    return preds, scores

def decode(labels, decoder_binary, possible_labels=[], scaling_factors=None, stop_at_d=None):
    counts = get_read_counts(labels, possible_labels=possible_labels)
    if scaling_factors is not None:
        counts = rescale_counts(counts, scaling_factors)
    best_d, best_msg = compute_decoding_helper(counts, decoder_binary, stop_at_d=stop_at_d)
    return best_d, best_msg


# In[8]:


cnn_label_file = "test_labels.csv"
model_file = "../molbit_classification/saved_models/molbit_classification_v4_0_1.20190827.pt"
decoder = "../ecc/decoder"
possible_labels = range(96)


# In[9]:


scaling_factors = [117.44079692,  296.08219178,   79.3902663 ,   63.8680128 ,
        301.24041812,  106.23305345,   50.35782934,   94.36710933,
        261.39458779,   23.42805573,  236.19903327,  215.71332122,
         72.68457433, 1674.82258065,  359.61558442,   92.43485034,
         55.15762106,  147.49710313,  161.68942521,   41.8235584 ,
         72.38555587,  124.39775226,  207.99019608,  599.71731449,
        410.15625   ,  146.23955432,   81.21546961,  151.60891089,
        265.91895803,   93.01442673,   59.58171206,   41.92334018,
         75.73638033,  100.18461538,  178.88385542,  176.9227836 ,
         35.15      ,   99.06164932,  435.15123095,  124.01737387,
        100.70515917,  113.01108647,  127.24327323,   34.53376496,
        113.68327138,   86.11075652,  317.00898411,  239.53629243,
         83.78780013,  276.0384821 ,   89.75808133,   32.18069662,
        250.71262136,  310.93798916,   76.84392204,  187.19391084,
        211.31315136,  165.0372093 ,   71.34651475,  403.21590909,
         35.59571978,  201.41721854,  126.01242971,   66.43719769,
       1425.49333333,  102.0477251 ,   39.45092251,   84.89571202,
         68.85702018,  148.00922935,  204.68155712,  104.81568627,
         66.45394046,  150.09968354,   32.68883529,   74.21318208,
        797.16806723,   93.0257416 ,  348.76102941,  372.37684004,
         95.12844828,   56.96902426,  143.82404692,  231.58237146,
        171.5491644 ,   65.69370442,   68.64634526,  119.36073553,
        128.91764706,   32.27093687,  114.79353994,  433.62242374,
         92.13242249,  293.19063545,  129.10751105,   86.49629995]
scaling_factors = np.array(scaling_factors)


# In[65]:


best_msg, best_d = decode_run(f5_dir, model_file, decoder, possible_labels, molbit_data_file, cnn_label_save_file,
                              overwrite=False, conf_thresh=0.95, batch_size=500, n_workers_cnn=30,
                              scaling_factors=scaling_factors, decoding_guarantee=9)


# In[ ]:


# The true message can gbe given as a bit string (just set true_msg directly) or as a list of molbits
molbits_in_set = [1, 2, 3,]  # etc.
true_msg = "".join([str(x) for x in get_read_counts(molbits_in_set, possible_labels=possible_labels)])[:32]


# In[67]:


print(best_msg)
print(true_msg)
print("".join(["0" if s1 == s2 else "1" for s1, s2 in zip(best_msg, true_msg)]))
print(sum([0 if s1 == s2 else 1 for s1, s2 in zip(best_msg, true_msg)]))


# In[ ]:




