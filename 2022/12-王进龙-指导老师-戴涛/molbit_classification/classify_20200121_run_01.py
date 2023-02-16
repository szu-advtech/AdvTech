#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' export CUDA_LAUNCH_BLOCKING=1')


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from matplotlib import pyplot as plt
import torch
import logging
import os
import re
import h5py
import numpy as np
import pandas as pd
import random
from torch import nn
import torchvision
import torch.nn.functional as F
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

logger = logging.getLogger(__name__)
if logger.handlers:
    logger.handlers = []
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())        


# In[4]:


device = torch.device('cuda')
torch.cuda.get_device_properties(0)


# In[5]:


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


# In[6]:


def median_smoothing(signal, width=3):
    s = signal[:]
    half_width = int(width/2)
    for i in range(half_width, len(signal)):
        s[i-half_width] = torch.median(signal[i-half_width:i+half_width])
    return s


# # Extract data to predict from fast5s

# ## Get run info

# In[7]:


def import_gdrive_sheet(gdrive_key, sheet_id):
    run_spreadsheet = pd.read_csv("https://docs.google.com/spreadsheet/ccc?key=" +                                   gdrive_key + "&output=csv&gid=" + sheet_id)
    if "date" in run_spreadsheet.columns:
        run_spreadsheet["date"] = run_spreadsheet["date"].astype(str)
    return run_spreadsheet

gdrive_key = "gdrive_key_here"
sheet_id = "0"
set_sheet_id = "512509543"

run_spreadsheet = import_gdrive_sheet(gdrive_key, sheet_id)
set_spreadsheet = import_gdrive_sheet(gdrive_key, set_sheet_id)


# In[9]:


run_name = "01_21_20_run_01"


# In[10]:


run_data = dict(run_spreadsheet[run_spreadsheet["run_name"] == run_name].iloc[0, :])


# In[11]:


basecall_dir = run_data.get("basecall_dir")
try:
    os.makedirs(os.path.join(basecall_dir, "cnn"))
except:
    pass


# In[12]:


label_file = list(run_spreadsheet[run_spreadsheet["run_name"] == run_name]["filtered_sw_labels"])[0]
molbit_all_data_file = f"/path/to/data/v4_test/{run_name}_all.hdf5"
molbit_labeled_data_file = f"/path/to/data/v4_test/{run_name}_only_labeled.hdf5"

cnn_label_file_all = os.path.join(basecall_dir, "cnn", "model_v4_0_1_all_reads_" + label_file.split("_")[-1])
cnn_label_file = os.path.join(basecall_dir, "cnn", "model_v4_0_1_labeled_reads_" + label_file.split("_")[-1])

print(cnn_label_file_all, "\t", cnn_label_file, "\t", molbit_all_data_file, "\t", molbit_labeled_data_file)


# ## Define functions for preprocessing data

# In[13]:


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


# In[14]:


def get_signal_regions_less_than(signal, signal_threshold=180.):
    signal_mask = np.where(signal <= signal_threshold, 1, 0)
    signal_chg_pts = np.where(np.abs(np.diff(signal_mask)) == 1)[0]

    if signal[0] <= signal_threshold:
        signal_chg_pts = np.insert(signal_chg_pts, 0, 0)
    if signal[-1] <= signal_threshold:
        signal_chg_pts = np.append(signal_chg_pts, len(signal_mask))

    signal_regions = list(zip(signal_chg_pts[::2], signal_chg_pts[1::2]))

    return signal_regions

def get_signal_regions_greater_than(signal, signal_threshold=180.):
    signal_mask = np.where(signal >= signal_threshold, 1, 0)
    signal_chg_pts = np.where(np.abs(np.diff(signal_mask)) == 1)[0]

    if signal[0] >= signal_threshold:
        signal_chg_pts = np.insert(signal_chg_pts, 0, 0)
    if signal[-1] >= signal_threshold:
        signal_chg_pts = np.append(signal_chg_pts, len(signal_mask))

    signal_regions = list(zip(signal_chg_pts[::2], signal_chg_pts[1::2]))

    return signal_regions

def median_smoothing(signal, width=3):
    s = signal[:]
    half_width = int(width/2)
    for i in range(half_width, len(signal)):
        s[i-half_width] = torch.median(signal[i-half_width:i+half_width])
    return s


# In[15]:


def trim_start_heuristic(signal, thresh=2, offset=20):
    try:
        above = np.where(signal[offset:] > thresh)[0][0] + offset
    except IndexError:
        above = 0
    return signal[above:]


# ## Import fast5s

# ### Get a list of fast5 files to import

# In[16]:


fast5_root_dir = list(run_spreadsheet[run_spreadsheet["run_name"] == run_name]["raw_fast5_dir_multi"])[0]
fast5_dirs = [os.path.join(fast5_root_dir, x) for x in os.listdir(fast5_root_dir) if "fast5" in x]
fast5_files = []
for f5_dir in fast5_dirs:
    fast5_files.extend([os.path.join(f5_dir, x) for x in os.listdir(f5_dir) if x.endswith("fast5")])


# ## Import, sorting by labeled vs. not

# In[17]:


all_data = []
all_fast5_read_ids = []

labeled_data = []
labeled_read_ids = []
labels = []
labels_in_order = []

max_len = 3000

sw = pd.read_csv(label_file, sep="\t", index_col=0)
labels = np.array(sw["molbit"])
read_ids = list(sw.index)

raw_lens = []

for f5_file in fast5_files:
    with h5py.File(f5_file, "r") as f5:
        for group in f5.get("/").values():
            read_id = re.findall(r'read_(.*)" \(', str(group))[0]
            all_fast5_read_ids.append(read_id)
            
            raw = group.get("Raw/Signal")
            raw_lens.append(len(raw))
            raw = raw[:15000]                    
                    
            x = torch.FloatTensor(list(raw))
            x = _scale_data(x)
            x = trim_start_heuristic(x)
            x = x[:max_len]
            y = torch.zeros(max_len)
            y[:len(x)] = x
            
            all_data.append(y)
            
            if read_id in sw.index:  
                i = read_ids.index(read_id)
                
                labeled_data.append(y)
                labeled_read_ids.append(read_id)
                labels_in_order.append(labels[i])


# In[18]:


all_fast5_data = torch.stack(all_data)
all_fast5_data = all_fast5_data.unsqueeze(1)


# In[19]:


labeled_data = torch.stack(labeled_data)
labeled_data = labeled_data.unsqueeze(1)


# In[20]:


with h5py.File(molbit_all_data_file, "w", swmr=True) as f:
    f.create_dataset("read_ids", shape=(len(all_fast5_read_ids), ), data=np.array(all_fast5_read_ids, dtype="S"))
    f.create_dataset("data", shape=all_fast5_data.shape, dtype=np.float, data=all_fast5_data)    


# In[21]:


with h5py.File(molbit_labeled_data_file, "w", swmr=True) as f:
    f.create_dataset("data", shape=labeled_data.shape, dtype=np.float, data=labeled_data)
    f.create_dataset("labels", shape=(len(labels_in_order), ), dtype=int, data=labels_in_order)
    f.create_dataset("read_ids", shape=(len(labeled_read_ids), ), data=np.array(labeled_read_ids, dtype="S"))


# ## Load saved data

# In[22]:


with h5py.File(molbit_all_data_file, "r") as f:
    all_fast5_read_ids = f.get("read_ids")[()]
    all_fast5_data = f.get("data")[()]


# In[23]:


with h5py.File(molbit_labeled_data_file, "r") as f:
    labeled_data = f.get("data")[()]
    labels_in_order = f.get("labels")[()]
    labeled_read_ids = f.get("read_ids")[()]


# # Create a MolbitDataset from the extracted data

# In[24]:


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


# In[25]:


md_all = MolbitDataset(molbit_all_data_file, unknown_labels=True)


# In[26]:


md_labeled = MolbitDataset(molbit_labeled_data_file, unknown_labels=True)


# # Load saved model

# In[27]:


class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        O_1 = 64
        O_2 = 128
        O_3 = 256
        O_4 = 512
        O_5 = 1024

        K_1 = 15
        K_2 = 8
        K_3 = 6
        K_4 = 4
        K_5 = 2

        KP_1 = 6
        KP_2 = 3
        KP_3 = 2
        KP_4 = 2
        KP_5 = 1

        FN_1 = 1000
        FN_2 = 500

        self.conv1 = nn.Sequential(nn.Conv1d(1, O_1, K_1, stride=1), nn.ReLU(),
                                   nn.AvgPool1d(KP_1))
        self.conv1_bn = nn.BatchNorm1d(O_1)

        self.conv2 = nn.Sequential(nn.Conv1d(O_1, O_2, K_2), nn.ReLU(),
                                   nn.AvgPool1d(KP_2))
        self.conv2_bn = nn.BatchNorm1d(O_2)

        self.conv3 = nn.Sequential(nn.Conv1d(O_2, O_3, K_3), nn.ReLU(),
                                   nn.AvgPool1d(KP_3))
        self.conv3_bn = nn.BatchNorm1d(O_3)

        self.conv4 = nn.Sequential(nn.Conv1d(O_3, O_4, K_4), nn.ReLU(),
                                   nn.AvgPool1d(KP_4))
        self.conv4_bn = nn.BatchNorm1d(O_4)

        self.conv5 = nn.Sequential(nn.Conv1d(O_4, O_5, K_5), nn.ReLU(),
                                   nn.AvgPool1d(KP_5))
        self.conv5_bn = nn.BatchNorm1d(O_5)

        self.gru1 = nn.GRU(input_size=92160, hidden_size=10, num_layers=1)

        self.fc1 = nn.Linear(37888, FN_1, nn.Dropout(0.5)) # 37888 20480 28672 9216
        self.fc1_bn = nn.BatchNorm1d(FN_1)

        self.fc2 = nn.Linear(FN_1, FN_2, nn.Dropout(0.5))
        self.fc2_bn = nn.BatchNorm1d(FN_2)

        self.fc3 = nn.Linear(FN_2, 96)

    def forward(self, x):
        x = x.float()
#         print("a", np.shape(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_bn(x))
#         print("b", np.shape(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_bn(x))
#         print("c", np.shape(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_bn(x))
#         print("d", np.shape(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.relu(self.conv4_bn(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.relu(self.conv5_bn(x))
#         print("e", np.shape(x))
        x = x.view(len(x), -1)
#         print("f", np.shape(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1_bn(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_bn(x))
#         print("g", np.shape(x))

        x = self.fc3(x)
#         print("h", np.shape(x))
        return x


# In[28]:


model_file = "/ssd1/home/kdorosch/code/punchcard-tagger/v4/molbit_classification/saved_models/molbit_classification_v4_0_1.20190827.pt"


# In[29]:


model = CNN()
model.load_state_dict(torch.load(model_file))
model.cuda()
model.eval()


# # Predict on all data

# In[30]:


test_loader_params = {"batch_size": 500,
          "shuffle": False,
          "num_workers": 30}
test_generator = DataLoader(md_all, **test_loader_params)


# In[31]:


y_pred_all = []
y_pred_score = []
for i, (local_batch, local_labels) in enumerate(test_generator):
    # Transfer to GPU
    local_batch = local_batch.to(device)
    y_pred = model(local_batch).to(dtype=torch.float64)
    
    softmax_score = torch.nn.functional.softmax(y_pred, dim=1)
    pred_score, prediction = torch.max(softmax_score, dim=1)

    y_pred_all.extend([int(x) if s > 0.9 else -1 for s, x in zip(pred_score.cpu(), prediction.cpu())])
    y_pred_score.extend([float(x) for x in pred_score.cpu()])


# In[32]:


len(np.where(np.array(y_pred_all) == -1)[0])


# In[33]:


len(y_pred_all)


# ## Save to tsv file

# In[34]:


cnn_df = pd.DataFrame()
cnn_df["read_id"] = all_fast5_read_ids.astype(str)
cnn_df["cnn_label"] = y_pred_all
cnn_df["cnn_score"] = y_pred_score


# In[35]:


cnn_df.to_csv(cnn_label_file_all, sep="\t", index=False)


# In[36]:


cnn_label_file_all


# ## Sum up read counts

# In[37]:


df = pd.read_csv(cnn_label_file_all, sep="\t")
y_pred_saved = list(df["cnn_label"])


# In[38]:


l_all, c_all = np.unique(y_pred_all, return_counts=True)
print(list(zip(l_all, c_all)))


# In[39]:


def plot_read_counts(labels, counts, labels_on_flowcell, labels_in_run,
                     possible_labels=None, ax=None, vmax=None, title_note=None):
    from matplotlib.patches import Patch
    if vmax is None:
        vmax = max(counts) + max(int(0.1 * max(counts)), 100)
    if ax == None:
        fig, ax = plt.subplots(figsize=(30, 8))
    if possible_labels is None:
        possible_labels = labels[:]
    g = sns.barplot(x=labels, y=counts, order=possible_labels, ax=ax)
    title = "#/reads identified per barcode"
    if title_note is not None:
        title += "\n(%s)" % title_note
    ax.set_title(title)
    ax.set_xlabel("Barcode ID")
    ax.set_ylabel("Read counts")
    ax.set_ylim([0, vmax])
    
    prev_text_height = 0
    for j, label in enumerate(possible_labels):
        if label in labels:
            count = counts[labels.index(label)]
        else:
            count = 0
        if label == "-1":
            continue
        
        if label in labels_in_run:
            g.containers[0].get_children()[j].set_facecolor("tab:red")
            font_kwargs = {"color": "tab:red", "weight": "bold"} 
        elif label in labels_on_flowcell:
            g.containers[0].get_children()[j].set_facecolor("tab:blue")
            font_kwargs = {"color": "k", "weight": "bold"}
        else:
            g.containers[0].get_children()[j].set_facecolor("k")
            font_kwargs = {"color": "k", "weight": "normal"}
            
        diff = prev_text_height - (count + 0.01 * vmax)
        if count < 100:
            text_height = count + .01 * vmax
            ax.text(j, text_height, count, ha="center", **font_kwargs)
        elif diff < 0 and np.abs(diff) < vmax * .06:
            text_height = np.max([prev_text_height + .035 * vmax, count + .01 * vmax])
            ax.text(j, text_height, count, ha="center", **font_kwargs)
        elif np.abs(diff) < vmax * .05:
            text_height = np.min([prev_text_height - .01 * vmax, count + .01 * vmax])
            ax.text(j, text_height, count, ha="center", **font_kwargs)
        else:
            text_height = count + .01 * vmax
            ax.text(j, text_height, count, ha="center", **font_kwargs)
        prev_text_height = text_height
        
    legend_elements = [Patch(facecolor='k', edgecolor='k',
                         label='never been run on this flowcell'),
                       Patch(facecolor='tab:blue', edgecolor='tab:blue',
                         label='prev run on flowcell'),
                       Patch(facecolor='tab:red', edgecolor='tab:red',
                         label='current run on flowcell')]
    leg = ax.legend(handles=legend_elements)
    t1, t2, t3 = leg.get_texts()
    t2._fontproperties = t1._fontproperties.copy()
    t3._fontproperties = t1._fontproperties.copy()
    t2.set_weight('bold')
    t3.set_weight('bold')
    t3.set_color("tab:red")
    return ax


# In[40]:


# Create set_N variables based on spreadsheet
molbit_sets = {}
for ix, row in set_spreadsheet.iterrows():
    set_no = re.findall(r"set ([\d]+)", row["set"])[0]
    molbits = row["molbits_in_set"]
    molbit_sets[set_no] = molbits.split(", ")


# In[41]:


run_data = dict(run_spreadsheet[run_spreadsheet["run_name"] == run_name].iloc[0, :])
molbit_set_in_run = str(int(run_data.get("molbit_set")))
molbit_sets_on_flowcell = run_data.get("prev_on_flowcell")

molbits_in_run = molbit_sets[molbit_set_in_run]
molbits_on_flowcell = molbits_in_run[:]
if molbit_sets_on_flowcell != "none":
    molbit_sets_on_flowcell = molbit_sets_on_flowcell.split(", ")
    for m in molbit_sets_on_flowcell:
        print(m)
        print(molbit_sets[m])
        molbits_on_flowcell.extend(molbit_sets[m])
    print(molbits_on_flowcell)


# In[42]:


molbits = [str(x) for x in l_all]
counts = list(c_all)
on_flowcell = [str(x) for x in molbits_on_flowcell]
in_run = [str(x) for x in molbits_in_run]
poss_molbits = [str(x) for x in range(96)]
plot_read_counts(molbits, counts,
                 on_flowcell, in_run,
                 possible_labels=poss_molbits,
#                  vmax=800,
                 title_note=f"{run_name} test tag: predictions from all reads (model v4_0_1)")
plt.savefig(f"/ssd1/home/kdorosch/code/punchcard-tagger/v4/plots/by_run/{run_name}_model_v4_0_1_preds.svg")


# ## Try with scaling factors

# In[43]:


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


# In[44]:


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


# In[45]:


molbits = [str(x) for x in l_all]
poss_molbits = [str(x) for x in range(96)]
counts = get_read_counts([str(x) for x in y_pred_all], possible_labels=poss_molbits)
counts = np.multiply(np.array(counts), scaling_factors)
counts = [int(x) for x in np.ceil(counts / sum(counts) * len(y_pred_all))]
on_flowcell = [str(x) for x in molbits_on_flowcell]
in_run = [str(x) for x in molbits_in_run]
plot_read_counts(poss_molbits, counts,
                 on_flowcell, in_run,
                 possible_labels=poss_molbits,
#                  vmax=800,
                 title_note=f"{run_name} test tag: predictions from all reads (model v4_0_1), scaled")
plt.savefig(f"/ssd1/home/kdorosch/code/punchcard-tagger/v4/plots/by_run/{run_name}_model_v4_0_1_preds_scaled.svg")


# In[46]:


sns.set(font_scale=1.1, style="white")


# In[47]:


def plot_read_counts_v2(labels, counts, labels_in_run,
                     possible_labels=None, ax=None, vmax=None, title_note=None):
    from matplotlib.patches import Patch
    
    bit0_facecolor = "tab:gray"
    bit0_textcolor = "tab:gray"
    bit1_facecolor = "tab:blue"
    bit1_textcolor = "k"
    
    if vmax is None:
        vmax = max(counts) + max(int(0.1 * max(counts)), 100)
    if ax == None:
        fig, ax = plt.subplots(figsize=(30, 8))
    if possible_labels is None:
        possible_labels = labels[:]
    g = sns.barplot(x=labels, y=counts, order=possible_labels, ax=ax)
    title = "#/reads identified per molbit"
    if title_note is not None:
        title += "\n(%s)" % title_note
    ax.set_title(title)
    ax.set_xlabel("Molbit ID")
    ax.set_ylabel("Read counts")
    ax.set_ylim([0, vmax])
    
    prev_text_height = 0
    for j, label in enumerate(possible_labels):
        if label in labels:
            count = counts[labels.index(label)]
        else:
            count = 0
        if label == "-1":
            continue
        
        if label in labels_in_run:
            g.containers[0].get_children()[j].set_facecolor(bit1_facecolor)
            font_kwargs = {"color": bit1_textcolor, "weight": "normal"} 
        else:
            g.containers[0].get_children()[j].set_facecolor(bit0_facecolor)
            font_kwargs = {"color": bit0_textcolor, "weight": "normal"}
            
        rotate_text = 0
        diff = prev_text_height - (count + 0.01 * vmax)
        if count < 100:
            text_height = count + .01 * vmax
            ax.text(j, text_height, count, ha="center", rotation=rotate_text, **font_kwargs)
        elif diff < 0 and np.abs(diff) < vmax * .06:
            text_height = np.max([prev_text_height + .035 * vmax, count + .01 * vmax])
            ax.text(j, text_height, count, ha="center", rotation=rotate_text, **font_kwargs)
        elif np.abs(diff) < vmax * .05:
            text_height = np.min([prev_text_height - .03 * vmax, count + .01 * vmax])
            ax.text(j, text_height, count, ha="center", rotation=rotate_text, **font_kwargs)
        else:
            text_height = count + .01 * vmax
            ax.text(j, text_height, count, ha="center", rotation=rotate_text, **font_kwargs)
        prev_text_height = text_height
        
    legend_elements = [Patch(facecolor=bit0_facecolor, edgecolor=bit0_facecolor,
                         label='0-bits'),
                       Patch(facecolor=bit1_facecolor, edgecolor=bit1_facecolor,
                         label='1-bits')]
    leg = ax.legend(handles=legend_elements)
    t1, t2 = leg.get_texts()
    t2._fontproperties = t1._fontproperties.copy()
    t2.set_weight('normal')
    t1.set_weight('normal')
    t2.set_color(bit1_textcolor)
    t1.set_color(bit0_textcolor)
    return ax


# In[48]:


list(zip(*np.unique(y_pred_saved, return_counts=True)))


# In[49]:


molbits = [str(x) for x in l_all]
poss_molbits = [str(x) for x in range(96)]
counts = get_read_counts([str(x) for x in y_pred_all], possible_labels=poss_molbits)
counts = np.multiply(np.array(counts), scaling_factors)
counts = [int(x) for x in np.ceil(counts / sum(counts) * len(y_pred_all))]
on_flowcell = [str(x) for x in molbits_on_flowcell]
in_run = [str(x) for x in molbits_in_run]


# In[50]:



plot_read_counts_v2(poss_molbits, counts,
                 in_run,
                 possible_labels=poss_molbits,
#                  vmax=800,
                 title_note=f"{run_name} test tag: predictions from all reads (model v4_0_1), scaled")
plt.savefig(f"/ssd1/home/kdorosch/code/punchcard-tagger/v4/plots/by_run/{run_name}_model_v4_0_1_preds_scaled_altplot.svg")


# In[ ]:





# In[ ]:





# # Predict on labeled data

# In[51]:


test_loader_params = {"batch_size": 500,
          "shuffle": False,
          "num_workers": 30}
test_generator = DataLoader(md_labeled, **test_loader_params)


# In[52]:


torch.cuda.empty_cache()


# In[53]:


y_pred_labeled = []
y_true_labeled = []
y_pred_score = []
for i, (local_batch, local_labels) in enumerate(test_generator):
    # Transfer to GPU
    local_batch = local_batch.to(device)
    local_labels = local_labels.to(device)
    y_pred = model(local_batch).to(dtype=torch.float64)
    local_labels = local_labels.to(dtype=torch.long)
    
    softmax_score = torch.nn.functional.softmax(y_pred, dim=1)
    pred_score, prediction = torch.max(softmax_score, dim=1)

    y_pred_labeled.extend([int(x) for x in prediction.cpu()])
    y_true_labeled.extend([int(x) for x in local_labels.cpu()])
    y_pred_score.extend([float(x) for x in pred_score.cpu()])


# In[54]:


cnn_df_labeled = pd.DataFrame()
cnn_df_labeled["read_id"] = labeled_read_ids.astype(str)
cnn_df_labeled["cnn_label"] = y_pred_labeled
cnn_df_labeled["cnn_score"] = y_pred_score
cnn_df_labeled["sw_label"] = y_true_labeled


# In[55]:


cnn_df_labeled.to_csv(cnn_label_file, sep="\t", index=False)


# In[56]:


cnn_label_file


# # Evaluate error

# In[57]:


from sklearn.metrics import confusion_matrix


# In[58]:


np.sum([x == y for x, y in zip(y_true_labeled, y_pred_labeled)]), len(y_true_labeled)


# In[59]:


1. * np.sum([x == y for x, y in zip(y_true_labeled, y_pred_labeled)]) / len(y_true_labeled)


# In[60]:


l, c = np.unique(y_true_labeled, return_counts=True)


# In[61]:


norm = []
for molbit in range(96):
    if molbit in l:
        i = list(l).index(molbit)
        norm.append(c[i])
    else:
        norm.append(1)
norm = np.array(norm)


# In[62]:


cm = confusion_matrix(y_true_labeled, y_pred_labeled, labels=range(96))


# In[64]:


fig, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(cm / norm[:, np.newaxis], vmin=0, vmax=0.01)
# plt.savefig(f"/ssd1/home/kdorosch/code/punchcard-tagger/v4/plots/by_run/{run_name}_model_v4_0_1_cm.svg")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




