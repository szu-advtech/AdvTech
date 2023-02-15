from torch.utils.data import Dataset
import numpy as np
import os
from sklearn import preprocessing
from scipy.fftpack import fft


class SignalDataset(Dataset):
    def __init__(self, filelist_path):
        super(SignalDataset, self).__init__()
        self.filelist = []
        with open(filelist_path) as file:
            for line in file:
                self.filelist.append(line.rstrip())

    def __len__(self):
        return len(self.filelist)

    """
    return 1xT signal data
    """
    def __getitem__(self, i):
        return np.copy(np.load(self.filelist[i], mmap_mode='r'))


class LabelDataset(Dataset):
    """
    暂时训练固定长度的信号-label数据
    """
    def __init__(self, dataset_dir, read_limit=None, is_validate=False, use_fp32=False):
        super(LabelDataset, self).__init__()
        if is_validate:
            chunks_path = os.path.join(dataset_dir, 'validation', 'chunks.npy')
            seqs_npy_path = os.path.join(dataset_dir, 'validation', 'references.npy')
            seqs_len_npy_path = os.path.join(dataset_dir, 'validation', 'reference_lengths.npy')
        else:
            chunks_path = os.path.join(dataset_dir, 'chunks.npy')
            seqs_npy_path = os.path.join(dataset_dir, 'references.npy')
            seqs_len_npy_path = os.path.join(dataset_dir, 'reference_lengths.npy')

        self.chunks = np.load(chunks_path, mmap_mode='r')
        self.seqs = np.load(seqs_npy_path, mmap_mode='r')
        self.seqs_len = np.load(seqs_len_npy_path, mmap_mode='r')
        self.read_count = len(self.chunks)
        if read_limit is not None:
            self.read_count = read_limit
        self.use_fp32 = use_fp32

    def __len__(self):
        return self.read_count

    def __getitem__(self, i):
        signal = np.copy(self.chunks[i])
        seqs   = np.copy(self.seqs[i])
        seqs[seqs == 0] = 1  # nn.CTC_loss don't allow padding_value is blank_id
        seqs_len = np.copy(self.seqs_len[i])
        if self.use_fp32:
            return (
                signal.astype(np.float32),
                seqs.astype(np.long),
                seqs_len.astype(np.long)
            )
        else:
            return (
                signal.astype(np.float16),
                seqs.astype(np.long),
                seqs_len.astype(np.long)
            )


class LabelUnalignDataset(Dataset):
    def __init__(self, dataset_dir, read_limit=None, is_validate=False, use_fp32=False):
        super(LabelUnalignDataset, self).__init__()
        if is_validate:
            signal_path = os.path.join(dataset_dir, 'validation', 'signal_data.npy')
            signal_lengths_path = os.path.join(dataset_dir, 'validation', 'signal_lengths.npy')
            target_path = os.path.join(dataset_dir, 'validation', 'target_data.npy')
            target_lengths_path = os.path.join(dataset_dir, 'validation', 'target_lengths.npy')
        else:
            signal_path = os.path.join(dataset_dir, 'signal_data.npy')
            signal_lengths_path = os.path.join(dataset_dir, 'signal_lengths.npy')
            target_path = os.path.join(dataset_dir, 'target_data.npy')
            target_lengths_path = os.path.join(dataset_dir, 'target_lengths.npy')

        self.signal_data = np.load(signal_path, mmap_mode='r')
        self.target_data = np.load(target_path, mmap_mode='r')
        self.signal_lengths = np.load(signal_lengths_path, mmap_mode='r')
        self.target_lengths = np.load(target_lengths_path, mmap_mode='r')

        self.read_count = len(self.signal_data)
        if read_limit is not None:
            self.read_count = read_limit
        self.use_fp32 = use_fp32

    def __len__(self):
        return self.read_count

    def __getitem__(self, i):
        signal = np.copy(self.signal_data[i])
        target = np.copy(self.target_data[i])
        target[target == 0] = 1  # CTC_loss don't allow padding_value is blank_id
        signal_len = np.copy(self.signal_lengths[i])
        target_len = np.copy(self.target_lengths[i])
        if self.use_fp32:
            return (
                signal.astype(np.float32),
                target.astype(np.long),
                signal_len.astype(np.long),
                target_len.astype(np.long)
            )
        else:
            return (
                signal.astype(np.float16),
                target.astype(np.long),
                signal_len.astype(np.long),
                target_len.astype(np.long)
            )


class STFT_Dataset(Dataset):
    def stft_transform(self, raw_data):
        x = np.linspace(0, 40 - 1, 40, dtype=np.int64)
        w = 0.54 - 0.46 * np.cos(2 * np.pi * x / (40 - 1))  # 汉明窗
        fs = 4000
        time_window = 10  # 单位ms
        step = 1

        range0_end = int(len(raw_data) - time_window * fs / 1000) // step
        data_input = np.zeros((range0_end, 20), dtype=np.float64)  # 用于存放最终的频率特征数据
        data_line = np.zeros((1, 40), dtype=np.float64)
        for i in range(0, range0_end):
            p_start = i * step
            p_end = p_start + 40
            data_line = raw_data[p_start:p_end]
            data_line = data_line * w  # 加窗
            data_line = np.abs(fft(data_line))
            data_input[i] = data_line[0:20]
        data_input = np.log(data_input + 1)
        data_input = preprocessing.scale(data_input)
        return data_input

    """
    For normalized bonito dataset 
    """
    def __init__(self, dataset_dir, read_limit=None, is_validate=False, use_fp32=False):
        super(STFT_Dataset, self).__init__()
        if is_validate:
            chunks_path = os.path.join(dataset_dir, 'validation', 'chunks.npy')
            seqs_npy_path = os.path.join(dataset_dir, 'validation', 'references.npy')
            seqs_len_npy_path = os.path.join(dataset_dir, 'validation', 'reference_lengths.npy')
        else:
            chunks_path = os.path.join(dataset_dir, 'chunks.npy')
            seqs_npy_path = os.path.join(dataset_dir, 'references.npy')
            seqs_len_npy_path = os.path.join(dataset_dir, 'reference_lengths.npy')

        self.chunks = np.load(chunks_path, mmap_mode='r')
        self.read_count = len(self.chunks)
        if read_limit is not None:
            self.read_count = read_limit
        self.seqs = np.load(seqs_npy_path, mmap_mode='r')
        self.seqs_len = np.load(seqs_len_npy_path, mmap_mode='r')
        self.use_fp32 = use_fp32

    def __len__(self):
        return self.read_count

    def __getitem__(self, i):
        raw_signal = np.copy(self.chunks[i])
        signal_stft = self.stft_transform(raw_signal) # T x stft_feature_num
        seqs = np.copy(self.seqs[i])
        seqs[seqs == 0] = 1  # nn.CTC_loss don't allow padding_value is blank_id
        seqs_len = np.copy(self.seqs_len[i])
        if self.use_fp32:
            return (
                signal_stft.astype(np.float32),
                seqs.astype(np.long),
                seqs_len.astype(np.long)
            )
        else:
            return (
                signal_stft.astype(np.float16),
                seqs.astype(np.long),
                seqs_len.astype(np.long)
            )
