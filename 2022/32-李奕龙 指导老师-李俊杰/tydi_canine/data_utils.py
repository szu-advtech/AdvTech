import h5py
import paddle
from tydi_canine.h5df_config import *


class TydiDataset(paddle.io.Dataset):
    """
    Construct Dataset Class for Canine TydiQA task.

    Args:
        sample_ids (List[int]): index of samples will be included in dataset, this is used for
            split train and dev set during training.
        h5df_path (str): The path of h5df storing input sample files.
    """

    def __init__(self,
                 sample_ids,
                 h5df_path: str = "/data/tydi/train_samples/train.h5df",
                 is_train=False
                 ):
        super(TydiDataset, self).__init__()
        self.sample_ids = sample_ids
        self.h5df_path = h5df_path
        self.feature_dataset = None
        self.label_dataset = None
        self.meta_dataset = None
        self.is_train = is_train

    def __getitem__(self, index):
        if self.feature_dataset is None:
            self.feature_dataset = h5py.File(self.h5df_path, 'r', swmr=True)[feature_group_name]
            if self.is_train:
                self.label_dataset = h5py.File(self.h5df_path, 'r', swmr=True)[label_group_name]
            else:
                self.meta_dataset = h5py.File(self.h5df_path, 'r', swmr=True)[meta_group_name]
        features = self.feature_dataset[self.sample_ids[index]]
        data = {
            'input_ids': features[0],
            'input_mask': features[1],
            'segment_ids': features[2],
        }
        if self.is_train:
            labels = self.label_dataset[self.sample_ids[index]]
            data.update({
                'start_positions': labels[0],
                'end_positions': labels[1],
                'answer_types': labels[2]
            })
        else:
            data['unique_ids'] = self.meta_dataset[self.sample_ids[index]][0]
        return data

    def __len__(self):
        return len(self.sample_ids)


def get_dataloader(sample_ids, h5df_path, batch_size, shuffle=False, is_train=False, num_workers=4,
                   batchify_fn=None):
    """
    Return the dataloader for training or testing.
    Args:
        sample_ids (List[int]): index of samples will be included in dataset, this is used for
            split train and dev set during training.
        h5df_path (str):  The path of h5df storing input sample files.
        batchify_fn (Callable): collate_fn for Dataloader.
        batch_size (int): data batch size.
        shuffle (bool): whether to shuffle the dataset.
        is_train (bool): whether the dataset is for training.
        num_workers (int): Number of workers for dataloader, since we use h5df, `num_workers>0` will improve
            training efficiency.
    Return:
        paddle.io.DataLoader
    """
    ds = TydiDataset(sample_ids=sample_ids, h5df_path=h5df_path, is_train=is_train)
    batch_sampler = paddle.io.DistributedBatchSampler(dataset=ds,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle)

    data_loader = paddle.io.DataLoader(dataset=ds,
                                       batch_sampler=batch_sampler,
                                       collate_fn=batchify_fn,
                                       num_workers=num_workers)
    return data_loader


if __name__ == "__main__":
    from tqdm import tqdm
    import time

    h5df_path = "../data/tydi/dev.h5df"
    with h5py.File(h5df_path, 'r') as fp:
        num_samples = fp['features'].len()
    print(num_samples)
    ids = list(range(num_samples))
    dataloader = get_dataloader(sample_ids=ids, h5df_path=h5df_path, batchify_fn=None, batch_size=16,
                                num_workers=4)
    # random.shuffle(ids)
    # for random seed 5% speed loss if gzip compression, 50% if not compression
    time1 = time.time()
    # chunked=True: 50800-2:30-340 it/s
    # chunked=1,3,len_seq: 3200 it/s
    for batch in tqdm(dataloader):
        break
        # 4 worder shuffle 550 swmr
        # 4 worder 720 swmr
    # 125.42
    print(f"{time.time() - time1:.2f}")
