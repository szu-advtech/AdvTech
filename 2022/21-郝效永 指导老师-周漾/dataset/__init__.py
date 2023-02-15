import importlib
import torch.utils.data
from dataset.dataset import SelfDataset

def create_dataloader():
    dataset = SelfDataset('./dataset/deepfashionHD/')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    return dataloader