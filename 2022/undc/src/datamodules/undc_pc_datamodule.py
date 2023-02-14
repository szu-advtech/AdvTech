from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from src.datamodules.components.pc_dataset import ABC_pc_hdf5
from src import utils

log = utils.get_pylogger(__name__)

class UNDC_PC_DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "",
        batch_size: int = 1,  # Must be 1
        grid_size: int = 64,
        num_points: int = 4096,
        knn_num: int = 8,
        pooling_radius: int = 2,
        receptive_padding: int = 3,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        if not self.data_train :
            log.info("Loading Train dataset...")
            self.data_train = ABC_pc_hdf5(
                self.hparams.data_dir, 
                self.hparams.num_points, 
                self.hparams.grid_size, 
                self.hparams.knn_num,
                self.hparams.pooling_radius, 
                train=True)
            log.info(f"Train dataset [{type(self.data_train).__name__}] of size {len(self.data_train)} had been created.")

        if not self.data_val :
            log.info("Loading Val dataset...")
            self.data_val = ABC_pc_hdf5(
                self.hparams.data_dir, 
                self.hparams.num_points, 
                self.hparams.grid_size, 
                self.hparams.knn_num,
                self.hparams.pooling_radius, 
                train=False)
            log.info(f"Val dataset [{type(self.data_val).__name__}] of size {len(self.data_val)} had been created.")

        if not self.data_test :
            log.info("Loading Test dataset...")
            self.data_test = ABC_pc_hdf5(
                self.hparams.data_dir, 
                self.hparams.num_points, 
                self.hparams.grid_size, 
                self.hparams.knn_num,
                self.hparams.pooling_radius, 
                train=False)
            log.info(f"Test dataset [{type(self.data_test).__name__}] of size {len(self.data_test)} had been created.")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(
        root / "configs" / "datamodule" / "undc_pc.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
