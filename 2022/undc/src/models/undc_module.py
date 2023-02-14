from typing import Any, List

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np

from src.utils.ndc_utils import write_obj_triangle, dual_contouring_undc_test,postprocessing
import os

GEN_OBJ_PRE_BATCH = 10
GEN_OBJ = True

class UNDCLitModule(LightningModule):

    def __init__(
        self,
        net: torch.nn.Module,
        net_bool: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        out_dir: str,
        train_float: bool,
        net_bool_pth: str,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net","net_bool"])
        self.out_dir = out_dir
        self.val_count = 0
        self.train_float = train_float
        self.net = net
        self.net_bool = net_bool
        if train_float:
            self.net_bool.load_state_dict(torch.load(net_bool_pth),strict=False)
            self.net_bool.eval()
        self.criterion_bool = torch.nn.BCELoss()
        self.criterion_float = torch.nn.MSELoss()
        # loss function

        # metric objects for calculating and averaging accuracy across batches
        self.train_bool_acc = MeanMetric()
        self.val_bool_acc = MeanMetric()
        self.test_bool_acc = MeanMetric()

        # for averaging loss across batches
        self.train_bool_loss = MeanMetric()
        self.val_bool_loss = MeanMetric()
        self.test_bool_loss = MeanMetric()
        self.train_float_loss = MeanMetric()
        self.val_float_loss = MeanMetric()
        self.test_float_loss = MeanMetric()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_bool_acc_best = MaxMetric()
        self.epoch_idx = 1
        self.objs_id = 0

    def forward(self, x: torch.Tensor):
        return self.net(x) 

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_bool_acc_best.reset()


    def step(self, batch: Any, val=False):
        pc_KNN_idx_,pc_KNN_xyz_, voxel_xyz_int_,voxel_KNN_idx_,voxel_KNN_xyz_, \
            gt_output_bool_,gt_output_float_,gt_output_float_mask_ = batch
        # * [1,32768] -> [32768]
        pc_KNN_idx = pc_KNN_idx_[0] 
        # * [1,32768, 3] -> [32768, 3]
        pc_KNN_xyz = pc_KNN_xyz_[0] 
        # * [1, 9127, 3] -> [9127, 3]
        voxel_xyz_int = voxel_xyz_int_[0] 
        # * [1, 73016] -> [73016]
        voxel_KNN_idx = voxel_KNN_idx_[0] 
        # * [1, 73016, 3] -> [73016, 3]
        voxel_KNN_xyz = voxel_KNN_xyz_[0] 
        # * [1, 9127, 3] -> [9127, 3]
        gt_output_bool = gt_output_bool_[0] 
        # * [1, 9127, 3] -> [9127, 3]
        gt_output_float = gt_output_float_[0] 
        gt_output_float_mask = gt_output_float_mask_[0]
        pred_output_bool, pred_output_float = self.net(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)
        
        # print(f"pred_bool: {pred_output_bool.shape}")
        # print(f"pred_float: {pred_output_float.shape}")
        if self.train_float:
            if val:
                pred_output_bool, _ = self.net_bool(pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz)
                bool_loss = self.criterion_bool(pred_output_bool, gt_output_bool) 
                bool_acc = torch.mean(gt_output_bool*(pred_output_bool>0.5).float()+(1-gt_output_bool)*(pred_output_bool<=0.5).float())
            else:
                bool_loss = torch.zeros(1)
                bool_acc = torch.zeros(1)
            distance = ((pred_output_float-gt_output_float)**2 )*gt_output_float_mask # MSE
            float_loss = torch.sum(distance)/torch.clamp(torch.sum(gt_output_float_mask),min=1)
            return bool_loss, bool_acc, pred_output_bool, float_loss, pred_output_float
        else:
            bool_loss = self.criterion_bool(pred_output_bool, gt_output_bool) 
            bool_acc = torch.mean(gt_output_bool*(pred_output_bool>0.5).float()+(1-gt_output_bool)*(pred_output_bool<=0.5).float())
            float_loss = torch.zeros(1)
            return bool_loss, bool_acc, pred_output_bool, float_loss, pred_output_float

    def training_step(self, batch: Any, batch_idx: int):
        bool_loss, bool_acc, pred_bool, float_loss, pred_float = self.step(batch)
        if self.train_float:
            self.train_float_loss(float_loss)
            self.log("train/float_loss", self.train_float_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/loss", self.train_float_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            return {"loss": float_loss}
        else:
            self.train_bool_loss(bool_loss)
            self.train_bool_acc(bool_acc)
            self.log("train/bool_loss", self.train_bool_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/bool_acc", self.train_bool_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/loss", self.train_bool_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            return {"loss": bool_loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        if not self.train_float:
            torch.save(self.net.state_dict(),f"{self.out_dir}/net_bool-{self.epoch_idx}.pth")
        self.epoch_idx += 1

    def validation_step(self, batch: Any, batch_idx: int):
        bool_loss, bool_acc, pred_bool, float_loss, pred_float = self.step(batch,val=True)
        if self.train_float:
            self.val_float_loss(float_loss)
            self.log("val/float_loss", self.val_float_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/loss", self.val_float_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if GEN_OBJ and batch_idx % GEN_OBJ_PRE_BATCH == 0 :
                self.gen_obj(pred_bool, pred_float, batch)
            return {"loss": float_loss}
        else:
            self.val_bool_loss(bool_loss)
            self.val_bool_acc(bool_acc)
            self.log("val/bool_loss", self.val_bool_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/bool_acc", self.val_bool_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/loss", self.val_bool_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if GEN_OBJ and batch_idx % GEN_OBJ_PRE_BATCH == 0 :
                self.gen_obj(pred_bool, pred_float, batch)
            return {"loss": bool_loss}

    def validation_epoch_end(self, outputs: List[Any]):
        bool_acc = self.val_bool_acc.compute()
        self.val_bool_acc_best(bool_acc)
        self.log("val/acc_bool_best", self.val_bool_acc_best.compute(), prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int):
        bool_loss, bool_acc, pred_bool, float_loss, pred_float = self.step(batch,val=True)
        if self.train_float:
            self.test_float_loss(float_loss)
            self.log("test/float_loss", self.test_float_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/loss", self.test_float_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if GEN_OBJ and batch_idx % GEN_OBJ_PRE_BATCH == 0 :
                self.gen_obj(pred_bool, pred_float, batch)
            return {"loss": float_loss}
        else:
            self.test_bool_loss(bool_loss)
            self.test_bool_acc(bool_acc)
            self.log("test/bool_loss", self.test_bool_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/bool_acc", self.test_bool_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/loss", self.test_bool_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            if GEN_OBJ and batch_idx % GEN_OBJ_PRE_BATCH == 0 :
                self.gen_obj(pred_bool, pred_float, batch)
            return {"loss": bool_loss}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            monitor = ""
            if self.train_float:
                monitor = "val/float_loss"
            else :
                monitor = "val/bool_loss"

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def generate_obj_test(
        self,
        voxel_xyz_int,
        pred_bool,
        pred_float, 
        name,
        grid_size:int =64,
        post_process=True
    ):
        device = "cpu"
        pred_bool_grid = torch.zeros([grid_size+1,grid_size+1,grid_size+1,3], dtype=torch.int32, device=device)
        pred_float_grid = torch.full([grid_size+1,grid_size+1,grid_size+1,3], 0.5, dtype=torch.float32, device=device)

        pred_bool_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = (pred_bool > 0.5).int()
        pred_float_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = pred_float

        if post_process:
            pred_bool_grid = postprocessing(pred_bool_grid)

        pred_bool_numpy = pred_bool_grid.detach().numpy()
        pred_float_numpy = pred_float_grid.detach().numpy()

        pred_float_numpy = np.clip(pred_float_numpy,0,1)
        vertices, triangles = dual_contouring_undc_test(pred_bool_numpy, pred_float_numpy)

        if not os.path.exists(f"{self.hparams.out_dir}/objs/{self.objs_id}"):
            os.makedirs(f"{self.hparams.out_dir}/objs/{self.objs_id}")
        write_obj_triangle(f"{self.hparams.out_dir}/objs/{self.objs_id}/{name}.obj", vertices, triangles)

    def gen_obj(self,pred_bool, pred_float, batch):

        pc_KNN_idx_, pc_KNN_xyz_, voxel_xyz_int_, voxel_KNN_idx_, voxel_KNN_xyz_, \
            gt_output_bool_, gt_output_float_, gt_output_float_mask_ = batch

        voxel_xyz_int = voxel_xyz_int_[0].cpu()
        gt_output_bool = gt_output_bool_[0].cpu()
        gt_output_float = gt_output_float_[0].cpu()
        pred_bool = pred_bool.detach().cpu()
        pred_float = pred_float.detach().cpu()

        self.generate_obj_test(voxel_xyz_int,gt_output_bool,gt_output_float, "gt")
        if self.train_float:
            self.generate_obj_test(voxel_xyz_int,gt_output_bool,pred_float,"gt_float")
            self.generate_obj_test(voxel_xyz_int,pred_bool,pred_float,"bool_float")
        else:
            self.generate_obj_test(voxel_xyz_int,pred_bool,gt_output_float,"bool_gt")

        self.objs_id += 1


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "undc.yaml")
    _ = hydra.utils.instantiate(cfg)
