import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..datasets import TDIUCDataset
from .datamodule_base import get_pretrained_tokenizer

class TDIUCDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):

        super().__init__()

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]

        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )
        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )
        self.tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

        self.dist = dist

    @property
    def dataset_cls(self):
        return TDIUCDataset

    @property
    def dataset_name(self):
        return "tdiuc"

    def setup(self, stage):

        self.train_dataset = self.dataset_cls('train', self.train_transform_keys, self.image_size, self.max_text_len, self.tokenizer)
        self.val_dataset = self.dataset_cls('val', self.val_transform_keys, self.image_size, self.max_text_len, self.tokenizer)

        if self.dist: # ddp中对不同进程分发数据
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
        else:
            self.train_sampler = None
            self.val_sampler = None

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader