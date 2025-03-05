import os
import torch
from .data_factory import DataProvider


class DataLoaders:
    def __init__(
        self,
        dataset: str,
        data_path: str,
        percentage: float,
        batch_size: int,
        win_size: int,
    ):
        super().__init__()
        self.dataset = dataset
        self.data_path = data_path
        self.batch_size = batch_size
        self.percentage = percentage
        self.win_size = win_size
    
        _ , self.train = self.train_dataloader()
        _ , self.valid = self.val_dataloader()
        _ , self.test = self.test_dataloader() 
        _ , self.init = self.init_dataloader()      
 
        
    def train_dataloader(self):
        return self._make_dloader("train")

    def val_dataloader(self):        
        return self._make_dloader("val")

    def test_dataloader(self):
        return self._make_dloader("test", step=self.win_size)
    
    def init_dataloader(self):
        return self._make_dloader("init", step=self.win_size)


    def _make_dloader(self, split, step=1):
        data_set, data_loader = DataProvider(
                root_path=self.data_path,
                datasets=self.dataset,
                batch_size=self.batch_size,
                win_size=self.win_size,
                step=step,
                mode=split,
                percentage=self.percentage,
            )
        return data_set, data_loader

def get_dls_ad(params):
    return DataLoaders(dataset = params.dset_finetune, data_path=params.dset_path, percentage=params.percentage, 
                       batch_size=params.batch_size, win_size=params.win_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument('--dset_finetune', type=str, default='MSL', help='dataset name')
    parser.add_argument('--dset_path', type=str, default='/home/bigmodel/Decoder_version_1/data/ad_datasets', help='dataset name')
    parser.add_argument('--win_size', type=int, default=100, help='sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--percentage', type=float, default=0.5, help='dataset path')

    args = parser.parse_args()

    dp = get_dls_ad(args)
    for batch in dp.train:
        a, b = batch
        print(a.shape, b.shape)