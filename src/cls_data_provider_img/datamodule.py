import os
import torch
from .datautils import load_UCR, load_UEA
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np



class DataLoaders:
    def __init__(
        self,
        loader:str,
        dataset: str,
        data_path: str,
        percentage: float,
        batch_size: int,
    ):
        super().__init__()
        self.loader = loader
        self.dataset = dataset
        self.data_path = data_path
        self.batch_size = batch_size
        self.percentage = percentage
    
        self.train, self.label_num = self.train_dataloader()
        self.valid, self.label_num  = self.test_dataloader()
        self.test, self.label_num = self.test_dataloader()   
        self.sample_len, self.n_vars = self._get_dim(self.train)
 
        
    def train_dataloader(self):
        return self._make_dloader("train")

    def test_dataloader(self):
        return self._make_dloader("test")

    
    def _get_dim(self, dl):
        i, batch = next(enumerate(self.train))
        a, b, c = batch
        return a.shape[1], a.shape[2]

    def _make_dloader(self, split):
        if self.loader == 'UCR':
            train_data, train_labels, train_img, test_data, test_labels, test_img = load_UCR(self.dataset, self.data_path)
        elif self.loader == 'UEA':
            train_data, train_labels, test_data, test_labels = load_UEA(self.dataset, self.data_path)

        if split == 'train':
            # train_data , train_labels = self._get_percentage_data(train_data, train_labels, self.percentage)
            label_num = self._get_label_num(train_labels, test_labels)
            train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float),
                                      F.one_hot(torch.from_numpy(train_labels).to(torch.long), num_classes=label_num).to(torch.float), 
                                      torch.stack(train_img, dim=0))
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False), label_num
        elif split == 'test':
            label_num = self._get_label_num(train_labels, test_labels)
            test_dataset = TensorDataset(torch.from_numpy(test_data).to(torch.float),
                                     F.one_hot(torch.from_numpy(test_labels).to(torch.long), num_classes=label_num).to(torch.float),
                                     torch.stack(test_img, dim=0))
            return DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False), label_num


    def _get_percentage_data(self, data, labels, percentage):
        split = train_test_split(
            data, labels,
            train_size=percentage, random_state=0, stratify=labels, shuffle=True
            )
        data = split[0]
        labels = split[2]
        return data, labels
    
    def _get_label_num(self, train_labels, test_labels):
        label_num1 = len(np.unique(train_labels))
        label_num2 = len(np.unique(test_labels))
        if label_num1 > label_num2:
            label_num2 = label_num1
        return label_num2


def get_dls_cls(params):
    return DataLoaders(
        loader=params.loader, 
        dataset = params.dset_finetune, 
        data_path=params.dset_path, 
        percentage=params.percentage, 
        batch_size=params.batch_size
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument('--dset_finetune', type=str, default='Beef', help='dataset name')
    parser.add_argument('--loader', type=str, default='UCR', help='UCR or UEA')
    parser.add_argument('--dset_path', type=str, default='/home/bigmodel/Decoder_version_1/data/cls_datasets_img/', help='dataset name')
    parser.add_argument('--target_points', type=int, default=96, help='sequence length')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--percentage', type=float, default=30, help='dataset path')
    args = parser.parse_args()

    dp = get_dls_cls(args)
    for batch in dp.train:
        a, b, c = batch
        print(a.shape, b.shape, c.shape)