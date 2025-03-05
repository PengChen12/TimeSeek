import warnings
import torch
from torch.utils.data import ConcatDataset, DataLoader
from src.data.batch_scheduler import BatchSchedulerSampler
from src.data.random_batch_scheduler_1channel import BatchSchedulerSampler_1channel
from src.data.pred_dataset import *
from src.data._batch_sampler import AdaptiveBatchSampler


class DataLoaders:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int=0,
        collate_fn=None,
        shuffle_train = True,
        shuffle_val = False
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size
        
        if "split" in dataset_kwargs.keys():
           del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val
    
        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()        
 
        
    def train_dataloader(self):
        return self._make_dloader("train", shuffle=self.shuffle_train)

    def val_dataloader(self):        
        return self._make_dloader("test", shuffle=self.shuffle_val)

    def test_dataloader(self):
        return self._make_dloader("test", shuffle=False)

    def _make_dloader(self, split, shuffle=False):
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)
        if len(dataset) == 0: return None
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            drop_last=False
        )

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )

    def add_dl(self, test_data, batch_size=None, **kwargs):
        # check of test_data is already a DataLoader
        from ray.train.torch import _WrappedDataLoader
        if isinstance(test_data, DataLoader) or isinstance(test_data, _WrappedDataLoader): 
            return test_data

        # get batch_size if not defined
        if batch_size is None: batch_size=self.batch_size        
        # check if test_data is Dataset, if not, wrap Dataset
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)        
        
        # create a new DataLoader from Dataset 
        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data



class DataProviders():
    def __init__(self, args):
        # config
        self.batch_size = args.batch_size
        self.drop_last = False
        self.dataset_list = args.dset_pretrain
        self.num_workers = args.num_workers
        self.size = [args.context_points, 0, args.target_points]
        self.features = args.features
        self.half = args.is_half
        self.all = args.is_all
        self.one_channel=args.one_channel
        self.dset_path=str(args.dset_path)+'/'
        self.img = args.img
        
        self.train = self.data_provider("train")
        if not self.all:
            self.valid = self.data_provider("val")
            self.test = self.data_provider("test")
        else:
            self.valid=None


    
    def concat_dataset(self,  split="train"):
        concat_dataset = []
        for dataset_name in self.dataset_list:
            if self.img:
                factory = Dataset_Custom_image
            else:
                factory = Dataset_Custom_SampleScale
            dataset_kwargs={
                    'root_path': self.dset_path,
                    'data_path': dataset_name + ".csv",
                    'features': self.features,
                    'scale': True,
                    'size': self.size,
                    'use_time_features': False,
                    'half':self.half,
                    'all':self.all,
                    'one_channel':self.one_channel
                    }
            dataset = factory(**dataset_kwargs,split=split)
            try:
                print(f'{dataset_name} len: ', len(dataset))
                if len(dataset) > 0:
                    concat_dataset.append(dataset)
            except:
                pass
            
        concat_dataset = ConcatDataset(concat_dataset)

        return concat_dataset

    def data_provider(self, split, shuffle=False):
        concat_dataset = self.concat_dataset(split=split)
        if self.one_channel:
            data_loader = DataLoader(
            dataset=concat_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True)
        else:
            # data_loader = DataLoader(
            # dataset=concat_dataset,
            # num_workers=self.num_workers,
            # batch_sampler=AdaptiveBatchSampler(concat_dataset, self.batch_size), # shuffle=true, drop_last=false
            # # sampler=BatchSchedulerSampler(dataset=concat_dataset, batch_size=self.batch_size))
            # )
            data_loader = DataLoader(
            dataset=concat_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=True)
        return data_loader





if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument('--dset_pretrain', type=list, default=['australian_electricity_demand_dataset_fillna(0)', 'kdd_cup_2018_dataset_without_missing_values_fillna(0)'], help='dataset name')
    parser.add_argument('--dset_path', type=str, default='/home/data/monash_csv_fillna')
    parser.add_argument('--context_points', type=int, default=96, help='sequence length')
    parser.add_argument('--target_points', type=int, default=96, help='forecast horizon')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for DataLoader')
    parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
    parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
    parser.add_argument('--one_channel', type=int, default=0)
    parser.add_argument('--is_half', type=float, default=1, help='half of the train_set')
    parser.add_argument('--is_all', type=float, default=0, help='half of the train_set')
    parser.add_argument('--img', type=float, default=0, help='half of the train_set')
    args = parser.parse_args()

    dp = DataProviders(args)
    for batch in dp.train:
        a, b = batch
        print(a.shape)
