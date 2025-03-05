from torch.utils.data import RandomSampler, Sampler
from numpy import random


class AdaptiveBatchSampler(Sampler):
    def __init__(self, concat_dataset, batch_size, nums=None, repeat=False):
        """
        input:
            concat_dataset (torch.utils.data.ConcatDataset)
            batch_size (int): 'batch_size' univariate data, if multivariate data, 'adaptivate batch size' = 'batch_size' // number_of_channels
            [nums (int)]: Maximum number of samples. Default-None: 
            [repeat (bool)]: Default-False: The dataset is not repeated sampled. If True, smaller datasets may have repeated sampling

        Note:
            Dataset class must have self.number_of_channels
        
        data_loader = torch.utils.data.DataLoader(
            dataset=concat_dataset,
            batch_sampler=AdaptiveBatchSampler(concat_dataset, batch_size),
            num_workers=8,
        )

        """
        # dataset info
        self.concat_dataset = concat_dataset
        self.number_of_datasets = len(self.concat_dataset.datasets)
        self.child_dataset_start_index = [0] + self.concat_dataset.cumulative_sizes[:-1]

        self.batch_size = batch_size
        self.total_nums = nums if nums is not None else self.concat_dataset.cumulative_sizes[-1]
        self.cur_number_of_samples = 0
        self.repeat = repeat

        self.child_datasets = list(range(self.number_of_datasets))
      
        self.samplers = []
        self.sampler_iterators = []
        for dataset_id in self.child_datasets:
            cur_dataset = self.concat_dataset.datasets[dataset_id]
            sampler = RandomSampler(cur_dataset)
            self.samplers.append(sampler)
            self.sampler_iterators.append(sampler.__iter__())
               
    
    def _next_batch(self):
        cur_batch_samples = []
        while self.cur_number_of_samples < self.total_nums and self.child_datasets:
            random_dataset_id = random.choice(self.child_datasets)
            cur_batch_sampler = self.sampler_iterators[random_dataset_id]
            adaptive_batch_size = max(self.batch_size // self.concat_dataset.datasets[random_dataset_id].number_of_channels, 1)
            for _ in range(adaptive_batch_size):
                try:
                    cur_sample_index = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_index + self.child_dataset_start_index[random_dataset_id]
                    cur_batch_samples.append(cur_sample)
                    self.cur_number_of_samples += 1
                except StopIteration:
                    if self.repeat: # restart the iterator
                        self.sampler_iterators[random_dataset_id] = self.samplers[random_dataset_id].__iter__()
                        cur_batch_sampler = self.sampler_iterators[random_dataset_id]
                        cur_sample_index = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_index + self.child_dataset_start_index[random_dataset_id]
                        cur_batch_samples.append(cur_sample)
                        self.cur_number_of_samples += 1
                    else:
                        self.child_datasets.remove(random_dataset_id)
                        break
            if cur_batch_samples: break
            
        return cur_batch_samples


    def _reset(self):
        self.cur_number_of_samples = 0
        if not self.repeat: self.child_datasets = list(range(self.number_of_datasets))
        for dataset_id in self.child_datasets:
            self.sampler_iterators[dataset_id] = self.samplers[dataset_id].__iter__()


    def __iter__(self):
        while True:
            cur_batch_samples = self._next_batch()
            if cur_batch_samples: 
                yield cur_batch_samples
            else: 
                break
        self._reset()
    

    def __len__(self):
        return self.total_nums
