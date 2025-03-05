import math
from torch.utils.data import RandomSampler, Sampler
import numpy as np
import random


class BatchSchedulerSampler_1channel(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])
        self.each_dataset_size=[]
        for cur_dataset in dataset.datasets:
            self.each_dataset_size=np.append(self.each_dataset_size,len(cur_dataset)) 
        self.dataset_p=self.each_dataset_size/(float(sum(self.each_dataset_size)))
        # print('self.dataset_p:'+str(self.dataset_p))

    def __len__(self):
        # [] max len
        # return self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets)
        return int(sum(self.batch_size * np.ceil(self.each_dataset_size/ self.batch_size)))

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        # epoch_samples = self.largest_dataset_size * self.number_of_datasets
        epoch_samples =int(sum(self.each_dataset_size))

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0,epoch_samples):
            choice_index = self.roulette_wheel_selection()
            cur_batch_sampler = sampler_iterators[choice_index]
            try:
                cur_sample_org = cur_batch_sampler.__next__()
                cur_sample = cur_sample_org + push_index_val[choice_index]
                final_samples_list.append(cur_sample)
            except StopIteration:
                # got to the end of iterator - restart the iterator and continue to get samples
                # until reaching "epoch_samples"
                sampler_iterators[choice_index] = samplers_list[choice_index].__iter__()
                cur_batch_sampler = sampler_iterators[choice_index]
                cur_sample_org = cur_batch_sampler.__next__()
                cur_sample = cur_sample_org + push_index_val[choice_index]
                final_samples_list.append(cur_sample)
        return iter(final_samples_list)


        # for _ in range(0, epoch_samples, step):
        #     for i in range(self.number_of_datasets):
        #         choice_index = self.roulette_wheel_selection()
        #         cur_batch_sampler = sampler_iterators[choice_index]
        #         cur_samples = []
        #         for _ in range(samples_to_grab):
        #             try:
        #                 cur_sample_org = cur_batch_sampler.__next__()
        #                 cur_sample = cur_sample_org + push_index_val[choice_index]
        #                 cur_samples.append(cur_sample)
        #             except StopIteration:
        #                 # got to the end of iterator - restart the iterator and continue to get samples
        #                 # until reaching "epoch_samples"
        #                 sampler_iterators[choice_index] = samplers_list[choice_index].__iter__()
        #                 cur_batch_sampler = sampler_iterators[choice_index]
        #                 cur_sample_org = cur_batch_sampler.__next__()
        #                 cur_sample = cur_sample_org + push_index_val[choice_index]
        #                 cur_samples.append(cur_sample)
        #         final_samples_list.extend(cur_samples)

        # return iter(final_samples_list)
    

    def roulette_wheel_selection(self):
        rand_val = random.uniform(0, 1)
        cumulative_prob = 0.0

        for i, prob in enumerate(self.dataset_p):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                return i
