from torch.utils.data import ConcatDataset, DataLoader
from .data_loader import TrainSegLoader, TrainSampleLoader
from ._batch_scheduler import BatchSchedulerSampler
from ._read_data import data_info
import os


# from prefetch_generator import BackgroundGenerator
# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

# def PretrainDataProvider(root_path, datasets, batch_size, win_size, step, nums):
#     """
#     return DataLoader,

#     Args:
#         root_path (_type_): _description_
#         datasets (_type_): _description_
#         batch_size (_type_): _description_
#         win_size (_type_): _description_
#         step (_type_): _description_
#         nums (_type_): _description_
#     Returns:
#         concat_dataset, data_loader
#     """
#     datasets = datasets.split(",")
#     concat_dataset = []
#     for dataset in datasets:
#         print(f"loading {dataset}...", end="")
#         filenames, train_lens, file_nums, discrete_channels = data_info(root_path=root_path, dataset=dataset)
#         for i in range(file_nums):
#             data_path = os.path.join(root_path, filenames[i])
#             data_set = TrainSegLoader(data_path, train_lens[i], win_size, step, mode="pretrain", discrete_channels=discrete_channels)
#             concat_dataset.append(data_set)
#         print("done!")
#     concat_dataset = ConcatDataset(concat_dataset)
#     data_loader = DataLoaderX(
#         dataset=concat_dataset,
#         batch_size=batch_size,
#         num_workers=8,
#         drop_last=False,
#         sampler=BatchSchedulerSampler(dataset=concat_dataset, batch_size=batch_size, nums=nums),
#     )

#     return concat_dataset, data_loader


def DataProvider(root_path, datasets, batch_size, win_size, step, mode="train", percentage=0.1):
    if mode == "train":
        shuffle = True
    else: shuffle = False
    print(f"loading {datasets}({mode})...", end="")
    filenames, train_lens, file_nums, discrete_channels = data_info(root_path=root_path, dataset=datasets)
    assert file_nums == 1
    data_path = os.path.join(root_path, filenames[0])
    data_set = TrainSegLoader(data_path, train_lens[0], win_size, step, mode, percentage, discrete_channels, root_path=root_path)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
    print("done!")
    return data_set, data_loader


def AnormDataProvider(root_path, datasets, batch_size, win_size, step, nums=-1):
    anorm_dataset = TrainSampleLoader(root_path, datasets, win_size, step, type="Anorm", nums=nums)
    anorm_loader = DataLoaderX(
        dataset=anorm_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=True,
    )
    return anorm_dataset, anorm_loader


def PretrainDataProvider(root_path, datasets, batch_size, win_size, step, nums=-1):
    norm_dataset = TrainSampleLoader(root_path, datasets, win_size, step, type="Norm", nums=nums)
    anorm_dataset = TrainSampleLoader(root_path, datasets, win_size, step, type="Anorm", nums=nums)
    norm_loader = DataLoaderX(
        dataset=norm_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=True,
    )
    anorm_loader = DataLoaderX(
        dataset=anorm_dataset,
        batch_size=batch_size,
        num_workers=8,
        drop_last=False,
        shuffle=True,
    )
    return norm_loader, anorm_loader

if __name__ == '__main__':
    dataset, loader = DataProvider("/workspace/dataset/dataset", "MSL", 32, 100, 100)
    for i, (x, y, img) in enumerate(loader):
        print(img.shape)