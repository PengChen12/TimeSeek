

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange','solar','pems03','pems04','pems07','pems08','nn5'
         ,'london_smart_meters','weather_dataset','kdd_cup', 'synthetic', 'zafnoo', 'czelan'
        ]

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    if params.dset == 'ettm1':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        if params.img :
            dls = DataLoaders(
                    datasetCls=Dataset_ETT_minute_image,
                    dataset_kwargs={
                    'root_path': root_path,
                    'data_path': 'ETTm1.csv',
                    'features': params.features,
                    'scale': True,
                    'size': size,
                    'use_time_features': params.use_time_features,
                    "half":params.is_half,
                    'all':params.is_all
                    },
                    batch_size=params.batch_size,
                    workers=params.num_workers,
                    )
        else:
            dls = DataLoaders(
                    datasetCls=Dataset_ETT_minute,
                    dataset_kwargs={
                    'root_path': root_path,
                    'data_path': 'ETTm1.csv',
                    'features': params.features,
                    'scale': True,
                    'size': size,
                    'use_time_features': params.use_time_features,
                    "half":params.is_half,
                    'all':params.is_all
                    },
                    batch_size=params.batch_size,
                    workers=params.num_workers,
                    )


    elif params.dset == 'ettm2':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        if params.img :
            dls = DataLoaders(
                    datasetCls=Dataset_ETT_minute_image,
                    dataset_kwargs={
                    'root_path': root_path,
                    'data_path': 'ETTm2.csv',
                    'features': params.features,
                    'scale': True,
                    'size': size,
                    'use_time_features': params.use_time_features,
                    "half":params.is_half,
                    'all':params.is_all
                    },
                    batch_size=params.batch_size,
                    workers=params.num_workers,
                    )
        else:
            dls = DataLoaders(
                    datasetCls=Dataset_ETT_minute,
                    dataset_kwargs={
                    'root_path': root_path,
                    'data_path': 'ETTm2.csv',
                    'features': params.features,
                    'scale': True,
                    'size': size,
                    'use_time_features': params.use_time_features,
                    "half":params.is_half,
                    'all':params.is_all
                    },
                    batch_size=params.batch_size,
                    workers=params.num_workers,
                    )

    elif params.dset == 'etth1':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points] # [seq_len, label_len, pred_len]
        if params.img :
            dls = DataLoaders(
                    datasetCls=Dataset_ETT_hour_image,
                    dataset_kwargs={
                    'root_path': root_path,
                    'data_path': 'ETTh1.csv',
                    'features': params.features,
                    'scale': True,
                    'size': size,
                    'use_time_features': params.use_time_features,
                    "half":params.is_half,
                    'all':params.is_all
                    },
                    batch_size=params.batch_size,
                    workers=params.num_workers,
                    )
        else:
            dls = DataLoaders(
                    datasetCls=Dataset_ETT_hour,
                    dataset_kwargs={
                    'root_path': root_path,
                    'data_path': 'ETTh1.csv',
                    'features': params.features,
                    'scale': True,
                    'size': size,
                    'use_time_features': params.use_time_features,
                    "half":params.is_half,
                    'all':params.is_all
                    },
                    batch_size=params.batch_size,
                    workers=params.num_workers,
                    )



    elif params.dset == 'etth2':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        if params.img :
            dls = DataLoaders(
                    datasetCls=Dataset_Custom,
                    dataset_kwargs={
                    'root_path': root_path,
                    'data_path': 'ETTh2.csv',
                    'features': params.features,
                    'scale': True,
                    'size': size,
                    'use_time_features': params.use_time_features,
                    "half":params.is_half,
                    'all':params.is_all
                    },
                    batch_size=params.batch_size,
                    workers=params.num_workers,
                    )
        else:
            dls = DataLoaders(
                    datasetCls=Dataset_ETT_hour,
                    dataset_kwargs={
                    'root_path': root_path,
                    'data_path': 'ETTh2.csv',
                    'features': params.features,
                    'scale': True,
                    'size': size,
                    'use_time_features': params.use_time_features,
                    "half":params.is_half,
                    'all':params.is_all
                    },
                    batch_size=params.batch_size,
                    workers=params.num_workers,
                    )
    

    elif params.dset == 'electricity':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'electricity.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'traffic':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'traffic.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    
    elif params.dset == 'weather':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        if params.img:
            dls = DataLoaders(
                    datasetCls=Dataset_Custom_image,
                    dataset_kwargs={
                    'root_path': root_path,
                    'data_path': 'weather.csv',
                    'features': params.features,
                    'scale': True,
                    'size': size,
                    'use_time_features': params.use_time_features,
                    "half":params.is_half
                    },
                    batch_size=params.batch_size,
                    workers=params.num_workers,
                    )
        else:
            dls = DataLoaders(
                    datasetCls=Dataset_Custom,
                    dataset_kwargs={
                    'root_path': root_path,
                    'data_path': 'weather.csv',
                    'features': params.features,
                    'scale': True,
                    'size': size,
                    'use_time_features': params.use_time_features,
                    "half":params.is_half
                    },
                    batch_size=params.batch_size,
                    workers=params.num_workers,
                    )

    elif params.dset == 'illness':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'illness.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )

    elif params.dset == 'exchange':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'exchange.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'solar':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'solar.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'pems03':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'pems04':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'pems04.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'pems07':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'pems08':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'pems08.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'nn5':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'pems08.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'london_smart_meters':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'london_smart_meters.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
        
    elif params.dset == 'weather_dataset':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'weather_dataset.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'kdd_cup':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'kdd_cup.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'synthetic':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'series_produced.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'czelan':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'CzeLan.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
    elif params.dset == 'zafnoo':
        root_path = 'data/predict_datasets/'
        size = [params.context_points, 0, params.target_points]
        dls = DataLoaders(
                datasetCls=Dataset_Custom,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': 'ZafNoo.csv',
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features,
                "half":params.is_half
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
           

    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
