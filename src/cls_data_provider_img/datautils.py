import os
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler
from PIL import Image
from torchvision import transforms

def load_UCR(dataset, data_path):
    root_path = os.path.join(data_path + 'UCR', dataset)
    train_file = os.path.join(data_path + 'UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join(data_path + 'UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_img, train_len = get_img(root_path, 'train')
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_img, _ = get_img(root_path, 'test', train_len)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, train_img, test[..., np.newaxis], test_labels, test_img
    
    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, train_img, test[..., np.newaxis], test_labels, test_img


def load_UEA(dataset, data_path):
    train_data = loadarff(data_path + f'UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(data_path + f'UEA/{dataset}/{dataset}_TEST.arff')[0]
    
    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([ d.tolist() for d in t_data ])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    
    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)
    
    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)
    
    labels = np.unique(train_y)
    transform = { k : i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return set_nan_to_zero(train_X), train_y, set_nan_to_zero(test_X), test_y

def get_img(root_path, flag, num=0):
    root_path = os.path.join(root_path, flag)
    img_list = []
    count_dir = len([d for d in os.listdir(root_path) if os.path.join(root_path, d)])
    for id in range(count_dir):
        id += num
        id = str(id)
        img_path = root_path + '/' + id
        img_path = os.path.join(img_path, '{}.png'.format(id))
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_transform = transforms.Compose([transforms.ToTensor()])
        img = img_transform(img)    # C x H x W
        img_list.append(img)

    return img_list, count_dir

def _get_time_features(dt):
    return np.stack([
        dt.minute.to_numpy(),
        dt.hour.to_numpy(),
        dt.dayofweek.to_numpy(),
        dt.day.to_numpy(),
        dt.dayofyear.to_numpy(),
        dt.month.to_numpy(),
        dt.weekofyear.to_numpy(),
    ], axis=1).astype(np.float)

def set_nan_to_zero(a):
    where_are_NaNs = np.isnan(a)
    a[where_are_NaNs] = 0
    return a

def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)
