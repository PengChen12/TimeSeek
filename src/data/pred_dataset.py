import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from PIL import Image
from torchvision import transforms

from src.data.timefeatures import time_features
import warnings
import random
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_time_features=False,half=0,all=0,one_channel=0
                 ):
        
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.half =half
        self.all = all
        self.one_channel=one_channel
        self.split = split
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0 * 30 * 24 ,  12 * 30 * 24 - self.seq_len,  12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24,  12 * 30 * 24 + 4 * 30 * 24,  12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # if self.half!=0 and border1 == 0:
        #     border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))
        if self.all :
            border1 = 0
            border2 = len(df_raw)
        print(border1, border2)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.number_of_channels = data.shape[1]

    def __getitem__(self, index):
        s_begin = index
        if self.split == 'train':
            s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # if self.split == 'train':
        #     seq_x, seq_y = channel_mixing(seq_x, seq_y)

        # if self.scale:
        #     self.scaler.fit(seq_x)
        #     seq_x = self.scaler.transform(seq_x)
        #     seq_y = self.scaler.transform(seq_y)
        #     seq_x, seq_y = filter_data(seq_x, seq_y)
        if self.split == 'train':
            seq_x, seq_y = channel_mixing(seq_x, seq_y)
        if self.one_channel:
            seq_x, seq_y=self.channel_chosing(seq_x, seq_y, 7)
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.split == 'train':
            return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y

class Dataset_ETT_hour_image(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_time_features=False,half=0,all=0,one_channel=0
                 ):
        
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.half =half
        self.all = all
        self.one_channel=one_channel
        self.split = split
        self.mix = 0
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0 * 30 * 24 ,  12 * 30 * 24 - self.seq_len,  12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24,  12 * 30 * 24 + 4 * 30 * 24,  12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.half!=0 and border1 == 0:
            border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))
        if self.all :
            border1 = 0
            border2 = len(df_raw)
        print(border1, border2)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.number_of_channels = data.shape[1]

        # read image
        data_name = extract_filename(self.data_path)
        if self.mix:
            self.img = self.__getimg__(data_name, data_name, image_type='cat')
        else:
            img = []
            for i in range(self.number_of_channels):
                data_name_i = data_name  + f'_{i}'
                img_i = self.__getimg__(data_name, data_name_i, image_type='divide')
                img.append(img_i)
            self.img = torch.stack(img, dim=0)
        
        
    def __getimg__(self, data, data_name, image_type):

        if image_type == 'cat':
            img_path = os.path.join(self.root_path , 'image_data', data, 'process_data_cat', data_name + '.png')
        elif image_type == 'divide':
            img_path = os.path.join(self.root_path , 'image_data', data, 'process_data', data_name + '.png')

        img = Image.open(img_path)
        img = img.convert('RGB')
        img_transform = transforms.Compose([transforms.ToTensor()])
        img = img_transform(img)    # C x H x W
        return img



    def __getitem__(self, index):
        s_begin = index
        # if self.split == 'train':
        #     s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x = _torch(seq_x)[0]
        seq_y = _torch(seq_y)[0]
        if self.one_channel:
            seq_x, seq_y=self.channel_chosing(seq_x, seq_y, 7)
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]


        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return seq_x, seq_y , self.img

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        # if self.split == 'train':
        #     return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        # else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y



class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 use_time_features=False,half=0,all=0,one_channel=0
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.half = half
        self.all = all
        self.one_channel=one_channel
        self.split =  split
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0 * 30 * 24 * 4, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # if self.half!=0 and border1 == 0:
        #     border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))
        if self.all :
            border1 = 0
            border2 = len(df_raw)
            print(border1,border2)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        if self.split == 'train':
            s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        if self.split == 'train':
            seq_x, seq_y = channel_mixing(seq_x, seq_y)
        if self.one_channel:
            seq_x,seq_y=self.channel_chosing(seq_x, seq_y, 1)

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.split == 'train':
            return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y

class Dataset_ETT_minute_image(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 use_time_features=False,half=0,all=0,one_channel=0
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.use_time_features = use_time_features

        self.root_path = root_path
        self.data_path = data_path
        self.half = half
        self.all = all
        self.one_channel=one_channel
        self.split =  split
        self.mix = 0
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0 * 30 * 24 * 4, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        # if self.half!=0 and border1 == 0:
        #     border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))
        if self.all :
            border1 = 0
            border2 = len(df_raw)
            print(border1,border2)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.number_of_channels = data.shape[1]

        data_name = extract_filename(self.data_path)
        if self.mix:
            self.img = self.__getimg__(data_name, data_name, image_type='cat')
        else:
            img = []
            for i in range(self.number_of_channels):
                data_name_i = data_name  + f'_{i}'
                img_i = self.__getimg__(data_name, data_name_i, image_type='divide')
                img.append(img_i)
            self.img = torch.stack(img, dim=0)
        
        
    def __getimg__(self, data, data_name, image_type):

        if image_type == 'cat':
            img_path = os.path.join(self.root_path , 'image_data', data, 'process_data_cat', data_name + '.png')
        elif image_type == 'divide':
            img_path = os.path.join(self.root_path , 'image_data', data, 'process_data', data_name + '.png')

        img = Image.open(img_path)
        img = img.convert('RGB')
        img_transform = transforms.Compose([transforms.ToTensor()])
        img = img_transform(img)    # C x H x W
        return img

    def __getitem__(self, index):
        s_begin = index
        if self.split == 'train':
            s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        if self.one_channel:
                seq_x,seq_y=self.channel_chosing(seq_x, seq_y, 1)
        seq_x = _torch(seq_x)[0]
        seq_y = _torch(seq_y)[0]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return seq_x, seq_y, self.img

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.split == 'train':
            return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y


class Dataset_Custom(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.7, test_split=0.2, half =0,all=0,one_channel=0
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.split =  split

        self.all = all
        # train test ratio
        if self.all:
            self.train_split, self.test_split = 1, 0
        else:
            self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.half = half
        self.all = all
        self.one_channel=one_channel
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: [time_col_name, ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.half!=0 and border1 == 0:
            border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))
          
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.number_of_channels = data.shape[1]

    def __getitem__(self, index):
        s_begin = index

        # if self.split == 'train':
        #     s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # if self.split == 'train':
        #     seq_x, seq_y = channel_mixing(seq_x, seq_y)
        if self.one_channel:
            seq_x,seq_y=self.channel_chosing(seq_x, seq_y, 1)

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        # if self.split == 'train':
        #     return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        # else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y
        

class Dataset_Custom_SampleScale(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.7, test_split=0.2, half =0,all=0,one_channel=0
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.split =  split

        self.all = all
        # train test ratio
        if self.all:
            self.train_split, self.test_split = 1, 0
        else:
            self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.half = half
        self.all = all
        self.one_channel=one_channel
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: [time_col_name, ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.half!=0 and border1 == 0:
            border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))
          
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # if self.scale:
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     data = self.scaler.transform(df_data.values)
        # else:
        data = df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.number_of_channels = data.shape[1]

    def __getitem__(self, index):
        s_begin = index

        # if self.split == 'train':
        #     s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        if self.scale:
            self.scaler.fit(seq_x)
            seq_x = self.scaler.transform(seq_x)
            seq_y = self.scaler.transform(seq_y)

        seq_x, seq_y = filter_data(seq_x, seq_y)

        if self.one_channel:
            seq_x,seq_y=self.channel_chosing(seq_x, seq_y, 1)

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)


    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        # if self.split == 'train':
        #     return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        # else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y
    


class Dataset_Custom_image(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.7, test_split=0.2, half =0,all=0,one_channel=0
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.split =  split

        self.all = all
        # train test ratio
        if self.all:
            self.train_split, self.test_split = 1, 0
        else:
            self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.half = half
        self.all = all
        self.one_channel=one_channel
        self.mix = 0

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: [time_col_name, ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.half!=0 and border1 == 0:
            border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))
          
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # data = df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.number_of_channels = data.shape[1]

        # read image
        data_name = extract_filename(self.data_path)
        if self.mix:
            self.img = self.__getimg__(data_name, data_name, image_type='cat')
        else:
            img = []
            for i in range(self.number_of_channels):
                data_name_i = data_name  + f'_{i}'
                img_i = self.__getimg__(data_name, data_name_i, image_type='divide')
                img.append(img_i)
            self.img = torch.stack(img, dim=0)
        
        
    def __getimg__(self, data, data_name, image_type):

        if image_type == 'cat':
            img_path = os.path.join(self.root_path , 'image_data', data, 'process_data_cat', data_name + '.png')
        elif image_type == 'divide':
            img_path = os.path.join(self.root_path , 'image_data', data, 'process_data', data_name + '.png')

        img = Image.open(img_path)
        img = img.convert('RGB')
        img_transform = transforms.Compose([transforms.ToTensor()])
        img = img_transform(img)    # C x H x W
        return img




    def __getitem__(self, index):
        s_begin = index

        # if self.split == 'train':
        #     s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # if self.scale:
        #     self.scaler.fit(seq_x)
        #     seq_x = self.scaler.transform(seq_x)
        #     seq_y = self.scaler.transform(seq_y)
        #     seq_x, seq_y = filter_data(seq_x, seq_y)
        seq_x = _torch(seq_x)[0]
        seq_y = _torch(seq_y)[0]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return seq_x, seq_y, self.img

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        # if self.split == 'train':
        #     return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        # else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y
    

def _torch(*dfs):
    return tuple(torch.from_numpy(x).float() for x in dfs)

def extract_filename(file_path):
        # 去掉文件扩展名
        filename = file_path.split('.')[0]
        return filename


def filter_data(seq_x, seq_y):
    """
    Filter function to set seq_x and seq_y to zero arrays 
    if any absolute value in seq_x or seq_y is greater than 9.
    """
    if np.any(np.abs(seq_x) > 9) or np.any(np.abs(seq_y) > 9):
        return np.zeros_like(seq_x), np.zeros_like(seq_y)
    return seq_x, seq_y


def channel_mixing(seq_x, seq_y):
    T, C = seq_x.shape
    L, _ = seq_y.shape
    
    # 1. 通道洗牌
    shuffle_indices = np.random.permutation(C)
    
    # 2. 生成随机值
    random_values = np.random.normal(0, 1, C)
    
    # 3. 通道调整
    shuffled_seq_x = seq_x[:, shuffle_indices]
    shuffled_seq_y = seq_y[:, shuffle_indices]
    
    # 4. 乘以随机值
    mixed_seq_x = shuffled_seq_x * random_values
    mixed_seq_y = shuffled_seq_y * random_values
    
    # 5. 加上原始序列
    mixed_seq_x += seq_x
    mixed_seq_y += seq_y
    
    return mixed_seq_x, mixed_seq_y


class Dataset_Exo(Dataset):
    def __init__(self, root_path, split='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 time_col_name='date', use_time_features=False, 
                 train_split=0.7, test_split=0.2, half =0,all=0,one_channel=0
                 ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_col_name = time_col_name
        self.use_time_features = use_time_features
        self.split =  split

        self.all = all
        # train test ratio
        if self.all:
            self.train_split, self.test_split = 1, 0
        else:
            self.train_split, self.test_split = train_split, test_split

        self.root_path = root_path
        self.data_path = data_path
        self.half = half
        self.all = all
        self.one_channel=one_channel
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        '''
        df_raw.columns: [time_col_name, ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        #cols.remove(self.target) if self.target
        #cols.remove(self.time_col_name)
        #df_raw = df_raw[[self.time_col_name] + cols + [self.target]]
        
        num_train = int(len(df_raw) * self.train_split)
        num_test = int(len(df_raw) * self.test_split)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.half!=0 and border1 == 0:
            border1 = int((border2-self.seq_len-self.pred_len)*(1-self.half))
          
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[[self.time_col_name]][border1:border2]
        df_stamp[self.time_col_name] = pd.to_datetime(df_stamp[self.time_col_name])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp[self.time_col_name].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp[self.time_col_name].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp[self.time_col_name].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp[self.time_col_name].apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop([self.time_col_name], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp[self.time_col_name].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.number_of_channels = data.shape[1]

    def __getitem__(self, index):
        s_begin = index

        # if self.split == 'train':
        #     s_begin = int(s_begin * (1//self.half))
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # if self.split == 'train':
        #     seq_x, seq_y = channel_mixing(seq_x, seq_y)
        if self.one_channel:
            seq_x,seq_y=self.channel_chosing(seq_x, seq_y, 1)

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        # if self.split == 'train':
        #     return int((len(self.data_x) - self.seq_len - self.pred_len + 1)*self.half)
        # else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

    
    def channel_chosing(self,seq_x ,seq_y, num_channel):
        T, C = seq_x.shape
        if C > num_channel:
            numbers = list(range(0,C))
            selected_numbers = random.sample(numbers, num_channel)
            return seq_x[:,selected_numbers],seq_y[:,selected_numbers]
        else:
            return seq_x,seq_y
       