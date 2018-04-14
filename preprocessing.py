import pandas as pd
import numpy as np
from config import *

class DataManager:

    def __init__(self, config):
        self.index = config['index']
        self.start = config['start']
        self.end = config['end']
        self.rng = config['rng']
        self.standard = config['standard']
        self.window = config['window']
        self.testing_ratio = config['testing_ratio']


    def _filter(self,data, rng, standard):

        # filter out continuous same price
        col_index = data.iloc[-rng:,:].pct_change().std() > standard
        return data.loc[:,col_index]


    def load_data(self, index, start, end):
        # read target data
        INDEX = pd.read_hdf('./data/'+index+'_index.h5')['close'][start:end]

        # read input data
        items = ['close','open','high','low','volume']
        data = {}
        for k, item in enumerate(items):
            temp = pd.read_hdf('./data/'+index+'.h5',item)[start:end].dropna(1) 
            data[item] = self._filter(temp, self.rng, self.standard)
            # intersect rows and columns
            if k == 0:
                col_index = set(data[item].columns)
                rol_index = set(data[item].index)
            else:
                col_index = col_index & set(data[item].columns)
                rol_index = rol_index & set(data[item].index)

        rol_index = rol_index & set(INDEX.index) # intersect with INDEX
        col_index = list(col_index)
        self.assets = col_index
        rol_index = list(rol_index)
        # extract intersections
        for item in items:
            data[item] = data[item].loc[rol_index, col_index].sort_index()
        INDEX = INDEX.loc[rol_index].sort_index()

        data = pd.Panel(data).values.transpose(1,2,0)
        INDEX = INDEX.values.reshape(-1,1)

        return data, INDEX

    def normalize(self,data):
        # convert to return and normalize
        if data.ndim == 3 :
            # shape (m, n_assets, n_featurs)
            rets = data[1:,:,:]/data[:-1,:,:] - 1
            mean = rets.mean(0)
            std = rets.std(0)
            rets = (rets-mean)/std
        else:
            rets = data[1:,:]/data[:-1,:] - 1
            mean = rets.mean(0)
            std = rets.std(0)
            rets = (rets-mean)/std

        return rets

    def _to_price_tensor(self, data, window):
        m, n_assets, n_features  = data.shape
        tensor = np.zeros((m-window+1,n_assets, window, n_features))
        for i in range(window, m+1):
            # (n_assets, window, n_features)
            tensor[i-window] = data[i-window:i].transpose(1,0,2)
        return tensor


    def get_data(self):
        data, I = self.load_data(self.index, start=self.start, end=self.end)
        S = self._to_price_tensor(data, self.window)
        y = data[self.window:,:,0]/data[self.window-1:-1,:,0] # close price
        I = I[self.window:]/I[self.window-1:-1]
        I = I.reshape(-1,)
        # split training and testing data
        truncate = int(S.shape[0]*(1-self.testing_ratio))
        S_train = S[:truncate]
        y_train = y[:truncate]
        I_train = I[:truncate]

        S_test = S[truncate:]
        y_test = y[truncate:]
        I_test = I[truncate:]

        print('loading data complete')
        print('='*30)
        print('number of training data:', S_train.shape[0])
        print('number of testing data:', S_test.shape[0])
        print('number of assets:', S.shape[1])
        print('number of window:', S.shape[2])
        print('number of features:', S.shape[3])
        print('='*30)

        return S_train, y_train, I_train, S_test, y_test, I_test 
