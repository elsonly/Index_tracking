import pandas as pd
import numpy as np

class DataManager:

    def __init__(self, config):
        self.index = config['index']
        self.start = config['start']
        self.end = config['end']
        self.standard = config['standard']
        self.window = config['window']
        self.validation_ratio = config['validation_ratio']
        self.holding_period = config['holding_period']
        #self.testing_period = config['testing_period']


    def _filter(self,data, standard):

        # filter out continuous same price
        rng_1 = 300
        rng_2 = 200
        rng_3 = 100
        #col_index = data.iloc[-rng:,:].pct_change().std() > standard
        cond_1 = data.iloc[-rng_1:,:].std() > standard
        cond_2 = data.iloc[-rng_2:,:].std() > standard
        cond_3 = data.iloc[-rng_3:,:].std() > standard
        col_index = cond_1&cond_2&cond_3

        return data.loc[:,col_index]


    def load_data(self):
    
        # read target data
        INDEX = pd.read_hdf('./data/'+self.index+'_index.h5')['close'][self.start:self.end].pct_change().dropna()
        # read input data
        items = ['close','high','low']#,'volume','open']
        data = {}
        for k, item in enumerate(items):
            #temp = pd.read_hdf('./data/'+self.index+'.h5',item)[self.start:self.end]\
            #                                            .pct_change().dropna(how='all',axis=0).dropna(axis=1)

            temp = pd.read_hdf('./data/tw.h5',item)[self.start:self.end].dropna(how='all',axis=0).fillna(method='ffill')\
                                                        .pct_change().dropna(how='all',axis=0).dropna(axis=1)

            
            none_zero_index = (temp==0).sum(axis=1) != len(temp.columns)
            temp = temp.loc[none_zero_index,:] # drop market close data

            data[item] = self._filter(temp, self.standard)
            # intersect rows and columns
            if k == 0:
                col_index = set(data[item].columns)
                row_index = set(data[item].index)
            else:
                col_index = col_index & set(data[item].columns)
                row_index = row_index & set(data[item].index)

        row_index = row_index & set(INDEX.index) # intersect with INDEX
        col_index = list(col_index)
        self.assets = col_index
        row_index = list(row_index)
        # extract intersections
        for item in items:
            data[item] = data[item].loc[row_index, col_index].sort_index()
        INDEX = INDEX.loc[row_index].sort_index()

        data = pd.Panel(data).values.transpose(1,2,0)
        INDEX = INDEX.values.reshape(-1,1)

        return data, INDEX


    def _to_price_tensor(self, data, window):
        if len(data.shape)==3: #S
            m, n_assets, n_features  = data.shape
            tensor = np.zeros((m-window+1, n_assets, window, n_features))
            for i in range(window, m+1):
                # (n_assets, window, n_features)
                tensor[i-window] = data[i-window:i].transpose(1,0,2)

        elif len(data.shape)==2:
            m, n_assets = data.shape

            if n_assets == 1: # I
                tensor = np.zeros((m-window+1, window))
                for i in range(window, m+1):
                    tensor[i-window] = data[i-window:i].reshape(-1,)
            else: # y
                tensor = np.zeros((m-window+1, n_assets, window))
                for i in range(window, m+1):
                    
                    tensor[i-window] = data[i-window:i].transpose(1,0)

        return tensor


    def get_data(self):
        print('loading data ...')
        data, I = self.load_data()

        data = data * 10 # returns * 10
        y = data[:,:,0] #close price
        S = self._to_price_tensor(data, self.window) # (m,n_assets,window,n_feature)
        I = I * 10 # returns * 10
        
        I = self._to_price_tensor(I, self.holding_period) # (m, holding_period)
        y = self._to_price_tensor(y, self.holding_period) # (m, n_assets, holding_period)


        # adjust time
        y = y[-S.shape[0]:] # set to equal lenth 
        I = I[-S.shape[0]:] 

        S = S[:-self.holding_period]
        y = y[self.holding_period:]
        I = I[self.holding_period:]

        # split training and validation data
        truncate = int(S.shape[0]*(1-self.validation_ratio))
        S_train = S[:truncate]
        y_train = y[:truncate]
        I_train = I[:truncate]

        S_val = S[truncate:]
        y_val = y[truncate:]
        I_val = I[truncate:]

        print('loading data complete')
        print('='*30)
        print('number of training data:', S_train.shape[0])
        print('number of validation data:', S_val.shape[0])
        print('number of assets:', S.shape[1])
        print('number of window:', S.shape[2])
        print('number of holding_period:', self.holding_period)
        print('number of features:', S.shape[3])
        print('='*30)

        return S_train, y_train, I_train, S_val, y_val, I_val

    def get_data_testing(self):
        start_test = pd.to_datetime(self.end) + pd.offsets.Day()
        INDEX = pd.read_hdf('./data/'+self.index+'_index.h5')['close'].pct_change().dropna()
        INDEX = INDEX.loc[INDEX!=0]

        # read input data
        items = ['close','high','low']#,'volume','open']
        S = {}
        for k,item in enumerate(items):

            temp = pd.read_hdf('./data/tw.h5',item)[self.assets].fillna(method='ffill').pct_change().dropna(axis=0)

            none_zero_index = (temp==0).sum(axis=1) != len(temp.columns)
            temp = temp.loc[none_zero_index,:].sort_index() # drop market close data

            S[item] = temp

        row_index = set(temp.index) & set(INDEX.index) # intersect with INDEX
        row_index = list(row_index)

        # extract intersections
        INDEX = INDEX.loc[row_index].sort_index()
        INDEX = INDEX.loc[start_test:]

        for item in items:
            S[item] = S[item].loc[row_index].sort_index()
            m = len(S[item].loc[:start_test])
            S[item] = S[item].iloc[m-self.window+1 : ]
        
        y = S['close'].loc[INDEX.index]


        #print('same time index:', (y.index==INDEX.index).all())

        S = pd.Panel(S).values.transpose(1,2,0) * 10 # return * 10
        S = self._to_price_tensor(S, self.window) # (m,n_assets,window,n_feature)
        INDEX = INDEX.values.reshape(-1,1)
        y = y.values
        
        return S, y, INDEX
        
        
        