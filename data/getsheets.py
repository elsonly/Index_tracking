import pandas as pd
import numpy as np


INDEX = ['tw50']#'XIN9',
items = ['open','high','low','close','volume']
for index in INDEX:
    data = {}
    for item in items:
        temp = pd.read_csv(index+'_'+item+'.csv')
        temp = temp.set_index('Dates')
        temp.index = pd.to_datetime(temp.index)
        data[item] = temp


    cols = list(data[item].columns)

    for col in cols:
        for i, item in enumerate(items):
            temp_item = data[item][col]
            temp_item.name = item
            try:
                temp_item = temp_item.astype(np.float32)
            except:
                if temp_item[0]=='#N/A Requesting Data...':
                    temp_item = temp_item[1:]
                    temp_item = temp_item.astype(np.float32)

            if i == 0:
                temp = temp_item
            else:
                temp = pd.concat((temp,temp_item), axis=1)

        temp.to_hdf(index+'.h5',col)

###########

INDEX = ['XIN9','sp500','tw50']#,
items = ['open','high','low','close','volume']
for index in INDEX:
    data = {}
    for item in items:
        temp = pd.read_csv(index+'_'+item+'.csv')
        temp = temp.set_index('Dates')
        temp.index = pd.to_datetime(temp.index)
        try:
            temp = temp.astype(np.float32)
        except:
            temp = temp[1:]
            temp = temp.astype(np.float32)
        temp.to_hdf(index+'.h5',item)



    cols = list(data[item].columns)

    for col in cols:
        for i, item in enumerate(items):
            temp_item = data[item][col]
            temp_item.name = item
            try:
                temp_item = temp_item.astype(np.float32)
            except:
                if temp_item[0]=='#N/A Requesting Data...':
                    temp_item = temp_item[1:]
                    temp_item = temp_item.astype(np.float32)

            if i == 0:
                temp = temp_item
            else:
                temp = pd.concat((temp,temp_item), axis=1)

        temp.to_hdf(index+'.h5',col)


###########

index = 'tw50'
data = data.set_index('Dates')
data.index = pd.to_datetime(data.index)
data = data.astype(np.float32)
data.to_hdf(index+'_index.h5',index+'_index')

