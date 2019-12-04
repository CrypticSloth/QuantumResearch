import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def load_data(path, num_days = 30):
    '''
    load in all stock data from the path and return the close values
    and normalize the values to the reward
    '''

    paths = os.listdir(path)

    data = []
    names = []
    for p in paths:
        try:
            data.append(pd.read_csv(path + p).Close.values.tolist()[0:num_days])
            names.append(p[:-4])
        except:
            print("In file ", p , "the column 'Close' could not be found.")

    # Check that all data are the same size
    assert len(set([len(d) for d in data])) == 1, "Stock data has differing number of days recorded"

    return data, names


def wrangle_data(path, sample = None):
    '''
        Input the path to the directory that contains the

    '''
    os.chdir(path)

    if sample == 'train':
        r = []
        # for f in os.listdir():
        #     os.chdir(path + f)
        for d in os.listdir():
            if (d[-4:] == '.csv') and ('train' in d):
                # print(d)
                df = pd.read_csv(path + '/' +  d)
                r.append(df['rewards'])
    if sample == 'test':
        r = []
        roi = []
        for f in os.listdir():
            os.chdir(path + f)
            for d in os.listdir():
                if d[-4:] == '.csv' and 'test' in d:
                    # print(d)
                    df = pd.read_csv(path + f + '/' +  d)
                    r.append(df['returns'])
                    roi.append(df['roi'])

    df = pd.DataFrame(r).T
    return df

df = wrangle_data('D:/GitHub/QuantumResearch/NES_Meta_Trading/results/cavia/train/E=2000_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=5_WS=10', sample = 'train')

def sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

df["Mean"] = df.mean(axis=1)
df["STD"] = df.std(axis=1)

# df['mean_s'] = sigmoid(df['Mean'])
# df['std_s'] = sigmoid(df['STD'])

# df = df[:-1]

# %%
# Load in the stock data to simulate what would happen without trading actions

# data, names = load_data("D:/Github/CSCI380-CollabResearchCS/NES_Meta/dataset/test/",180)
# data = data[0][int(len(data[0])*.7):-1]
# len(data)
#
# data = (np.array(data) - data[0]) * 10 + 10000
# data
# df['market_value'] = data[:-1]

# %%

plt.plot(df.index, df.Mean) # For plotting training graphs
# plt.plot(df.market_value) # For plotting testing vs market value

plt.fill_between(df.index, df.Mean - df.STD, df.Mean + df.STD, color = (0.1,0.2,0.7,0.3))
# plt.show()
plt.title('CAVIA Train')
plt.savefig('D:/GitHub/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_10iters_CCData_train.png')

# %%
