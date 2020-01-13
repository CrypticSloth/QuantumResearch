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
            print("Either the file ", p , " could not be read or the column 'Close' could not be found.")

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
        # for f in os.listdir():
        #     os.chdir(path + f)
        for d in os.listdir():
            if d[-4:] == '.csv' and 'test' in d:
                # print(d)
                df = pd.read_csv(path +  '/' +  d)
                r.append(df['returns'])
                roi.append(df['roi'])

    df = pd.DataFrame(r).T
    return df

df = wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=5_WS=10_ND=360', sample = 'test')

def sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1


df["Mean"] = df.mean(axis=1)
df["STD"] = df.std(axis=1)





# %%
# For plotting training graphs

df = df[15:] # remove some rows if needed

plt.plot(df.index, df.Mean)
plt.fill_between(df.index, df.Mean - df.STD, df.Mean + df.STD, color = (0.1,0.2,0.7,0.3))
plt.ylim(-3,3)
plt.xlabel('Epochs')
plt.ylabel('Rewards')
plt.title('MAML Training on CC Stocks')
plt.savefig('C:/Github/QuantumResearch/NES_Meta_Trading/graphicsNEWTRAIN_10000Epochs_9iters_defaultparams_training_MAML_2.png')
# plt.show()

# %%

# For plotting market tests

# TODO: Do this with portfolio data...
# Load in the stock data to simulate what would happen without trading actions
data, names = load_data("C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/",360)
data = [d[int(len(d)*.7):-1] for d in data]
len(data[0])

limit = 5
starting_money = 10000
data = [(np.array(d) - d[0]) * limit for d in data]
data = [sum(x) + starting_money for x in zip(*data)]
data
df['market_value'] = data[:-1]

#%%
plt.plot(df.index, df.Mean, label='Mean Balance')
plt.plot(df.market_value, label='Market Value', color='green')
plt.fill_between(df.index, df.Mean - df.STD, df.Mean + df.STD, color = (0.1,0.2,0.7,0.3))
plt.legend(loc='upper left')
plt.xlabel('Timestep')
plt.ylabel('Total Value ($)')
plt.title('MAML Market Test on Portfolio')
plt.tight_layout()
plt.savefig('C:/Github/QuantumResearch/NES_Meta_Trading/graphics/10000Epochs_9iters_defaultparams_marketTest_classic_MAML.png')
# plt.show()

# %%
