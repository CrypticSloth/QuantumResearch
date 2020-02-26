import numpy as np
import pandas as pd
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
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

df = pd.DataFrame({'a': [[[1,2,3]],[[1,2,3]],[[1,2,3]]], 'b':  [[[1,2,3]],[[1,2,3]],[[1,2,3]]]})
df

def wrangle_data(path, sample):
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

    if sample == 'portfolio':
        vals = defaultdict(list)
        for d in os.listdir():
            if d[-4:] == '.csv' and 'portfolio' in d:
                df = pd.read_csv(path +  '/' +  d)
                for i in range(len(df.columns)):
                    vals[i].append(list(df[df.columns[i]]))
        return vals

    if sample == 'train' or 'test':
        df = pd.DataFrame(r).T

    return df

df = pd.DataFrame({'a':[1,2,3]})
list(df['a'])

def training_plot(trial_path, num_remove, plot_title, plot_save_loc):
    df = wrangle_data(trial_path, sample = 'train')

    def sigmoid(x):
        return (2 / (1 + np.exp(-x))) - 1


    df["Mean"] = df.mean(axis=1)
    df["STD"] = df.std(axis=1)
    len(df)
    df

    df = df[num_remove:] # remove some rows if needed

    plt.plot(df.index, df.Mean)
    plt.fill_between(df.index, df.Mean - df.STD, df.Mean + df.STD, color = (0.1,0.2,0.7,0.3))
    # plt.ylim(-3,3)
    # plt.xlim(0,500)
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    plt.title(plot_title)
    plt.savefig(plot_save_loc)
    plt.show()

def market_test_plot(trial_path, data_path, num_days, plot_title, plot_save_loc, legend_loc, trading_limit, starting_money=10000, offset=-1):
    df = wrangle_data(trial_path, sample = 'test')

    def sigmoid(x):
        return (2 / (1 + np.exp(-x))) - 1


    df["Mean"] = df.mean(axis=1)
    df["STD"] = df.std(axis=1)

    print(len(df))

    data, names = load_data(data_path,num_days)
    data = [d[int(len(d)*.7):-1] for d in data]

    limit = trading_limit
    starting_money = starting_money
    data = [(np.array(d) - d[0]) * limit for d in data]
    data = [sum(x) + starting_money for x in zip(*data)]

    # print(data)
    print(len(data))

    df['market_value'] = data[:offset]
    # print(df)

    plt.plot(df.index, df.Mean, label='Mean Balance')
    plt.plot(df.market_value, label='Market Value', color='green')
    plt.fill_between(df.index, df.Mean - df.STD, df.Mean + df.STD, color = (0.1,0.2,0.7,0.3))
    plt.legend(loc=legend_loc)
    plt.xlabel('Timestep')
    plt.ylabel('Total Value ($)')
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(plot_save_loc)


def market_test_detail(trial_path, data_path, plot_title, plot_save_loc, num_days, trading_limit, starting_money=10000):
    # trial_path = 'C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360'
    df = wrangle_data(trial_path, sample = 'portfolio')

    means = []
    stds = []
    for i in range(1,len(df)):
        means.append(np.mean(np.array(df[i]), axis=0))
        stds.append(np.std(np.array(df[i]), axis=0))
    # means
    # stds
    # data_path = 'C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/'
    # num_days = 360
    data, names = load_data(data_path,num_days)
    data = [d[int(len(d)*.7):-1] for d in data]
    trading_limit = 10
    starting_money = 10000
    limit = trading_limit
    starting_money = starting_money
    data = [((np.array(d) - d[0]) * limit) + starting_money for d in data]

    tableau10 = [(31, 119, 180), (255, 127, 14),
             (44, 160, 44), (214, 39, 40),
             (148, 103, 189),  (140, 86, 75),
             (227, 119, 194),  (127, 127, 127),
             (188, 189, 34),  (23, 190, 207)]

    for i in range(len(tableau10)):
        r, g, b = tableau10[i]
        tableau10[i] = (r / 255., g / 255., b / 255.)

    # data
    # means

    means = np.around(means, 2)
    means = means * 100
    # sanity check
    [x.sum() for x in np.array(means).T]

    # names
    ind = np.arange(len(means[0]))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20,10))
    # %%
    for i in range(len(means)):
        if i == 0:
            # plt.plot(range(len(means[i])), means[i], label='Cash')
            ax1.bar(ind, means[i], label='Cash', color=tableau10[i], width=0.85)
        else:
            # plt.plot(range(len(means[i])), means[i], label=names[i-1])
            ax1.bar(ind, means[i], bottom=means[i-1], label=names[i-1], color=tableau10[i], width=0.85)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Total Percent Held of Portfolio')
    # ax1.tight_layout()
    # plt.plot()
    # plt.savefig(plot_save_loc)
    # %%
    # plt.figure(2)
    for i in range(len(data)):
        ax2.plot(range(len(data[i])),data[i],label=names[i],color=tableau10[i+1])

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Total Value ($)')
    # ax2.tight_layout()
    # ax2.plot()
    fig.suptitle(plot_title)
    fig.savefig(plot_save_loc, dpi=500)

    # %%
########
# MAML #
########
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/train/E=2500_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    num_remove=0,
    plot_title='MAML Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML/MAML_360-20Epochs_12iters_CCData_Limit10_training.png'
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    num_remove=0,
    plot_title='MAML Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML/MAML_360-20Epochs_12iters_CCData_Limit10_testTrain.png'
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/',
    num_days=360,
    plot_title='MAML Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML/MAML_360-20Epochs_12iters_CCData_Limit10_marketTest.png',
    legend_loc='upper left',
    trading_limit=10
)
market_test_detail(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML/test.png',
    plot_title='MAML Market Test Detail on Test Portfolio',
    num_days=360,
    trading_limit=10
)


#########
# CAVIA #
#########
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/train/E=2500_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    num_remove=0,
    plot_title='CAVIA Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA/CAVIA_2500-20Epochs_12iters_CCData_Limit10_training.png'
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    num_remove=0,
    plot_title='CAVIA Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA/CAVIA_2500-20Epochs_12iters_CCData_Limit10_testTrain.png'
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/',
    num_days=360,
    plot_title='CAVIA Quantum Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA/CAVIA_2500-20Epochs_12iters_CCData_Limit10_marketTest.png',
    legend_loc='upper left',
    trading_limit=10
)
market_test_detail(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA/CAVIA_2500-20Epochs_12iters_CCData_Limit10_marketTestDetail.png',
    plot_title='CAVIA Market Test Detail on Test Portfolio',
    num_days=360,
    trading_limit=10
)
################
# MAML Quantum BS=False#
################
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/train/E=50_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_BS=False',
    num_remove=0,
    plot_title='MAML Quantum (BS=False) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_training.png'
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_BS=False',
    num_remove=0,
    plot_title='MAML Quantum (BS=False) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_testTrain.png'
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=50,
    plot_title='MAML Quantum (BS=False) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=None
)
market_test_detail(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_marketTestDetail.png',
    plot_title='MAML Quantum (BS=False) Market Test Detail on Test Portfolio',
    num_days=50,
    trading_limit=200
)

#################
# CAVIA Quantum BS=False#
#################
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/train/E=50_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_NCP=2_BS=False',
    num_remove=0,
    plot_title='CAVIA Quantum (BS=False) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_training.png'
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_NCP=2_BS=False',
    num_remove=0,
    plot_title='CAVIA Quantum (BS=False) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_testTrain.png'
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_NCP=2_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=50,
    plot_title='CAVIA Quantum (BS=False) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=None
)
market_test_detail(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_NCP=2_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_marketTestDetail.png',
    plot_title='CAVIA Quantum (BS=False) Market Test Detail on Test Portfolio',
    num_days=50,
    trading_limit=200
)

################
# MAML Quantum BS=True#
################
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/train/E=50_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_BS=True',
    num_remove=0,
    plot_title='MAML Quantum (BS=True) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_training.png'
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_BS=True',
    num_remove=0,
    plot_title='MAML Quantum (BS=True) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_testTrain.png'
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_BS=True',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=50,
    plot_title='MAML Quantum (BS=True) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=None
)
market_test_detail(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_BS=True',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_marketTestDetail.png',
    plot_title='MAML Quantum (BS=True) Market Test Detail on Test Portfolio',
    num_days=50,
    trading_limit=200
)

#################
# CAVIA Quantum BS=True#
#################
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/train/E=50_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_NCP=2_BS=True',
    num_remove=0,
    plot_title='CAVIA Quantum (BS=True) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_training.png'
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_NCP=2_BS=True',
    num_remove=0,
    plot_title='CAVIA Quantum (BS=True) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_testTrain.png'
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_NCP=2_BS=True',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=50,
    plot_title='CAVIA Quantum (BS=True) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=None
)
market_test_detail(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=50_NCP=2_BS=True',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_marketTestDetail.png',
    plot_title='CAVIA Quantum (BS=True) Market Test Detail on Test Portfolio',
    num_days=50,
    trading_limit=200
)

# df = wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/train/E=5000_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=5_WS=10_ND=360', sample = 'train')
#
# def sigmoid(x):
#     return (2 / (1 + np.exp(-x))) - 1
#
#
# df["Mean"] = df.mean(axis=1)
# df["STD"] = df.std(axis=1)
# len(df)
# df
#
#
#
# # %%
# # For plotting training graphs
#
# df = df[:] # remove some rows if needed
#
# plt.plot(df.index, df.Mean)
# plt.fill_between(df.index, df.Mean - df.STD, df.Mean + df.STD, color = (0.1,0.2,0.7,0.3))
# # plt.ylim(-3,3)
# plt.xlim(0,500)
# plt.xlabel('Epochs')
# plt.ylabel('Rewards')
# plt.title('CAVIA Training on CC Stock Portfolios')
# plt.savefig('C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_9iters_CCData_training_3_zoom.png')
# plt.show()
#
# # %%
#
# # For plotting market tests
#
# # TODO: Do this with portfolio data...
# # Load in the stock data to simulate what would happen without trading actions
# data, names = load_data("C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/",360)
# data = [d[int(len(d)*.7):-1] for d in data]
# len(data[0])
#
# limit = 5
# starting_money = 10000
# data = [(np.array(d) - d[0]) * limit for d in data]
# data = [sum(x) + starting_money for x in zip(*data)]
# data
# df['market_value'] = data[:-1]
#
# #%%
# plt.plot(df.index, df.Mean, label='Mean Balance')
# plt.plot(df.market_value, label='Market Value', color='green')
# plt.fill_between(df.index, df.Mean - df.STD, df.Mean + df.STD, color = (0.1,0.2,0.7,0.3))
# plt.legend(loc='upper left')
# plt.xlabel('Timestep')
# plt.ylabel('Total Value ($)')
# plt.title('CAVIA Market Test on Portfolio')
# plt.tight_layout()
# plt.savefig('C:/Github/QuantumResearch/NES_Meta_Trading/graphics/10000-20Epochs_9iters_defaultparams_marketTest_classic_CAVIA.png')
# # plt.show()

# %%
