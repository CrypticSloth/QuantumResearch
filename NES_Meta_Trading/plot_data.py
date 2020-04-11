import numpy as np
import pandas as pd
from collections import defaultdict
import os
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
sns.set()

plt.rcParams.update({
                    'axes.titlesize': 25,
                    'axes.labelsize': 20,
                    'legend.fontsize': 20,
                    'xtick.labelsize': 16,
                    'ytick.labelsize': 16,
                    'figure.figsize' : [10,6],
                    'savefig.dpi' : 500
                    })

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

num_days = 50
df = pd.read_csv('C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/ABC.csv')[0:num_days]
df
df[int(len(df)*.7):-1]

def wrangle_data(path, sample):
    '''
        Input the path to the directory that contains the results and reshape it

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


# Return the market value of the stocks traded
# Classical
data, _ = load_data('C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/', 360)
data = [d[int(len(d)*.7):-1] for d in data]
limit = 10
starting_money = 10000
data = [(np.array(d) - d[0]) * limit for d in data]
data = [sum(x) + starting_money for x in zip(*data)]
data[-2]

# Quantum
data, _ = load_data('C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/', 50)
data = [d[int(len(d)*.7):-1] for d in data]
limit = 200
starting_money = 10000
data = [(np.array(d) - d[0]) * limit for d in data]
data = [sum(x) + starting_money for x in zip(*data)]
data[-1]

# Return the final dolar amount for each test run.
# MAML
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360', 'test').mean(axis=1))[-2]
# CAVIA
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360', 'test').mean(axis=1))[-2]
# Quantum MAML (BS=False)
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False', 'test').mean(axis=1))[-1]
# Quantum MAML (BS=True)
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=True', 'test').mean(axis=1))[-1]
# Quantum CAVIA (BS=False)
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False', 'test').mean(axis=1))[-1]
# Quantum CAVIA (BS=True)
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=True', 'test').mean(axis=1))[-1]

# Return the std for each test run.
# MAML
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360', 'test').std(axis=1))[-2]
# CAVIA
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360', 'test').std(axis=1))[-2]
# Quantum MAML (BS=False)
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False', 'test').std(axis=1))[-1]
# Quantum MAML (BS=True)
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=True', 'test').std(axis=1))[-1]
# Quantum CAVIA (BS=False)
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False', 'test').std(axis=1))[-1]
# Quantum CAVIA (BS=True)
np.array(wrangle_data('C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=True', 'test').std(axis=1))[-1]


def training_plot(trial_path, num_remove, ylim, plot_title, plot_save_loc, xlim=None):
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
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.xlabel('Epochs')
    plt.ylabel('Rewards')
    plt.title(plot_title)
    plt.savefig(plot_save_loc)
    plt.show()

def market_test_plot(trial_path, data_path, num_days, plot_title, plot_save_loc, legend_loc, ylim, trading_limit, starting_money=10000, offset=-1):
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
    plt.ylim(ylim)
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(plot_save_loc)

def market_test_detail(trial_path1, trial_path2, data_path, title1, title2, plot_save_loc, num_days, trading_limit, starting_money=10000):
    # trial_path = 'C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360'
    tableau10 = [(31, 119, 180), (255, 127, 14),
             (44, 160, 44), (214, 39, 40),
             (148, 103, 189),  (140, 86, 75),
             (227, 119, 194),  (127, 127, 127),
             (188, 189, 34),  (23, 190, 207)]

    for i in range(len(tableau10)):
        r, g, b = tableau10[i]
        tableau10[i] = (r / 255., g / 255., b / 255.)

    # Process first trial data
    df1 = wrangle_data(trial_path1, sample = 'portfolio')
    means1 = []
    stds = []
    for i in range(1,len(df1)):
        means1.append(np.mean(np.array(df1[i]), axis=0))
        stds.append(np.std(np.array(df1[i]), axis=0))
    means1 = np.around(means1, 2)
    means1 = means1 * 100

    # Process second trial data
    df2 = wrangle_data(trial_path2, sample = 'portfolio')
    means2 = []
    stds = []
    for i in range(1,len(df2)):
        means2.append(np.mean(np.array(df2[i]), axis=0))
        stds.append(np.std(np.array(df2[i]), axis=0))
    means2 = np.around(means2, 2)
    means2 = means2 * 100

    # Process data for plot 3
    data, names = load_data(data_path,num_days)
    data = [d[int(len(d)*.7):-1] for d in data]
    trading_limit = 10
    starting_money = 10000
    limit = trading_limit
    starting_money = starting_money
    data = [((np.array(d) - d[0]) * limit) + starting_money for d in data]
    # sanity check
    # print(len(means1))
    # print([x.sum() for x in np.array(means1).T])

    # Initialize plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(20,15))
    fig.tight_layout(pad=5.0)

    # Plot first plot
    ind = np.arange(len(means1[0]))
    for i in range(len(means1)):
        if i == 0:
            ax1.bar(ind, means1[i], label='Cash', color=tableau10[i], width=0.85)
        else:
            ax1.bar(ind, means1[i], bottom=np.sum(means1[0:i].T, axis=1), label=names[i-1], color=tableau10[i], width=0.85)

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0))
    ax1.set_title(title1)
    # ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Total Percent Held of Portfolio')

    # Plot second plot
    ind = np.arange(len(means2[0]))
    for i in range(len(means2)):
        if i == 0:
            ax2.bar(ind, means2[i], label='Cash', color=tableau10[i], width=0.85)
        else:
            ax2.bar(ind, means2[i], bottom=np.sum(means2[0:i].T, axis=1), label=names[i-1], color=tableau10[i], width=0.85)

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.set_title(title2)
    # ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Total Percent Held of Portfolio')

    # Plot third plot
    for i in range(len(data)):
        ax3.plot(range(len(data[i])),data[i],label=names[i],color=tableau10[i+1])

    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3.set_title('Market Value')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Total Value ($)')

    # fig.suptitle(plot_title)
    fig.savefig(plot_save_loc)

def market_test_stock(trial_path1, trial_path2, data_path, stock_name, plot_save_loc, num_days, offset, trading_limit=10, starting_money=10000):
    # trial_path1 = 'C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360'
    # trial_path2 = 'C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360'
    #
    # data_path = 'C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/'
    # stock_name = 'AKAM'
    # trading_limit = 10
    # starting_money = 10000
    # num_days = 360
    # offset = -1

    tableau10 = [(31, 119, 180), (255, 127, 14),
             (44, 160, 44), (214, 39, 40),
             (148, 103, 189),  (140, 86, 75),
             (227, 119, 194),  (127, 127, 127),
             (188, 189, 34),  (23, 190, 207)]

    for i in range(len(tableau10)):
        r, g, b = tableau10[i]
        tableau10[i] = (r / 255., g / 255., b / 255.)

    # Process market data
    data, names = load_data(data_path,num_days)
    data = [d[int(len(d)*.7):-1] for d in data]
    limit = trading_limit
    starting_money = starting_money
    data = [((np.array(d) - d[0]) * limit) + starting_money for d in data]

    # Process trading data 1
    df1 = wrangle_data(trial_path1, sample = 'portfolio')
    means1 = []
    stds1 = []
    for i in range(1,len(df1)):
        means1.append(np.mean(np.array(df1[i]), axis=0))
        stds1.append(np.std(np.array(df1[i]), axis=0))
    means1 = np.around(means1, 2)
    means1 = means1 * 100

    # Process trading data 1
    df2 = wrangle_data(trial_path2, sample = 'portfolio')
    means2 = []
    stds2 = []
    for i in range(1,len(df2)):
        means2.append(np.mean(np.array(df2[i]), axis=0))
        stds2.append(np.std(np.array(df2[i]), axis=0))
    means2 = np.around(means2, 2)
    means2 = means2 * 100

    # Select the data for the given stock
    ind = names.index(stock_name)
    means1 = means1[ind]
    means2 = means2[ind]
    stds1 = stds1[ind]
    stds2 = stds2[ind]
    data = data[ind]
    data = data[:offset]

    # %%
    fig,ax = plt.subplots()
    ax.title.set_text("Stock '{:}' Model Weights vs Market Value".format(stock_name))
    ax.plot(range(len(data)), means1, color=tableau10[0], label='MAML')
    ax.plot(range(len(data)), means2, color=tableau10[1], label='CAVIA')
    ax.set_xlabel("timesteps")
    ax.set_ylabel("Stock Weights")
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:}%'.format(int(x)) for x in vals])
    ax.legend(loc='upper right')

    ax2 = ax.twinx()
    ax2.plot(range(len(data)), data, color=tableau10[2], linewidth=2.5)
    ax2.set_ylabel("Market Value", color=tableau10[2])
    vals = ax2.get_yticks()
    ax2.set_yticklabels(['${:}'.format(int(x)) for x in vals])

    fig.savefig(plot_save_loc)
    # plt.show()

market_test_stock(
    trial_path1= 'C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    trial_path2= 'C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    data_path = 'C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/',
    stock_name = 'AKAM',
    plot_save_loc = 'C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA/detailed_trading_view_classical.png',
    num_days = 360,
    offset=-1
)

# Quantum BS=True
market_test_stock(
    trial_path1='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=True',
    trial_path2='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=True',
    data_path = 'C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    stock_name = 'ABC',
    plot_save_loc = 'C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/detailed_trading_view_quantum_BS=True',
    num_days = 180,
    offset=-1
)

# Quantum BS=False
market_test_stock(
    trial_path1='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False',
    trial_path2='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False',
    data_path = 'C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    stock_name = 'ABC',
    plot_save_loc = 'C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/',
    num_days = 180
)

    # %%
    # plt.plot(range(len(data)), means1, label='Mean Balance')
    # plt.plot(data, label='Market Value', color='green')
    # plt.fill_between(range(len(data)), means1 - stds, means1 + stds, color = (0.1,0.2,0.7,0.3))
    # # plt.legend(loc=legend_loc)
    # plt.xlabel('Timestep')
    # plt.ylabel('Total Value ($)')
    # # plt.ylim(ylim)
    # # plt.title(plot_title)
    # plt.tight_layout()
    # # plt.savefig(plot_save_loc)
    # plt.show()
    # %%

    return
    # %%
########
# MAML #
########
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/train/E=2500_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    num_remove=0,
    plot_title='MAML Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML/MAML_360-20Epochs_12iters_CCData_Limit10_training.png',
    ylim=(-1.5,9),
    xlim=(-10,1000)
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    num_remove=0,
    plot_title='MAML Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML/MAML_360-20Epochs_12iters_CCData_Limit10_testTrain.png',
    ylim=(-1.5,9)
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/',
    num_days=360,
    plot_title='MAML Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML/MAML_360-20Epochs_12iters_CCData_Limit10_marketTest.png',
    legend_loc='upper left',
    trading_limit=10,
    ylim=None
)

#########
# CAVIA #
#########
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/train/E=2500_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    num_remove=0,
    plot_title='CAVIA Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA/CAVIA_2500-20Epochs_12iters_CCData_Limit10_training.png',
    ylim=(-1,6),
    xlim=(-10,500)
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    num_remove=0,
    plot_title='CAVIA Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA/CAVIA_2500-20Epochs_12iters_CCData_Limit10_testTrain.png',
    ylim=(-1,6)
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/',
    num_days=360,
    plot_title = 'CAVIA Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA/CAVIA_2500-20Epochs_12iters_CCData_Limit10_marketTest.png',
    legend_loc='upper left',
    trading_limit=10,
    ylim=None
)

# Detailed trading plot
market_test_detail(
    trial_path1='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    trial_path2='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia/test/E=20_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=10_WS=10_ND=360',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test/',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA/CAVIA_MAML_2500-20Epochs_12iters_CCData_Limit10_marketTestDetail.png',
    title1='MAML Distribution of Portfolio',
    title2='CAVIA Distribution of Portfolio',
    num_days=360,
    trading_limit=10
)
################
# MAML Quantum BS=False#
################
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/train/E=25_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False',
    num_remove=0,
    plot_title='MAML Quantum (BS=False) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_training.png',
    ylim=(-15,72)
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False',
    num_remove=0,
    plot_title='MAML Quantum (BS=False) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_testTrain.png',
    ylim=(-15, 72)
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=180,
    plot_title='MAML Quantum (BS=False) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=-1,
    ylim=(4500,10900)
)

#################
# CAVIA Quantum BS=False#
#################
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/train/E=25_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False',
    num_remove=0,
    plot_title='CAVIA Quantum (BS=False) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_training.png',
    ylim=(-25,62)
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False',
    num_remove=0,
    plot_title='CAVIA Quantum (BS=False) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_testTrain.png',
    ylim=(-25, 62)
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=180,
    plot_title='CAVIA Quantum (BS=False) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSFalse_50-10Epochs_12iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=-1,
    ylim=(4500,10900)
)

market_test_detail(
    trial_path1='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False',
    trial_path2='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_MAML_Quantum_BSFalse_2500-20Epochs_12iters_CCData_Limit10_marketTestDetail.png',
    title1='MAML Quantum (BS=False) Distribution of Portfolio',
    title2='CAVIA Quantum (BS=False) Distribution of Portfolio',
    num_days=180,
    trading_limit=200
)

################
# MAML Quantum BS=True#
################
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/train/E=25_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=True',
    num_remove=0,
    plot_title='MAML Quantum (BS=True) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_training.png',
    ylim=(-18,39)
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=True',
    num_remove=0,
    plot_title='MAML Quantum (BS=True) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_testTrain.png',
    ylim=(-18, 39)
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=True',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=180,
    plot_title='MAML Quantum (BS=True) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/MAML_Quantum/MAML_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=-1,
    ylim=(4500,10900)
)

#################
# CAVIA Quantum BS=True#
#################
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/train/E=25_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=True',
    num_remove=0,
    plot_title='CAVIA Quantum (BS=True) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_training.png',
    ylim=(-15,37)
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=True',
    num_remove=0,
    plot_title='CAVIA Quantum (BS=True) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_testTrain.png',
    ylim=(-15, 37)
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=True',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=180,
    plot_title='CAVIA Quantum (BS=True) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_Quantum_BSTrue_50-10Epochs_12iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=-1,
    ylim=(4500,10900)
)

market_test_detail(
    trial_path1='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=True',
    trial_path2='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=True',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/CAVIA_Quantum/CAVIA_MAML_Quantum_BSTrue_2500-20Epochs_12iters_CCData_Limit10_marketTestDetail.png',
    title1='MAML Quantum (BS=True) Distribution of Portfolio',
    title2='CAVIA Quantum (BS=True) Distribution of Portfolio',
    num_days=180,
    trading_limit=200
)

#################
# QMAML and QCAVIA on 180 days of trading
#################
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/train/E=25_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False',
    num_remove=0,
    plot_title='MAML Quantum (BS=False) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/Quantum_180_Days/MAML_Quantum_BSFalse_25-5Epochs_8iters_CCData_LimitNone_training.png'
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False',
    num_remove=0,
    plot_title='MAML Quantum (BS=False) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/Quantum_180_Days/MAML_Quantum_BSFalse_25-5Epochs_8iters_CCData_LimitNone_testTrain.png'
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=180,
    plot_title='MAML Quantum (BS=False) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/Quantum_180_Days/MAML_Quantum_BSFalse_25-5Epochs_8iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=-1
)

######################################

training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/train/E=25_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False',
    num_remove=0,
    plot_title='CAVIA Quantum (BS=False) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/Quantum_180_Days/CAVIA_Quantum_BSFalse_25-5Epochs_8iters_CCData_LimitNone_training.png'
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False',
    num_remove=0,
    plot_title='CAVIA Quantum (BS=False) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/Quantum_180_Days/CAVIA_Quantum_BSFalse_25-5Epochs_8iters_CCData_LimitNone_testTrain.png'
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=180,
    plot_title='CAVIA Quantum (BS=False) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/Quantum_180_Days/CAVIA_Quantum_BSFalse_25-5Epochs_8iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=-1
)
market_test_detail(
    trial_path1='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_BS=False',
    trial_path2='C:/Github/QuantumResearch/NES_Meta_Trading/results/cavia_quantum/test/E=5_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=180_NCP=2_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/Quantum_180_Days/CAVIA_MAML_Quantum_BSFalse_25-5Epochs_8iters_CCData_LimitNone_marketTestDetail.png',
    title1='MAML Quantum (BS=False) Distribution of Portfolio',
    title2='CAVIA Quantum (BS=False) Distribution of Portfolio',
    num_days=180,
    trading_limit=200
)

#################
# QMAML and QCAVIA on 360 days of trading
#################
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/train/E=25_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=360_BS=False',
    num_remove=0,
    plot_title='MAML Quantum (BS=False) Training on CC Stock Portfolios',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/Quantum_360_Days/MAML_Quantum_BSFalse_25-5Epochs_8iters_CCData_LimitNone_training.png'
)
training_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=360_BS=False',
    num_remove=0,
    plot_title='MAML Quantum (BS=False) Test Training on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/Quantum_360_Days/MAML_Quantum_BSFalse_25-5Epochs_8iters_CCData_LimitNone_testTrain.png'
)
market_test_plot(
    trial_path='C:/Github/QuantumResearch/NES_Meta_Trading/results/maml_quantum/test/E=10_PS=15_S=0.1_LR=0.03_sk=1_IM=10000_L=None_WS=1_ND=360_BS=False',
    data_path='C:/Github/QuantumResearch/NES_Meta_Trading/dataset/test_cavia/',
    num_days=360,
    plot_title='MAML Quantum (BS=False) Market Test on Test Portfolio',
    plot_save_loc='C:/Github/QuantumResearch/NES_Meta_Trading/graphics/Quantum_360_Days/MAML_Quantum_BSFalse_25-5Epochs_8iters_CCData_LimitNone_marketTest.png',
    legend_loc='lower left',
    trading_limit=200,
    offset=-1
)

######################################
