
# coding: utf-8

# In[2]:

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
sns.set()

import os
os.chdir("C:/Github/QuantumResearch/NES_Meta_Trading/")

# In[58]:

def load_data(path, num_days = 30):
    '''
    load in all stock data from the path and return the close values
    '''

    paths = os.listdir(path)

    data = []
    names = []
    for p in paths:
        data.append(pd.read_csv(path + p).Close.values.tolist()[0:num_days])
        names.append(p[:-4])

    # Check that all data are the same size
    assert len(set([len(d) for d in data])) == 1, "Stock data has differing number of days recorded"

    return np.array([data]).flatten(), names


def get_state(data, t, n, num_days,num_stocks):
    '''
        returns an array of an array of size n with the time step of t
        of how much the close value differed from the day before
    '''

    data = data.reshape(num_stocks,num_days).tolist()

    stocks = []
    for s in data:
        d = t - n + 1
        block = s[d : t + 1] if d >= 0 else -d * [s[0]] + s[: t + 1]
        res = []
        for i in range(n - 1):
            res.append(block[i + 1] - block[i])
        stocks.append(res)
    return np.array([np.array([stocks]).flatten()])


# In[60]:


num_days = 30
close, names = load_data("dataset/train/",num_days)
num_stocks = len(names) # This will need to be used to calculate the iterations and input layer sizes along with num_days
num_stocks
np.shape(close)
len(close)
get_state(close, 19, 10, num_days,num_stocks)

close.reshape(num_stocks, num_days)
# In[63]:


class Deep_Evolution_Strategy:
    def __init__(
        self, weights, reward_function, population_size, sigma, learning_rate
    ):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        return self.weights

    def train(self, epoch = 100, print_every = 1):
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.weights, population[k]
                )
                rewards[k] = self.reward_function(weights_population, split = "train")
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 0.00001) # Normalized the rewards here. Do we need to normalize if they are log returns?
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T + 0.00001# Our task is to make this meta by storing each gradient into a global gradient from the MAML paper
                )
            if (i + 1) % print_every == 0:
                print(
                    'iter %d. reward: %f'
                    % (i + 1, self.reward_function(self.weights, return_reward = True, split = "train"))
                )
        print('time taken to train:', time.time() - lasttime, 'seconds')


# In[64]:

class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, layer_size),
            # np.random.randn(layer_size, output_size), # decision, output we need to do (Do nothing = 0; Buy = 1; sell = 2)
            # np.random.randn(layer_size, 1), # buy, how many units quantity we need to buy
            np.random.randn(layer_size,output_size), # This will have the softmax applied to it...
            np.random.randn(1, layer_size), # Bias layer for our first feed-forward
        ]

    # To make this deterministic, out ouput is going to be weights for each stock we have
    # For eg, if we have 5 stocks out output will be a column of (1,5) where each row is
    # a percentage of how much of that stock we want to have in our portfolio
    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        decision = np.dot(decision, self.weights[2])
        # portfolio = softmax(decision)
        # buy = [0.75]
        # buy = np.dot(feed, self.weights[2])
        # return decision, buy
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

# %% ###########################################################################
# Quantum Model

class QuantumModel:
    def __init__(self, num_layers, output_size, bs):
        self.num_layers = num_layers
        self.output_size = output_size
        self.bs = bs
        self.weights = 0.05 * np.random.randn(self.num_layers, 5 * 5)

    # NOTE: 5 wires gives a memory error so 4 seems to be the max
    dev = qml.device("strawberryfields.fock", wires=output_size, cutoff_dim=10)

    if bs == False:
        def layer(self, w):
            '''
                For each weight (w[i]) apply the quantum gates to it.

                Saves the updated weights to the quantum device.

                w: a list of scalar weights (of length 5)
            '''

            # Matrix multiplication of input layer
            qml.Rotation(w[0], wires=0)
            qml.Rotation(w[1], wires=1)
            qml.Rotation(w[2], wires=2)
            qml.Rotation(w[3], wires=3)
            # qml.Rotation(w[4], wires=4)

            qml.Squeezing(w[5], 0.0, wires=0)
            qml.Squeezing(w[6], 0.0, wires=1)
            qml.Squeezing(w[7], 0.0, wires=2)
            qml.Squeezing(w[8], 0.0, wires=3)
            # qml.Squeezing(w[9], 0.0, wires=4)

            qml.Rotation(w[10], wires=0)
            qml.Rotation(w[11], wires=1)
            qml.Rotation(w[12], wires=2)
            qml.Rotation(w[13], wires=3)
            # qml.Rotation(w[14], wires=4)

            # Bias
            qml.Displacement(w[15], 0.0, wires=0)
            qml.Displacement(w[16], 0.0, wires=1)
            qml.Displacement(w[17], 0.0, wires=2)
            qml.Displacement(w[18], 0.0, wires=3)
            # qml.Displacement(w[19], 0.0, wires=4)

            # Element-wise nonlinear transformation
            qml.Kerr(w[20], wires=0)
            qml.Kerr(w[21], wires=1)
            qml.Kerr(w[22], wires=2)
            qml.Kerr(w[23], wires=3)
            # qml.Kerr(w[24], wires=4)

        @qml.qnode(dev)
        def quantum_neural_net(self, weights, x=None):
            '''
                For each layer, apply the inputs to the gates to update the weights

                weights: list of lists of scalar weights (of length 5)
                x: list of stock closing values for 5 stocks
            '''
            # Encode input x into quantum state
            qml.Displacement(x[0], 0.0, wires=0)
            qml.Displacement(x[1], 0.0, wires=1)
            qml.Displacement(x[2], 0.0, wires=2)
            qml.Displacement(x[3], 0.0, wires=3)
            # qml.Displacement(x[4], 0.0, wires=4)

            # "layer" subcircuits
            for w in weights:
                self.layer(w)

            return [qml.expval(qml.X(0)),
                    qml.expval(qml.X(1)),
                    qml.expval(qml.X(2)),
                    qml.expval(qml.X(3))]
                    # qml.expval(qml.X(4))]

        def predict(self, weights, inputs):
            '''
                Loop through each of the training data and apply it to the quantum network to get a prediction for each value.

                Will need to somehow make the QNN shape the values to output 5 values for the softmax function. Not sure how to do this since the network only updates with scalar values and the output is the size of the number of inputs.
            '''
            preds = np.array([self.quantum_neural_net(self.weights, x=x) for x in inputs.T])

            return [np.mean(p) for p in preds.T]

    if bs == True:

        def layer(self, w):
            '''
                For each weight (w[i]) apply the quantum gates to it.

                Saves the updated weights to the quantum device.

                w: a list of scalar weights (of length 5)
            '''

            qml.Displacement(w[0], 0.0, wires=0)
            qml.Displacement(w[1], 0.0, wires=1)
            qml.Displacement(w[2], 0.0, wires=2)
            qml.Displacement(w[3], 0.0, wires=3)

            qml.Displacement(w[4], w[5], wires=0)
            qml.Displacement(w[6], w[7], wires=1)
            qml.Displacement(w[8], w[9], wires=2)
            qml.Displacement(w[10], w[11], wires=3)

            qml.Squeezing(w[12], w[13], wires=0)
            qml.Squeezing(w[14], w[15], wires=1)
            qml.Squeezing(w[16], w[17], wires=2)
            qml.Squeezing(w[18], w[19], wires=3)

            qml.Kerr(w[20], wires=0)
            qml.Kerr(w[21], wires=1)
            qml.Kerr(w[22], wires=2)
            qml.Kerr(w[23], wires=3)

            qml.Beamsplitter(w[24],0, wires=[0,1])
            qml.Beamsplitter(0,0, wires=[0,1])
            qml.Beamsplitter(w[25],0, wires=[1,2])
            qml.Beamsplitter(0,0, wires=[1,2])
            qml.Beamsplitter(w[26],0, wires=[2,3])
            qml.Beamsplitter(0,0, wires=[2,3])

            qml.Displacement(w[27], w[28], wires=0)
            qml.Displacement(w[29], w[30], wires=1)
            qml.Displacement(w[31], w[32], wires=2)
            qml.Displacement(w[33], w[34], wires=3)

            qml.Squeezing(w[35], w[36], wires=0)
            qml.Squeezing(w[37], w[38], wires=1)
            qml.Squeezing(w[39], w[40], wires=2)
            qml.Squeezing(w[41], w[42], wires=3)

            qml.Kerr(w[43], wires=0)
            qml.Kerr(w[44], wires=1)
            qml.Kerr(w[45], wires=2)
            qml.Kerr(w[46], wires=3)

            qml.Beamsplitter(w[47],0, wires=[0,1])
            qml.Beamsplitter(0,0, wires=[0,1])
            qml.Beamsplitter(w[48],0, wires=[1,2])
            qml.Beamsplitter(0,0, wires=[1,2])
            qml.Beamsplitter(w[49],0, wires=[2,3])
            qml.Beamsplitter(0,0, wires=[2,3])


        @qml.qnode(dev)
        def quantum_neural_net(self, weights, x=None):
            '''
                For each layer, apply the inputs to the gates to update the weights

                weights: list of lists of scalar weights (of length 5)
                x: list of stock closing values for 5 stocks
            '''
            # Encode input x into quantum state
            qml.Displacement(x[0], 0.0, wires=0)
            qml.Displacement(x[1], 0.0, wires=1)
            qml.Displacement(x[2], 0.0, wires=2)
            qml.Displacement(x[3], 0.0, wires=3)
            # qml.Displacement(x[4], 0.0, wires=4)

            # "layer" subcircuits
            for w in weights:
                self.layer(w)

            return [qml.expval(qml.X(0)),
                    qml.expval(qml.X(1)),
                    qml.expval(qml.X(2)),
                    qml.expval(qml.X(3))]
                    # qml.expval(qml.X(4))]

        def predict(self, weights, inputs):
            '''
                Loop through each of the training data and apply it to the quantum network to get a prediction for each value.

                Will need to somehow make the QNN shape the values to output 5 values for the softmax function. Not sure how to do this since the network only updates with scalar values and the output is the size of the number of inputs.
            '''
            preds = np.array([self.quantum_neural_net(self.weights, x=x) for x in inputs.T])

            return [np.mean(p) for p in preds.T]


# num_layers = 4 # 4-6 number of layers
# output_size = 4 # Number of stocks
# weights = 0.05 * np.random.randn(num_layers, 5 * 5) # the 5*5 just has to accomidate the number of stocks * number of gates we will have
# np.shape(weights)
#
# weights
#
# # for layer in weights:
# #     for w in layer:
# #         print(w)
#
# # NOTE: 5 wires gives a memory error so 4 seems to be the max
# dev = qml.device("strawberryfields.fock", wires=output_size, cutoff_dim=10)
#
# def layer(w):
#     '''
#         For each weight (w[i]) apply the quantum gates to it.
#
#         Saves the updated weights to the quantum device.
#
#         w: a list of scalar weights (of length 5)
#     '''
#     print()
#     # Matrix multiplication of input layer
#     qml.Rotation(w[0], wires=0)
#     qml.Rotation(w[1], wires=1)
#     qml.Rotation(w[2], wires=2)
#     qml.Rotation(w[3], wires=3)
#     # qml.Rotation(w[4], wires=4)
#
#     qml.Squeezing(w[5], 0.0, wires=0)
#     qml.Squeezing(w[6], 0.0, wires=1)
#     qml.Squeezing(w[7], 0.0, wires=2)
#     qml.Squeezing(w[8], 0.0, wires=3)
#     # qml.Squeezing(w[9], 0.0, wires=4)
#
#     qml.Rotation(w[10], wires=0)
#     qml.Rotation(w[11], wires=1)
#     qml.Rotation(w[12], wires=2)
#     qml.Rotation(w[13], wires=3)
#     # qml.Rotation(w[14], wires=4)
#
#     # Bias
#     qml.Displacement(w[15], 0.0, wires=0)
#     qml.Displacement(w[16], 0.0, wires=1)
#     qml.Displacement(w[17], 0.0, wires=2)
#     qml.Displacement(w[18], 0.0, wires=3)
#     # qml.Displacement(w[19], 0.0, wires=4)
#
#     # Element-wise nonlinear transformation
#     qml.Kerr(w[20], wires=0)
#     qml.Kerr(w[21], wires=1)
#     qml.Kerr(w[22], wires=2)
#     qml.Kerr(w[23], wires=3)
#     # qml.Kerr(w[24], wires=4)
#
# @qml.qnode(dev)
# def quantum_neural_net(weights, x=None):
#     '''
#         For each layer, apply the inputs to the gates to update the weights
#
#         weights: list of lists of scalar weights (of length 5)
#         x: list of stock closing values for 5 stocks
#     '''
#     # Encode input x into quantum state
#     qml.Displacement(x[0], 0.0, wires=0)
#     qml.Displacement(x[1], 0.0, wires=1)
#     qml.Displacement(x[2], 0.0, wires=2)
#     qml.Displacement(x[3], 0.0, wires=3)
#     # qml.Displacement(x[4], 0.0, wires=4)
#
#     # "layer" subcircuits
#     for w in weights:
#         layer(w)
#
#     return [qml.expval(qml.X(0)),
#             qml.expval(qml.X(1)),
#             qml.expval(qml.X(2)),
#             qml.expval(qml.X(3))]
#             # qml.expval(qml.X(4))]
#
# def predict(weights, inputs):
#     '''
#         Loop through each of the training data and apply it to the quantum network to get a prediction for each value.
#
#         Will need to somehow make the QNN shape the values to output 5 values for the softmax function. Not sure how to do this since the network only updates with scalar values and the output is the size of the number of inputs.
#     '''
#     preds = np.array([quantum_neural_net(weights, x=x) for x in inputs.T])
#
#     return [np.mean(p) for p in preds.T]

# def get_weights(self):
#     return self.weights
#
# def set_weights(self, weights):
#     self.weights = weights

num_days = 30
close, names = load_data("dataset/train/",num_days)
num_stocks = len(names) # This will need to be used to calculate the iterations and input layer sizes along with num_days
num_stocks
np.shape(close)

model = QuantumModel(num_layers = 4, output_size = num_stocks)

cur_state = get_state(close, 10, window_size + 1, num_days, num_stocks)
# act(model, np.array([0.,0.,0.,0.]))

cur_state = cur_state.reshape(num_stocks, window_size)
np.shape(cur_state)

dev.reset()
preds = model.predict(weights, cur_state) # This is all we need..
preds

# softmax(np.array([preds]))

# In[66]: ######################################################################

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=1) + 0.00001)

def act(model, sequence):
    decision = model.predict(np.array(sequence))
    return softmax(decision)

def buy_stock(portfolio, close_s, money, inventory, limit, t):
    """
        Function that takes in portfolio weights (percentage of each stock in the entire portfolio),
        the current stock prices (close price) and the money we currently have
        and calculates the maximum number of stocks we can buy with the weights given in the portfolio.

        Inventory is the dictionary containing how many stocks we own.
        Limit puts a maximum number of stock we can purchase
        t is the current time step

        TODO: instead of dealing with cash amounts we should deal with normalized return (Ri - mean_R) / (std_R)
    """

    c = 0
    cash = np.sum([close_s[i][t] * inventory[i] for i in range(len(close_s))]) + money # reset our inventory into cash

    portfolio_money = portfolio[0] * cash # portfolio is an array of an array : [[]]

    p = []
    for m in portfolio_money:
        num_stock = math.floor(m / (close_s[c][t] + 0.00001))
        p.append(close_s[c][t])
        if num_stock <= limit:
            inventory[c] = num_stock
        else:
            inventory[c] = limit

        cash -= (inventory[c] * close_s[c][t])
        c += 1

    return inventory, cash

def stock_value(inventory, money, close_s, t):
    """
    Calculate current stock value of stock inventory and cash based on timestep t
    """
    cash = np.sum([close_s[i][t] * inventory[i] for i in range(len(close_s))]) + money
    return cash

# Testing one iteration of the new reward function
# This assumes we can purchase partial stocks and has no limits

num_days = 30
close, names = load_data("dataset/train/",num_days)
num_stocks = len(names) # This will need to be used to calculate the iterations and input layer sizes along with num_days
num_stocks

window_size = 10

model = Model(window_size*num_stocks, 500, 3)
# model = QuantumModel(num_layers=4)

weight = model
initial_money = 10000
starting_money = initial_money

cur_state = get_state(close, 0, window_size + 1, num_days, num_stocks)
close_s = close.reshape(num_stocks,int(len(close)/num_stocks))
skip = 1

# Initialize a dictionary to keep track of which stocks we can buy
keys = range(num_stocks)
cur_inventory = {key: 0 for key in keys}
limit = 5

split = "train"
if split == "train":
    t = close[0:int(len(close)*.7)] # This is a list of list of stock data so this doesnt work
if split == "test":
    t = close[int(len(close)*.7):-1]

# close_s
# close_s[:,0:int(len(close_s[0])*.7)]
# close_s[:,int(len(close_s[0])*.7):len(close_s[0])]

for t in range(0, len(close_s[0]) - 1, skip):

    portfolio = act(weight, cur_state)
    next_state = get_state(close, t + 1, window_size + 1,num_days,num_stocks).reshape(num_stocks,window_size)

    next_inventory, initial_money = buy_stock(portfolio, close_s, initial_money, cur_inventory, limit, t)

    cur_state = next_state.flatten()
    cur_inventory = next_inventory
((initial_money - starting_money) / (starting_money + 0.00001)) * 100
(initial_money / starting_money - 1) * 100
np.log((initial_money + 0.00001) / (starting_money + 0.00001))

# %%

import time


class Agent:

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(
        self, model, money, limit, close, window_size, skip, num_stocks, num_days
    ):
        self.window_size = window_size
        self.window_size = window_size
        self.num_stocks = num_stocks
        self.num_days = num_days
        self.skip = skip
        self.close = close
        self.model = model
        self.initial_money = money
        self.limit = limit
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act_old(self, sequence):
        decision, buy = self.model.predict(np.array(sequence))
        return np.argmax(decision[0]), int(buy[0])

    def get_reward_old(self, weights):
        initial_money = self.initial_money
        starting_money = initial_money
        # len_close = len(self.close) - 1
        len_close = int(len(self.close)/ num_stocks) - 1

        self.model.weights = weights
        state = get_state(self.close, 0, self.window_size + 1, self.num_days, self.num_stocks)
        inventory = []
        quantity = 0
        for t in range(0, len_close, self.skip):
            action, buy = self.act(state)
            next_state = get_state(self.close, t + 1, self.window_size + 1, self.num_days, self.num_stocks)
            if action == 1 and initial_money >= self.close[t]:
                if buy < 0:
                    buy = 1
                if buy > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = buy
                total_buy = buy_units * self.close[t]
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
            elif action == 2 and len(inventory) > 0:
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                else:
                    sell_units = quantity
                quantity -= sell_units
                total_sell = sell_units * self.close[t]
                initial_money += total_sell

            state = next_state
        return ((initial_money - starting_money) / starting_money) * 100 + 0.00001

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis=1) + 0.00001)

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence))
        return softmax(decision)

    def buy_stock(portfolio, close_s, money, inventory, limit, t):
        """
            Function that takes in portfolio weights (percentage of each stock in the entire portfolio),
            the current stock prices (close price) and the money we currently have
            and calculates the maximum number of stocks we can buy with the weights given in the portfolio.

            Inventory is the dictionary containing how many stocks we own.
            Limit puts a maximum number of stock we can purchase
            t is the current time step

            TODO: Getting negetive inventory values from negetive cash
            TODO: instead of dealing with cash amounts we should deal with percentage gain (normalized)?
        """

        c = 0
        cash = np.sum([close_s[i][t] * inventory[i] for i in range(len(close_s))]) + money # reset our inventory into cash

        portfolio_money = portfolio[0] * cash

        p = []
        for m in portfolio_money:
            num_stock = math.floor(m / (close_s[c][t] + 0.000001))
            p.append(close_s[c][t])
            if num_stock <= limit:
                inventory[c] = num_stock
            else:
                inventory[c] = limit

            cash -= (inventory[c] * close_s[c][t])
            c += 1

        return inventory, cash

    def get_reward(self, weights, return_reward=False, split = "train"):
        '''
            Reward function.

            Model after the reward found here: https://github.com/wassname/rl-portfolio-management/blob/master/rl_portfolio_management/environments/portfolio.py
            In the paper the variables are:
                p1 = initial_money
                p0 = starting_money

            We could add cost of trading stocks as well to this in the future.
        '''

        self.model.weights = weights

        # weight = model
        initial_money = self.initial_money
        starting_money = initial_money
        close_s = self.close.reshape(self.num_stocks,int(len(self.close)/self.num_stocks))

        # Split the data into either train or test dataset
        if split == "train":
            close_s = close_s[:,0:round(len(close_s[0])*.7)]
            num_days = round(self.num_days*0.7)
        if split == "test":
            close_s = close_s[:,round(len(close_s[0])*.7):len(close_s[0])]
            num_days = round(self.num_days*0.3)

        close = close_s.flatten() # Use the split data for close

        # Initialize a dictionary to keep track of which stocks we can buy
        keys = range(self.num_stocks)
        cur_inventory = {key: 0 for key in keys}


        cur_state = get_state(close, 0, self.window_size + 1, num_days, self.num_stocks)

        for t in range(0, len(close_s[0]) - 1, self.skip):

            portfolio = self.act(cur_state)
            next_state = get_state(close, t + 1, self.window_size + 1, num_days, self.num_stocks).reshape(self.num_stocks,self.window_size)

            next_inventory, initial_money = buy_stock(portfolio, close_s, initial_money, cur_inventory, self.limit, t)

            cur_state = next_state.flatten()
            cur_inventory = next_inventory

        rho1 = (initial_money / starting_money - 1) * 100 # rate of returns
        r1 = np.log((initial_money + 0.00001) / (starting_money + 0.00001)) # log rate of return (eq10)

        if return_reward == True:
            return rho1
        else:
            return r1


    def fit(self, iterations, checkpoint):
        self.es.train(iterations, print_every = checkpoint)

    def buy(self, split):

        weight = self.model
        initial_money = self.initial_money
        starting_money = initial_money
        close_s = self.close.reshape(self.num_stocks,round(len(self.close)/self.num_stocks))
        skip = 1

        # Split the data into either train or test dataset
        if split == "train":
            close_s = close_s[:,0:round(len(close_s[0])*.7)]
            num_days = round(self.num_days*0.7)
        if split == "test":
            close_s = close_s[:,round(len(close_s[0])*.7):len(close_s[0])]
            num_days = round(self.num_days*0.3)

        close = close_s.flatten() # Use the split data for close

        # Initialize a dictionary to keep track of which stocks we can buy
        keys = range(self.num_stocks)
        cur_inventory = {key: 0 for key in keys}

        cur_state = get_state(close, 0, self.window_size + 1, num_days, self.num_stocks)

        inv = []

        for t in range(0, len(close_s[0]) - 1, self.skip):

            portfolio = self.act(cur_state)
            next_state = get_state(close, t + 1, self.window_size + 1, num_days, self.num_stocks).reshape(self.num_stocks,self.window_size)

            next_inventory, initial_money = buy_stock(portfolio, close_s, initial_money, cur_inventory, self.limit, t)

            # record the inventory
            inv_list = []
            for key,value in next_inventory.items():
                inv_list.append(value)
            inv.append(inv_list)

            cur_state = next_state.flatten()
            cur_inventory = next_inventory

        rho1 = (initial_money / starting_money - 1) * 100 # rate of returns

        inv = np.array(inv)
        inv_d = []
        for i in range(len(inv) - 1):
            inv_d.append(inv[i+1] - inv[i])
        inv_d = np.array(inv_d)

        inv_f = []
        for i in range(len(inv_d[0])):
            inv_f.append(np.array(inv_d)[:,i])
        inv_f = np.array(inv_f)

        print("Inventory at every timestep: \n ",inv)
        print(
            '\ntotal gained %f, total investment %f %%'
            % (initial_money - starting_money, rho1)
        )
        for i in range(len(close_s)):
            plt.figure(figsize = (20, 10))
            plt.title(names[i])
            plt.plot(close_s[i], label = 'true close', c = 'g')
            plt.plot(
                close_s[i], 'X', label = 'predict buy', markevery = list(np.where(inv_f[i] > 0)[0]), c = 'b'
            )
            plt.plot(
                close_s[i], 'o', label = 'predict sell', markevery = list(np.where(inv_f[i] < 0)[0]), c = 'r'
            )
            plt.legend()
            plt.show()

# In[78]:

num_days = 200
close, names = load_data("dataset/train/",num_days)
print(names)

model = Model(input_size = window_size*num_stocks, layer_size = 500, output_size = len(names))
agent = Agent(
    model = model,
    money = 10000,
    limit = 5,
    close = close,
    window_size = window_size,
    num_stocks = len(names),
    num_days = num_days,
    skip = 1,
)


# In[79]:


agent.fit(iterations = 500, checkpoint = 10)



# In[80]:


agent.buy(split="test")
