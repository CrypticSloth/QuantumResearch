
# coding: utf-8

# In[2]:

import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import pandas as pd
sns.set()

import os
os.chdir("D:/Github/QuantumResearch/NES_Meta_Trading/")

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
            np.random.randn(layer_size,output_size + 1), # This will have the softmax applied to it...
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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=1) + 0.00001)
#
# def act(model, sequence):
#     decision = model.predict(np.array(sequence))
#     return softmax(decision)
#
# def buy_stock(portfolio, close_s, money, inventory, limit, t):
#     """
#         Function that takes in portfolio weights (percentage of each stock in the entire portfolio),
#         the current stock prices (close price) and the money we currently have
#         and calculates the maximum number of stocks we can buy with the weights given in the portfolio.
#
#         Inventory is the dictionary containing how many stocks we own.
#         Limit puts a maximum number of stock we can purchase
#         t is the current time step
#
#         TODO: instead of dealing with cash amounts we should deal with normalized return (Ri - mean_R) / (std_R)
#     """
#
#     c = 0
#     cash = np.sum([close_s[i][t] * inventory[i] for i in range(len(close_s))]) + money # reset our inventory into cash
#
#     portfolio_money = portfolio[0] * cash # portfolio is an array of an array : [[]]
#
#     p = []
#     for m in portfolio_money:
#         num_stock = math.floor(m / (close_s[c][t] + 0.00001))
#         p.append(close_s[c][t])
#         if num_stock <= limit:
#             inventory[c] = num_stock
#         else:
#             inventory[c] = limit
#
#         cash -= (inventory[c] * close_s[c][t])
#         c += 1
#
#     return inventory, cash
#
# def stock_value(inventory, money, close_s, t):
#     """
#     Calculate current stock value of stock inventory and cash based on timestep t
#     """
#     cash = np.sum([close_s[i][t] * inventory[i] for i in range(len(close_s))]) + money
#     return cash

# Testing one iteration of the new reward function
# This assumes we can purchase partial stocks and has no limits

# # %%
# num_days = 30
# close, names = load_data("dataset/train/",num_days)
# num_stocks = len(names) # This will need to be used to calculate the iterations and input layer sizes along with num_days
# num_stocks
#
# window_size = 10
#
# model = Model(window_size*num_stocks, 500, 3)
# # model = QuantumModel(num_layers=4)
#
# weight = model
# initial_money = 10000
# starting_money = initial_money
#
# cur_state = get_state(close, 0, window_size + 1, num_days, num_stocks)
# close_s = close.reshape(num_stocks,int(len(close)/num_stocks))
# skip = 1
#
# # Initialize a dictionary to keep track of which stocks we can buy
# keys = range(num_stocks)
# cur_inventory = {key: 0 for key in keys}
# limit = 5
#
# split = "train"
# if split == "train":
#     t = close[0:int(len(close)*.7)] # This is a list of list of stock data so this doesnt work
# if split == "test":
#     t = close[int(len(close)*.7):-1]
#
# # close_s
# # close_s[:,0:int(len(close_s[0])*.7)]
# # close_s[:,int(len(close_s[0])*.7):len(close_s[0])]
#
# for t in range(0, len(close_s[0]) - 1, skip):
#
#     portfolio = act(weight, cur_state)
#     next_state = get_state(close, t + 1, window_size + 1,num_days,num_stocks).reshape(num_stocks,window_size)
#
#     next_inventory, initial_money = buy_stock(portfolio, close_s, initial_money, cur_inventory, limit, t)
#
#     cur_state = next_state.flatten()
#     cur_inventory = next_inventory
# ((initial_money - starting_money) / (starting_money + 0.00001)) * 100
# (initial_money / starting_money - 1) * 100
# np.log((initial_money + 0.00001) / (starting_money + 0.00001))

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

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis=1) + 0.00001)

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence)) / 10000 # Unsure how this fixes the problem of always buying one stock... TODO: Investigate this
        # print("Decision : ", decision)
        # print("Softmax : ", self.softmax(decision))

        return self.softmax(decision)

    def buy_stock(self, portfolio, close_s, inventory, limit, t):
        """
            Function that takes in portfolio decision (percentage of each stock in the entire portfolio),
            the current stock prices (close price) and the money we currently have
            and calculates the maximum number of stocks we can buy with the weights given in the portfolio.

            Inventory is the dictionary containing how many stocks we own.
            Limit puts a maximum number of stock we can purchase
            t is the current time step

            portfolio: list -> [[0.1, 0.3, 0.4, 0.0, 0.2]] where the first value is the amount of cash we are holding compared to our previous cash amount.

            TODO: add the cash in as a stock 'option' so the model has full information on how much money is left
                Now, the thing that is holding it back is the limit
                Now we want the 'limit' to not be artificial, but have the algorithm decide its limits via the portfolio of cash amount.
        """

        if limit == None:
            # If no limit is manually set, let the algorithm decide the percentage it will hold as cash asset.
            total_asset_value = np.sum([close_s[i][t] * inventory[i+1] for i in range(len(close_s))]) + inventory[0]

            portfolio_money = portfolio[0] * total_asset_value
            # print(np.sum(portfolio_money))

            c = 0
            for m in portfolio_money[1:]:
                inventory[c+1] = m / close_s[c][t]
                c += 1

            inventory[0] = portfolio_money[0]

            return inventory

        else:

            total_asset_value = np.sum([close_s[i][t] * inventory[i+1] for i in range(len(close_s))]) + inventory[0] # reset our inventory into cash (keeping current cash separate)

            portfolio_money = portfolio[0] * total_asset_value # portfolio[0] because portfolio is an array of array of size 1

            spending_money = total_asset_value - portfolio_money[0]

            c = 0
            for m in portfolio_money[1:]:
                num_stock = math.floor(m / (close_s[c][t] + 0.000001))

                if num_stock <= limit:
                    inventory[c+1] = num_stock
                else:
                    inventory[c+1] = limit

                spending_money -= (inventory[c+1] * close_s[c][t])
                c += 1

            inventory[0] = spending_money  # update the percentage

            return inventory

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
        keys = range(self.num_stocks + 1)  # Plus 1 to add the cash
        cur_inventory = {key: 0 for key in keys}
        cur_inventory[0] = initial_money # Put the cash into the inventory

        cur_state = get_state(close, 0, self.window_size + 1, num_days, self.num_stocks)

        for t in range(0, len(close_s[0]) - 1, self.skip):

            portfolio = self.act(cur_state)
            next_state = get_state(close, t + 1, self.window_size + 1, num_days, self.num_stocks).reshape(self.num_stocks,self.window_size)

            next_inventory = self.buy_stock(portfolio, close_s, cur_inventory, self.limit, t)

            cur_state = next_state.flatten()
            cur_inventory = next_inventory

        if self.limit == None:
            total_asset_value = np.sum([close_s[i][-1] * cur_inventory[i+1] for i in range(len(close_s))]) + cur_inventory[0]
            rho1 = (total_asset_value / starting_money - 1) * 100 # rate of returns
            r1 = np.log((total_asset_value + 0.00001) / (starting_money + 0.00001)) # log rate of return (eq10)
        else:
            total_asset_value = np.sum([close_s[i][-1] * cur_inventory[i+1] for i in range(len(close_s))]) + cur_inventory[0]
            rho1 = (total_asset_value / starting_money - 1) * 100 # rate of returns
            r1 = np.log((total_asset_value + 0.00001) / (starting_money + 0.00001)) # log rate of return (eq10)

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
        keys = range(self.num_stocks + 1)  # Plus 1 to add the cash
        cur_inventory = {key: 0 for key in keys}
        cur_inventory[0] = initial_money # Put the cash into the inventory

        cur_state = get_state(close, 0, self.window_size + 1, num_days, self.num_stocks)

        inv = []

        for t in range(0, len(close_s[0]) - 1, self.skip):

            portfolio = self.act(cur_state)
            next_state = get_state(close, t + 1, self.window_size + 1, num_days, self.num_stocks).reshape(self.num_stocks,self.window_size)

            next_inventory = self.buy_stock(portfolio, close_s, cur_inventory, self.limit, t)

            # record the inventory
            inv_list = []
            for key,value in next_inventory.items():
                inv_list.append(value)
            inv.append(inv_list[1:])

            cur_state = next_state.flatten()
            cur_inventory = next_inventory

        total_asset_value = np.sum([close_s[i][-1] * cur_inventory[i+1] for i in range(len(close_s))]) + cur_inventory[0]
        rho1 = (total_asset_value / starting_money - 1) * 100 # rate of returns

        inv = np.array(inv)
        inv_d = []
        for i in range(len(inv) - 1):
            inv_d.append(inv[i+1] - inv[i])
        inv_d = np.array(inv_d)

        inv_f = []
        for i in range(len(inv_d[0])):
            inv_f.append(np.array(inv_d)[:,i])
        inv_f = np.array(inv_f)

        print("Inventory at every timestep: \n ")
        for i in inv:
            f = ['%.2f' % x for x in i]
            print(str(f))

        print(
            '\ntotal gained %f, total investment %f %%'
            % (total_asset_value - starting_money, rho1)
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

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Train and test portfolio trainer')
    parser.add_argument('--iterations', type=int, help='How many iterations to train the model.')
    parser.add_argument('--checkpoint', type=int, help='How many iterations to print progress to console.')

    args = parser.parse_args()

    window_size = 10
    num_days = 200
    close, names = load_data("dataset/train/",num_days)
    print(names)

    model = Model(input_size = window_size*len(names), layer_size = 500, output_size = len(names))
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

    # agent.fit(iterations = args.iterations, checkpoint = args.checkpoint)
    agent.fit(iterations = 1000, checkpoint = 10)

    # In[80]:

    agent.buy(split="test")
