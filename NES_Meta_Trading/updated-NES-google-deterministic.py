
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()

import os
os.chdir("C:/Github/QuantumResearch/NES_Meta_Trading/")

# In[21]:


import pandas as pd
google = pd.read_csv('dataset/train/GOOG.csv')
google.head()


# In[58]:

len(google.Close.values.tolist()[0:30])

def load_data(num_days = 30):
    '''
    load in all stock data from the path and return the close values
    '''

    google = pd.read_csv('dataset/train/GOOG-year.csv')
    amd = pd.read_csv('dataset/train/AMD.csv')
    fb = pd.read_csv('dataset/train/FB.csv')

    np.shape(google)
    np.shape(amd)
    np.shape(fb)

    google_c = google.Close.values.tolist()[0:num_days]
    amd_c = amd.Close.values.tolist()[0:num_days]
    fb_c = fb.Close.values.tolist()[0:num_days]

    return np.array([google_c,amd_c,fb_c]).flatten()

def get_state(data, t, n, num_days = 30,num_stocks = 3):
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
num_stocks = 3 # This will need to be used to calculate the iterations and input layer sizes along with num_days
close = load_data(num_days)
np.shape(close)
len(close)
get_state(close, 19, 10, num_days)


# In[61]:


get_state(close, 1, 10, num_days)



# In[62]:

get_state(close, 2, 10, num_days)
np.shape(get_state(close, 2, 10, num_days))


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
                rewards[k] = self.reward_function(weights_population)
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
                    % (i + 1, self.reward_function(self.weights, return_reward = True))
                )
        print('time taken to train:', time.time() - lasttime, 'seconds')


# In[64]:

class Model:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
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
        # portfolio = softmax(decision)
        # buy = [0.75]
        # buy = np.dot(feed, self.weights[2])
        # return decision, buy
        return decision

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

# In[66]:

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
num_stocks = 3 # This will need to be used to calculate num of iterations as well as input layer size with window_size
window_size = 9
model = Model(window_size*num_stocks, 500, 3)

cur_state = get_state(close, 0, window_size + 1, num_days)
weight = model
initial_money = 10000
starting_money = initial_money

# cur_state = get_state(close, 0, window_size + 1).reshape(num_stocks,window_size)
close_s = close.reshape(num_stocks,int(len(close)/num_stocks))
skip = 1

# Initialize a dictionary to keep track of which stocks we can buy
keys = range(num_stocks)
cur_inventory = {key: 0 for key in keys}
limit = 5

for t in range(0, len(close_s[0]) - 1, skip):

    portfolio = act(weight, cur_state)
    next_state = get_state(close, t + 1, window_size + 1).reshape(num_stocks,window_size)

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
        self, model, money, max_buy, max_sell, limit, close, window_size, skip, num_stocks
    ):
        self.window_size = window_size
        self.num_stocks = num_stocks
        self.skip = skip
        self.close = close
        self.model = model
        self.initial_money = money
        self.max_buy = max_buy
        self.max_sell = max_sell
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
        state = get_state(self.close, 0, self.window_size + 1)
        inventory = []
        quantity = 0
        for t in range(0, len_close, self.skip):
            action, buy = self.act(state)
            next_state = get_state(self.close, t + 1, self.window_size + 1)
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

    def get_reward(self, weights, return_reward=False):
        '''
            Reward function.

            Model after the reward found here: https://github.com/wassname/rl-portfolio-management/blob/master/rl_portfolio_management/environments/portfolio.py
            In the paper the variables are:
                p1 = initial_money
                p0 = starting_money

            We could add cost as well to this in the future.
        '''

        self.model.weights = weights

        cur_state = get_state(self.close, 0, self.window_size + 1)
        # weight = model
        initial_money = self.initial_money
        starting_money = initial_money

        # cur_state = get_state(close, 0, window_size + 1).reshape(num_stocks,window_size)
        close_s = self.close.reshape(num_stocks,int(len(close)/num_stocks))

        # Initialize a dictionary to keep track of which stocks we can buy
        keys = range(self.num_stocks)
        cur_inventory = {key: 0 for key in keys}

        for t in range(0, len(close_s[0]) - 1, self.skip):

            portfolio = self.act(cur_state)
            next_state = get_state(self.close, t + 1, self.window_size + 1).reshape(self.num_stocks,self.window_size)

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

    def buy(self):
        initial_money = self.initial_money
        len_close = len(self.close) - 1
        state = get_state(self.close, 0, self.window_size + 1)
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        quantity = 0
        for t in range(0, len_close, self.skip):
            action, buy = self.act(state)
            next_state = get_state(self.close, t + 1, self.window_size + 1)
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
                states_buy.append(t)
                print(
                    'day %d: buy %d units at price %f, total balance %f'
                    % (t, buy_units, total_buy, initial_money)
                )
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                else:
                    sell_units = quantity
                if sell_units < 1:
                    continue
                quantity -= sell_units
                total_sell = sell_units * self.close[t]
                initial_money += total_sell
                states_sell.append(t)
                try:
                    invest = ((total_sell - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
                    % (t, sell_units, total_sell, invest, initial_money)
                )
            state = next_state

        invest = ((initial_money - starting_money) / (starting_money + 0.00001)) * 100
        print(
            '\ntotal gained %f, total investment %f %%'
            % (initial_money - starting_money, invest)
        )
        plt.figure(figsize = (20, 10))
        plt.plot(close, label = 'true close', c = 'g')
        plt.plot(
            close, 'X', label = 'predict buy', markevery = states_buy, c = 'b'
        )
        plt.plot(
            close, 'o', label = 'predict sell', markevery = states_sell, c = 'r'
        )
        plt.legend()
        plt.show()


# In[78]:


model = Model(input_size = window_size*num_stocks, layer_size = 500, output_size = num_stocks)
agent = Agent(
    model = model,
    money = 10000,
    max_buy = 5,
    max_sell = 5,
    limit = 5,
    close = close,
    window_size = window_size,
    num_stocks = num_stocks,
    skip = 1,
)


# In[79]:


agent.fit(iterations = 250, checkpoint = 10)



# In[80]:


agent.buy()
