# Paper Reference:
# https://arxiv.org/pdf/1810.03642.pdf

# Pytorch code reference:
# https://github.com/lmzintgraf/cavia/tree/master/rl

# coding: utf-8

# In[2]:

import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import pandas as pd
import random
sns.set()

import os
os.chdir("C:/Github/QuantumResearch/NES_Meta_Trading/")

# In[58]:

def load_data(path, num_portfolios, num_stocks, num_days = 30):
    '''
    sample num_portfolios from the stocks in path with num_stocks as the number of stocks in each portfolio
    '''

    paths = os.listdir(path)

    portfolios = []
    for i in range(num_portfolios):
        paths = random.sample(paths, num_stocks) # Randomly sample stocks to go in the portfolio
        data = []
        names = []
        for p in paths:
            data.append(pd.read_csv(path + p).Close.values.tolist()[0:num_days])
            if num_portfolios == 1:
                names.append(p[:-4])

        # Check that all data are the same size
        assert len(set([len(d) for d in data])) == 1, "Stock data has differing number of days recorded"

        portfolios.append(np.array([data]).flatten())

    if num_portfolios == 1:
        return np.array(portfolios), names
    else:
        return np.array(portfolios)

def get_state(data, t, n, num_stocks, num_days):
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
num_stocks = 5
num_portfolios = 5
close = load_data("dataset/train/", num_portfolios, num_stocks, num_days)

np.shape(close[0])
len(close)
get_state(close[0], 19, 10, num_stocks, num_days)

close.reshape(num_portfolios, num_stocks, num_days)
# In[63]:


class Deep_Evolution_Strategy:
    def __init__(
        self, theta, context_params, reward_function, population_size, sigma, learning_rate
    ):
        self.theta = theta
        self.context_params = context_params
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

    def train(self, epochs, num_tasks, print_every, split, save_results,path):
        lasttime = time.time()

        if save_results == True:
            results = []

        for e in range(epochs):

            r = []

            cp = []

            # Inner loop
            for i in range(num_tasks):

                population = []
                rewards = np.zeros(self.population_size)
                # Initialization

                # TODO: reset context params.
                self.context_params = np.zeros(len(self.context_params))

                for k in range(self.population_size):
                    x = []
                    for t in self.context_params:
                        x.append(np.random.randn(*t.shape))
                    population.append(x)

                for k in range(self.population_size):
                    weights_population = self._get_weight_from_population(
                        self.context_params, population[k]
                    )
                    rewards[k] = self.reward_function(self.theta, weights_population, loop='inner', index = i, split=split)
                rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 0.00001) # Normalize the rewards here

                # t = []
                for index, t in enumerate(self.context_params):
                    A = np.array([p[index] for p in population])
                    gradient = np.dot(A.T, rewards).T / (self.population_size * self.sigma) + 0.00001 # This is the shape of the NN layer
                    self.context_params[index] = (
                        t
                        + self.learning_rate
                        * gradient
                    )

                    # print(np.shape(gradient))
                    # t.append(self.theta[index] + self.learning_rate * gradient)

                # rollout test on new data? goes here...

                # Save the context params for updating outer theta
                cp.append(self.context_params)

                # Evaluate the upated context paramters by taking new actions with them and save the results
                r.append(self.reward_function(self.theta, self.context_params, loop='inner', index=i, split=split, return_reward=True))


            # Outer theta update

            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for t in self.theta:
                    x.append(np.random.randn(*t.shape))
                population.append(x)

            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(
                    self.theta, population[k]
                )
                # compute the rewards with the mean of the context params computed above
                rewards[k] = self.reward_function(weights_population, np.mean(cp), loop='outer', index = i, split=split)
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 0.00001)

            for index, t in enumerate(self.theta):
                A = np.array([p[index] for p in population])
                gradient = np.dot(A.T, rewards).T / (self.population_size * self.sigma + 0.00001) # This is the shape of the NN layer
                self.theta[index] = (
                    t
                    + self.learning_rate
                    * gradient
                )

            if (e + 1) % print_every == 0:
                print(
                    'Epoch {}, reward: {}'.format(e+1,np.mean(r))
                )

            if save_results == True:
                results.append(np.mean(r))

        print('-----------------')

        if save_results == True:
            df = pd.DataFrame()
            df['epochs'] = [i for i in range(0,epochs)]
            df['rewards'] = results

            if not os.path.exists(path):
                os.mkdir(path)
                print("Directory " , path ,  " Created ")

            ts = time.time()
            df.to_csv(path + '/{}_train_rewards.csv'.format(int(ts)), index=False)


# In[64]:

class Model:
    def __init__(self, input_size, layer_size, output_size, num_context_params):
        self.context_params = np.zeros(num_context_params)

        # self.weights = [
        #     np.random.randn(input_size + num_context_params, layer_size),
        #     np.random.randn(layer_size, layer_size),
        #     np.random.randn(layer_size,output_size + 1),
        #     np.random.randn(1, layer_size),
        # ]

        self.theta = [
            np.random.randn(input_size + num_context_params, layer_size),
            np.random.randn(layer_size, layer_size),
            np.random.randn(layer_size,output_size + 1),
            np.random.randn(1, layer_size),
        ]


    # To make this deterministic, out ouput is going to be weights for each stock we have
    # For eg, if we have 5 stocks out output will be a column of (1,5) where each row is
    # a percentage of how much of that stock we want to have in our portfolio
    def predict(self, inputs):
        inputs = np.concatenate((inputs[0],self.context_params)) # Concatenate the context params to the inputs
        feed = np.dot(inputs, self.theta[0]) + self.theta[-1]
        decision = np.dot(feed, self.theta[1])
        decision = np.dot(decision, self.theta[2])
        # portfolio = softmax(decision)
        # buy = [0.75]
        # buy = np.dot(feed, self.weights[2])
        # return decision, buy
        return decision

    def set_context_params(self, context_params):
        self.context_params = context_params

    def get_context_params(self):
        return self.context_params

    # def get_weights(self):
    #     return self.weights
    #
    # def set_weights(self, weights):
    #     self.weights = weights

    def get_theta(self):
        return self.theta

    def set_theta(self, theta):
        self.theta = theta

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=1) + 0.00001)

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
#
# #Testing one iteration of the new reward function
# #This assumes we can purchase partial stocks and has no limits
#
# # %%
# window_size = 10
# num_days = 30
# num_stocks = 5
# num_portfolios = 5
# close = load_data("dataset/train/",num_portfolios, num_stocks, num_days)
#
# model = Model(window_size*num_stocks, 500, 3, 5)
#
# weight = model
# initial_money = 10000
# starting_money = initial_money
#
# cur_state = get_state(close[0], 0, window_size + 1, num_stocks, num_days)
# close_s = close[0].reshape(num_stocks,int(len(close[0])/num_stocks))
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
# cur_state
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
        self, model, money, limit, close, window_size, skip, num_portfolios, num_stocks, num_days, split
    ):
        self.window_size = window_size
        self.num_portfolios = num_portfolios
        self.num_stocks = num_stocks
        self.num_days = num_days
        self.skip = skip
        self.close = close
        self.model = model
        self.initial_money = money
        self.limit = limit
        self.split = split
        self.es = Deep_Evolution_Strategy(
            self.model.get_theta(),
            self.model.get_context_params(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def get_path(self,epochs):
        dir_name = 'E={}_PS={}_S={}_LR={}_sk={}_IM={}_L={}_WS={}_ND={}/'.format(
            epochs,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
            self.skip,
            # self.beta,
            self.initial_money,
            self.limit,
            self.window_size,
            self.num_days
        )
        # This will need to be set per computer
        return 'C:/GitHub/QuantumResearch/NES_Meta_Trading/results/cavia/' + self.split + '/' + dir_name

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis=1) + 0.00001)

    def act(self, sequence):
        decision = self.model.predict(np.array(sequence)) / 1000 # Unsure how this fixes the problem of always buying one stock... TODO: Investigate this

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

            TODO: add limits that include the cash bias to number of stocks we can purchase (not partial stocks)
        """

        if limit == None:
            # If no limit is manually set, let the algorithm decide the percentage it will hold as cash asset.
            total_asset_value = np.sum([close_s[i][t] * inventory[i+1] for i in range(self.num_stocks)]) + inventory[0]

            portfolio_money = portfolio[0] * total_asset_value
            # print(np.sum(portfolio_money))

            c = 0
            for m in portfolio_money[1:]:
                inventory[c+1] = m / close_s[c][t]
                c += 1

            inventory[0] = portfolio_money[0]

            return inventory

        else:

            total_asset_value = np.sum([close_s[i][t] * inventory[i+1] for i in range(self.num_stocks)]) + inventory[0] # reset our inventory into cash (keeping current cash separate)

            portfolio_money = portfolio[0] * total_asset_value # portfolio[0] because portfolio is an array of array of size 1

            spending_money = total_asset_value #- portfolio_money[0]

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

    def get_reward(self, theta, context_params, loop, index, return_reward=False, split = "train", ):
        '''
            Reward function.

            Model after the reward found here: https://github.com/wassname/rl-portfolio-management/blob/master/rl_portfolio_management/environments/portfolio.py
            In the paper the variables are:
                p1 = initial_money
                p0 = starting_money

            We could add cost of trading stocks as well to this in the future.
        '''

        if loop == "inner":
            self.model.context_params = context_params
        if loop == "outer":
            self.model.theta = theta

        # weight = model
        initial_money = self.initial_money
        starting_money = initial_money
        close_s = self.close.reshape(
            self.num_portfolios,
            self.num_stocks,
            self.num_days
            )

        # Split the data into either train or test dataset
        if split == "train":
            close_s = close_s[index][:,0:round(self.num_days*.7)]
            num_days = round(self.num_days*0.7)
        if split == "test":
            close_s = close_s[index][:,round(self.num_days*.7):self.num_days]
            num_days = round(self.num_days*0.3)
        if split == None:
            close_s = close_s[index]
            num_days = self.num_days

        close = close_s.flatten() # Use the split data for close

        # Initialize a dictionary to keep track of which stocks we can buy
        keys = range(self.num_stocks + 1)  # Plus 1 to add the cash
        cur_inventory = {key: 0 for key in keys}
        cur_inventory[0] = initial_money # Put the cash into the inventory

        cur_state = get_state(close, 0, self.window_size + 1, self.num_stocks,  num_days)

        for t in range(0, num_days - 1, self.skip):

            portfolio = self.act(cur_state)
            next_state = get_state(close, t + 1, self.window_size + 1, self.num_stocks, num_days)

            next_inventory = self.buy_stock(portfolio, close_s, cur_inventory, self.limit, t)

            cur_state = next_state
            cur_inventory = next_inventory

        if self.limit == None:
            total_asset_value = np.sum([close_s[i][-1] * cur_inventory[i+1] for i in range(self.num_stocks)]) + cur_inventory[0]
            rho1 = (total_asset_value / starting_money - 1) * 100 # rate of returns
            r1 = np.log((total_asset_value + 0.00001) / (starting_money + 0.00001)) # log rate of return (eq10)
        else:
            total_asset_value = np.sum([close_s[i][-1] * cur_inventory[i+1] for i in range(self.num_stocks)]) + cur_inventory[0]
            rho1 = (total_asset_value / starting_money - 1) * 100 # rate of returns
            r1 = np.log((total_asset_value + 0.00001) / (starting_money + 0.00001)) # log rate of return (eq10)

        if return_reward == True:
            return rho1
        else:
            return r1


    def fit(self, epochs, num_tasks, checkpoint, split, save_results):
        self.es.train(epochs, num_tasks, print_every = checkpoint, split = split, save_results = save_results, path = self.get_path(epochs))

    def buy(self, split, names, save_results, epochs):

        # can only test on one portfolio, which is sufficient to test meta learning
        # for each stock in the portfolio...
        for i in range(len(self.close)):

            # Can only test on data size of one portfolio
            # if split == "train":
            #     close = self.close[i][0:round(self.num_days*.7)]
            #     num_days = round(self.num_days*0.7)
            # if split == "test":
            #     close = self.close[i][round(self.num_days*.7):-1]
            #     num_days = round(self.num_days*0.3)

            initial_money = self.initial_money
            starting_money = initial_money
            close_s = self.close[i].reshape(
                # self.num_portfolios,
                self.num_stocks,
                self.num_days)

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

            cur_state = get_state(close, 0, self.window_size + 1, self.num_stocks, num_days)

            inv = []

            total_values = []
            total_returns = []
            portfolio_values = []
            for t in range(0, len(close_s[0]) - 1, self.skip):

                portfolio = self.act(cur_state)
                next_state = get_state(close, t + 1, self.window_size + 1, self.num_stocks, num_days)

                next_inventory = self.buy_stock(portfolio, close_s, cur_inventory, self.limit, t)

                # record the inventory
                inv_list = []
                for key,value in next_inventory.items():
                    inv_list.append(value)
                inv.append(inv_list[1:])

                cur_state = next_state
                cur_inventory = next_inventory

                if save_results == True:
                    total_asset_value = np.sum([close_s[i][t] * cur_inventory[i+1] for i in range(self.num_stocks)]) + cur_inventory[0]

                    total_values.append(total_asset_value)
                    total_returns.append(((total_asset_value - starting_money) / starting_money + 0.00001) * 100)
                    portfolio_values.append(portfolio[0])

            total_asset_value = np.sum([close_s[i][-1] * cur_inventory[i+1] for i in range(self.num_stocks)]) + cur_inventory[0]
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

            if save_results != True:
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

            if save_results == True:
                path = self.get_path(epochs)

                df = pd.DataFrame()
                df['roi'] = total_returns
                df['returns'] = total_values

                ts = time.time()
                df.to_csv(path + '/{}_test_reward.csv'.format(int(ts)), index=True)

                df = pd.DataFrame(np.array(portfolio_values))
                df.to_csv(path + '/{}_portfolio_reward.csv'.format(int(ts)), index=True)

    def save(self, epochs):
        ''' save the results of the agent to disk '''
        # User input path = 'results/(train/test)/'

        ts = int(time.time())
        np.save(self.get_path(epochs) + '{}_model_weights.npy'.format(ts),model.get_theta())

# In[78]:

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Train and test portfolio trainer')
    parser.add_argument('--iterations', type=int, help='How many iterations to train the model.')
    parser.add_argument('--checkpointTrain', type=int, help='How many iterations to print progress to console.')
    parser.add_argument('--checkpointTest', type=int, help='How many iterations to print progress to console.')
    parser.add_argument('--epochsTrain', type=int, help='Epochs to train the NES algorithm')
    parser.add_argument('--epochsTest', type=int, help='Epochs to train the meta algorithm')
    parser.add_argument('--limit', help='Maximum algorithm can trade of one stock')
    parser.add_argument('--num_days',type=int, default=360, help='Number of days to trade on')
    parser.add_argument('--num_stocks',type=int, default=5, help='Number of stocks to trade on')
    parser.add_argument('--num_portfolios', type=int, default=5, help='Number of portfolios to trade on when training')
    args = parser.parse_args()

    if args.limit == "None":
        args.limit = None
    else:
         args.limit = int(args.limit)

    for i in range(args.iterations):
        window_size = 10
        num_days = args.num_days
        num_stocks = args.num_stocks
        num_portfolios = args.num_portfolios

        close = load_data("dataset/train/",num_portfolios, num_stocks, num_days)
        np.shape(close)
        close_s = close.reshape(num_portfolios,num_stocks,num_days)
        len(close_s[0])

        model = Model(input_size = window_size*num_stocks, layer_size = 500, output_size = num_stocks, num_context_params = 5)
        agent = Agent(
            model = model,
            money = 10000,
            limit = args.limit,
            close = close,
            window_size = window_size,
            num_portfolios = num_portfolios,
            num_stocks = num_stocks,
            num_days = num_days,
            skip = 1,
            split = "train"
        )


        # In[79]:

        # Training the meta
        # agent.fit(iterations = args.iterations, checkpoint = args.checkpoint)
        # epochs = 5000
        agent.fit(epochs = args.epochsTrain, num_tasks = num_portfolios, checkpoint = args.checkpointTrain, split="train", save_results = True)
        agent.save(epochs=args.epochsTrain)

        # In[80]:
        # Training the trained meta on one stock with fewer epochs
        testModel = Model(input_size = window_size*num_stocks, layer_size = 500, output_size = num_stocks, num_context_params = 5)
        testModel.set_theta = model.get_theta

        num_portfolios = 1 # Number of portfolios must be 1 when testing
        data, names = load_data("dataset/test/", num_portfolios, num_stocks, num_days)

        # close_s = data.reshape(num_portfolios,num_stocks,num_days)
        # close_s[0]

        agent = Agent(
            model = model,
            money = 10000,
            limit = args.limit,
            close = data,
            window_size = window_size,
            num_portfolios = num_portfolios,
            num_stocks = num_stocks,
            num_days = num_days,
            skip = 1,
            split = "test"
        )

        # Train with a few epochs to test the meta learning
        # epochs = 20
        agent.fit(epochs = args.epochsTest, num_tasks = num_portfolios, checkpoint = args.checkpointTest, split="train", save_results = True)
        agent.save(args.epochsTest)

        agent.buy(split="test", names=names, save_results=True, epochs=args.epochsTest)
