
import pennylane as qml
from pennylane import numpy as np

# NOTE: 5 wires gives a memory error so 4 seems to be the max
dev = qml.device("strawberryfields.fock", wires=3, cutoff_dim=7)

# num_layers = 4
# weights = 0.05 * np.random.randn(num_layers, 63)

# %%
def layer(w):
    '''
        For each weight (w[i]) apply the quantum gates to it.

        Saves the updated weights to the quantum device.

        w: a list of scalar weights (of length 5)
    '''

    # Matrix multiplication of input layer
    qml.Rotation(w[0], wires=0)
    qml.Rotation(w[1], wires=1)
    # qml.Rotation(w[2], wires=2)
    # qml.Rotation(w[3], wires=3)
    # qml.Rotation(w[4], wires=4)

    qml.Squeezing(w[2], 0.0, wires=0)
    qml.Squeezing(w[3], 0.0, wires=1)
    # qml.Squeezing(w[7], 0.0, wires=2)
    # qml.Squeezing(w[8], 0.0, wires=3)
    # qml.Squeezing(w[9], 0.0, wires=4)

    qml.Rotation(w[4], wires=0)
    qml.Rotation(w[5], wires=1)
    # qml.Rotation(w[12], wires=2)
    # qml.Rotation(w[13], wires=3)
    # qml.Rotation(w[14], wires=4)

    # Bias
    qml.Displacement(w[6], 0.0, wires=0)
    qml.Displacement(w[7], 0.0, wires=1)
    # qml.Displacement(w[17], 0.0, wires=2)
    # qml.Displacement(w[18], 0.0, wires=3)
    # qml.Displacement(w[19], 0.0, wires=4)

    # Element-wise nonlinear transformation
    qml.Kerr(w[8], wires=0)
    qml.Kerr(w[9], wires=1)
    # qml.Kerr(w[22], wires=2)
    # qml.Kerr(w[23], wires=3)
    # qml.Kerr(w[24], wires=4)

def layer_bs(w):
    '''
        For each weight (w[i]) apply the quantum gates to it.

        Saves the updated weights to the quantum device.

        w: a list of scalar weights (of length 5)
    '''

    qml.Displacement(w[0], 0.0, wires=0)
    qml.Displacement(w[1], 0.0, wires=1)
    qml.Displacement(w[2], 0.0, wires=2)
    # qml.Displacement(w[3], 0.0, wires=3)
    # qml.Displacement(w[50], 0.0, wires=4)

    qml.Displacement(w[4], w[5], wires=0)
    qml.Displacement(w[6], w[7], wires=1)
    qml.Displacement(w[8], w[9], wires=2)
    # qml.Displacement(w[10], w[11], wires=3)
    # qml.Displacement(w[51], w[52], wires=4)

    qml.Squeezing(w[12], w[13], wires=0)
    qml.Squeezing(w[14], w[15], wires=1)
    qml.Squeezing(w[16], w[17], wires=2)
    # qml.Squeezing(w[18], w[19], wires=3)
    # qml.Squeezing(w[53], w[54], wires=4)

    qml.Kerr(w[20], wires=0)
    qml.Kerr(w[21], wires=1)
    qml.Kerr(w[22], wires=2)
    # qml.Kerr(w[23], wires=3)
    # qml.Kerr(w[55], wires=4)

    qml.Beamsplitter(w[24],0, wires=[0,1])
    qml.Beamsplitter(0,0, wires=[0,1])
    qml.Beamsplitter(w[25],0, wires=[1,2])
    qml.Beamsplitter(0,0, wires=[1,2])
    # qml.Beamsplitter(w[26],0, wires=[2,3])
    # qml.Beamsplitter(0,0, wires=[2,3])
    # qml.Beamsplitter(w[56],0, wires=[3,4])
    # qml.Beamsplitter(0,0, wires=[3,4])

    qml.Displacement(w[27], w[28], wires=0)
    qml.Displacement(w[29], w[30], wires=1)
    qml.Displacement(w[31], w[32], wires=2)
    # qml.Displacement(w[33], w[34], wires=3)
    # qml.Displacement(w[57], w[58], wires=4)

    qml.Squeezing(w[35], w[36], wires=0)
    qml.Squeezing(w[37], w[38], wires=1)
    qml.Squeezing(w[39], w[40], wires=2)
    # qml.Squeezing(w[41], w[42], wires=3)
    # qml.Squeezing(w[59], w[60], wires=4)

    qml.Kerr(w[43], wires=0)
    qml.Kerr(w[44], wires=1)
    qml.Kerr(w[45], wires=2)
    # qml.Kerr(w[46], wires=3)
    # qml.Kerr(w[61], wires=4)

    qml.Beamsplitter(w[47],0, wires=[0,1])
    qml.Beamsplitter(0,0, wires=[0,1])
    qml.Beamsplitter(w[48],0, wires=[1,2])
    qml.Beamsplitter(0,0, wires=[1,2])
    # qml.Beamsplitter(w[49],0, wires=[2,3])
    # qml.Beamsplitter(0,0, wires=[2,3])
    # qml.Beamsplitter(w[62],0, wires=[3,4])
    # qml.Beamsplitter(0,0, wires=[3,4])

@qml.qnode(dev)
def quantum_neural_net(weights, x=None, bs=False):
    '''
        For each layer, apply the inputs to the gates to update the weights

        weights: list of lists of scalar weights (of length 5)
        x: list of stock closing values for 5 stocks
    '''

    # Encode input x into quantum state
    qml.Displacement(x[0], 0.0, wires=0)
    qml.Displacement(x[1], 0.0, wires=1)
    qml.Displacement(x[2], 0.0, wires=2)
    # qml.Displacement(x[3], 0.0, wires=3)
    # qml.Displacement(x[4], 0.0, wires=4)

    # "layer" subcircuits
    if bs == True:
        for w in weights:
            layer_bs(w)
    else:
        for w in weights:
            layer(w)

    return [qml.expval(qml.X(0)),
            qml.expval(qml.X(1)),
            qml.expval(qml.X(2))]
            # qml.expval(qml.X(3)),
            # qml.expval(qml.X(4))]

def predict(inputs, weights, bs=False):
    '''
        Loop through each of the training data and apply it to the quantum network to get a prediction for each value.

        Will need to somehow make the QNN shape the values to output 5 values for the softmax function. Not sure how to do this since the network only updates with scalar values and the output is the size of the number of inputs.
    '''

    preds = [quantum_neural_net(weights, x=x, bs=bs)
            for y in np.array(inputs).T
            for x in [y]]
    # preds = np.array([quantum_neural_net(weights, x=x ,bs=bs) for x in inputs.T])
    # print("P: ", preds)
    # print("Ps: ", [np.sum(p) for p in np.array(preds).T] )

    return [np.sum(p) for p in np.array(preds).T] # I feel that this could be wrong

# %%
# Testing code


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
#         TODO: Getting negetive inventory values from negetive cash
#         TODO: instead of dealing with cash amounts we should deal with percentage gain (normalized)?
#     """
#
#     c = 0
#     cash = np.sum([close_s[i][t] * inventory[i] for i in range(len(close_s))]) + money # reset our inventory into cash
#
#     portfolio_money = portfolio[0] * cash
#
#     p = []
#     for m in portfolio_money:
#         num_stock = math.floor(m / (close_s[c][t] + 0.000001))
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
# import os
# import numpy as np
# import math
# os.chdir("C:/Github/QuantumResearch/NES_Meta_Trading/")
#
# from updated_NES_google_deterministic import load_data, get_state
# import warnings
# warnings.filterwarnings('ignore')
#
# def test():
#     num_days = 30
#     close, names = load_data("dataset/train_q/",num_days)
#     num_stocks = len(names) # This will need to be used to calculate the iterations and input layer sizes along with num_days
#     num_stocks
#     np.shape(close)
#
#     def softmax(x):
#         """Compute softmax values for each sets of scores in x."""
#         e_x = np.exp(x - np.max(x))
#         return e_x / (e_x.sum(axis=1) + 0.00001)
#
#     def act(sequence, weights):
#         decision = predict(np.array(sequence).reshape(num_stocks,window_size), weights)
#         # print(softmax(np.array([decision]) * 1))
#         # print(decision)
#         # print(self.softmax([decision]) * 100)
#         return softmax(np.array([decision]) * 10)
#
#     num_layers = 4
#     weights = 0.05 * np.random.randn(num_layers, 10)
#
#     initial_money = 10000
#     window_size = 1
#     limit = 5
#     starting_money = initial_money
#     close_s = close.reshape(num_stocks,int(len(close)/num_stocks))
#
#     close = close_s.flatten() # Use the split data for close
#     np.shape(close)
#
#     # Initialize a dictionary to keep track of which stocks we can buy
#     keys = range(num_stocks)
#     cur_inventory = {key: 0 for key in keys}
#
#
#     cur_state = get_state(close, 0, window_size + 1, num_days, num_stocks)
#     np.shape(cur_state)
#     for t in range(0, len(close_s[0]) - 1):
#
#         portfolio = act(cur_state, weights)
#         print(portfolio)
#         next_state = get_state(close, t + 1, window_size + 1, num_days, num_stocks)
#         next_inventory, initial_money = buy_stock(portfolio, close_s, initial_money, cur_inventory, limit, t)
#
#         cur_state = next_state
#         print(next_inventory)
#         cur_inventory = next_inventory
#
#     return cur_inventory, (initial_money / starting_money - 1) * 100 # rate of returns
#
# test()
# test()

# close
# cur_state = get_state(close, 10, window_size + 1, num_days, num_stocks)
# cur_state
# # cur_state.reshape(num_stocks, window_size)
# bs = False
# preds = [quantum_neural_net(weights, x=x, bs=bs)
#         for y in np.array(cur_state).reshape(num_stocks, window_size).T
#         for x in [y]]
# [x
# for y in np.array(cur_state).reshape(num_stocks, window_size).T
# for x in [y]]
#
# cur_state.reshape(num_stocks, window_size)
# np.array(preds).T
#
# [np.sum(p) for p in np.array(preds).T]
#
# port = act(cur_state, weights)
# port
#
# test()
#
# x_pred = np.array([np.linspace(-1,1,10), np.linspace(-1,1,10)])
# predictions = [quantum_neural_net(weights, x=x_) for x_ in x_pred.T]
# predictions
#
# t1 = []
# t2 = []
# for i in range(30):
#     inv, rew = test()
#     t2.append(rew)
#     t1.append(inv)
# t1
# t2
#
# test()
# test()


# %%

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

    def train(self, epochs, print_every, save_results, path):
        lasttime = time.time()
        results = []
        for i in range(epochs):
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
            if save_results == True:
                results.append(self.reward_function(self.weights, return_reward=True, split="train"))
        print('time taken to train:', time.time() - lasttime, 'seconds')

        print('-----------------')

        if save_results == True:
            df = pd.DataFrame()
            df['epochs'] = [i for i in range(0,epochs)]
            df['rewards'] = results

            if not os.path.exists(path):
                os.mkdir(path)
                print("Directory " , path ,  " Created ")

            ts = time.time()
            df.to_csv(path + '{}_train_rewards.csv'.format(int(ts)), index=False)
# %%

import time


class Agent:

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    # global weights_g
    # weights_l = weights_g

    def __init__(
        self, money, limit, close, window_size, skip, num_stocks, num_days, weights, bs
    ):
        self.window_size = window_size
        self.num_stocks = num_stocks
        self.num_days = num_days
        self.skip = skip
        self.close = close
        self.initial_money = money
        self.limit = limit
        self.weights = weights
        self.bs = bs
        self.es = Deep_Evolution_Strategy(
            self.weights,
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
        return 'C:/GitHub/QuantumResearch/NES_Meta_Trading/results/quantum/' + dir_name

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis=1) + 0.00001)

    def act(self, sequence):
        decision = predict(np.array(sequence).reshape(self.num_stocks,self.window_size), self.weights, bs=self.bs)
        # print(decision)
        # print(self.softmax([decision]) * 100)
        return self.softmax(np.array([decision]) * 10)

    def buy_stock(self, portfolio, close_s, money, inventory, limit, t):
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

        # This line is probably where the problem lies.


        # global weights_g
        # print("WG1: ", weights_g)
        # print("W: ", weights)
        self.weights = weights # This needs to update the weights that act() sees...
        # print("WG2: ", weights_g)
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

            next_inventory, initial_money = self.buy_stock(portfolio, close_s, initial_money, cur_inventory, self.limit, t)

            cur_state = next_state.flatten()
            cur_inventory = next_inventory

        rho1 = (initial_money / starting_money - 1) * 100 # rate of returns
        r1 = np.log((initial_money + 0.00001) / (starting_money + 0.00001)) # log rate of return (eq10)

        if return_reward == True:
            return rho1
        else:
            return r1


    def fit(self, epochs, checkpoint, save_results):
        self.es.train(epochs, print_every = checkpoint, save_results=save_results, path=self.get_path(epochs))

    def buy(self, split, names, save_results, epochs):

        # weight = self.model
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

        total_values = []
        total_returns = []
        for t in range(0, len(close_s[0]) - 1, self.skip):

            portfolio = self.act(cur_state)
            next_state = get_state(close, t + 1, self.window_size + 1, num_days, self.num_stocks).reshape(self.num_stocks,self.window_size)

            next_inventory, initial_money = self.buy_stock(portfolio, close_s, initial_money, cur_inventory, self.limit, t)

            # record the inventory
            inv_list = []
            for key,value in next_inventory.items():
                inv_list.append(value)
            inv.append(inv_list)

            cur_state = next_state.flatten()
            cur_inventory = next_inventory

            if save_results == True:

                total_values.append(initial_money) #TODO: this might have to be looked at.
                total_returns.append((initial_money / starting_money - 1) * 100)

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

        if save_results == False:
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

        if save_results == True:
            path = self.get_path(epochs)

            df = pd.DataFrame()
            df['roi'] = total_returns
            df['returns'] = total_values

            ts = time.time()
            df.to_csv(path + '{}_test_reward.csv'.format(int(ts)), index=True)

    def save(self, epochs):
        ''' save the results of the agent to disk '''
        # User input path = 'results/(train/test)/'

        ts = int(time.time())
        np.save(self.get_path(epochs) + '{}_model_weights.npy'.format(ts),self.weights)
# %%

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    import os
    os.chdir("C:/Github/QuantumResearch/NES_Meta_Trading/")

    from updated_NES_google_deterministic import load_data, get_state

    for i in range(5):
        num_days = 60
        close, names = load_data("dataset/train_q/",num_days)
        num_stocks = len(names) # This will need to be used to calculate the iterations and input layer sizes along with num_days
        num_stocks
        np.shape(close)

        window_size = 1
        cur_state = get_state(close, 10, window_size + 1, num_days, num_stocks)

        # %%

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / (e_x.sum() + 0.00001)

        num_layers = 4
        weights = 0.05 * np.random.randn(num_layers, 65)
        # weights = 0.05 * np.random.randn(num_layers, 10)
        # print(weights)


        # model = Model(input_size = window_size*num_stocks, layer_size = 500, output_size = len(names))
        agent = Agent(
            money = 10000,
            limit = 10,
            close = close,
            window_size = window_size,
            num_stocks = len(names),
            num_days = num_days,
            skip = 1,
            weights = weights,
            bs = True
        )


        # In[79]:

        epochs = 1000
        agent.fit(epochs = epochs, checkpoint = 1, save_results=True)
        agent.save(epochs)


    # Weights are changing...
    # print(agent.weights)
      # In[80]:

        agent.buy(split="test", names=names, save_results=True, epochs=epochs)

# In[66]: ######################################################################

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
