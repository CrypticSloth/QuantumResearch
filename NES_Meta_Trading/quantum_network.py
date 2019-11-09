import pennylane as qml
from pennylane import numpy as np

# NOTE: 5 wires gives a memory error so 4 seems to be the max
dev = qml.device("strawberryfields.fock", wires=5, cutoff_dim=7)

num_layers = 4
weights = 0.05 * np.random.randn(num_layers, 63)

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

def layer_bs(w):
    '''
        For each weight (w[i]) apply the quantum gates to it.

        Saves the updated weights to the quantum device.

        w: a list of scalar weights (of length 5)
    '''

    qml.Displacement(w[0], 0.0, wires=0)
    qml.Displacement(w[1], 0.0, wires=1)
    qml.Displacement(w[2], 0.0, wires=2)
    qml.Displacement(w[3], 0.0, wires=3)
    qml.Displacement(w[50], 0.0, wires=4)

    qml.Displacement(w[4], w[5], wires=0)
    qml.Displacement(w[6], w[7], wires=1)
    qml.Displacement(w[8], w[9], wires=2)
    qml.Displacement(w[10], w[11], wires=3)
    qml.Displacement(w[51], w[52], wires=4)

    qml.Squeezing(w[12], w[13], wires=0)
    qml.Squeezing(w[14], w[15], wires=1)
    qml.Squeezing(w[16], w[17], wires=2)
    qml.Squeezing(w[18], w[19], wires=3)
    qml.Squeezing(w[53], w[54], wires=4)

    qml.Kerr(w[20], wires=0)
    qml.Kerr(w[21], wires=1)
    qml.Kerr(w[22], wires=2)
    qml.Kerr(w[23], wires=3)
    qml.Kerr(w[55], wires=4)

    qml.Beamsplitter(w[24],0, wires=[0,1])
    qml.Beamsplitter(0,0, wires=[0,1])
    qml.Beamsplitter(w[25],0, wires=[1,2])
    qml.Beamsplitter(0,0, wires=[1,2])
    qml.Beamsplitter(w[26],0, wires=[2,3])
    qml.Beamsplitter(0,0, wires=[2,3])
    qml.Beamsplitter(w[56],0, wires=[3,4])
    qml.Beamsplitter(0,0, wires=[3,4])

    qml.Displacement(w[27], w[28], wires=0)
    qml.Displacement(w[29], w[30], wires=1)
    qml.Displacement(w[31], w[32], wires=2)
    qml.Displacement(w[33], w[34], wires=3)
    qml.Displacement(w[57], w[58], wires=4)

    qml.Squeezing(w[35], w[36], wires=0)
    qml.Squeezing(w[37], w[38], wires=1)
    qml.Squeezing(w[39], w[40], wires=2)
    qml.Squeezing(w[41], w[42], wires=3)
    qml.Squeezing(w[59], w[60], wires=4)

    qml.Kerr(w[43], wires=0)
    qml.Kerr(w[44], wires=1)
    qml.Kerr(w[45], wires=2)
    qml.Kerr(w[46], wires=3)
    qml.Kerr(w[61], wires=4)

    qml.Beamsplitter(w[47],0, wires=[0,1])
    qml.Beamsplitter(0,0, wires=[0,1])
    qml.Beamsplitter(w[48],0, wires=[1,2])
    qml.Beamsplitter(0,0, wires=[1,2])
    qml.Beamsplitter(w[49],0, wires=[2,3])
    qml.Beamsplitter(0,0, wires=[2,3])
    qml.Beamsplitter(w[62],0, wires=[3,4])
    qml.Beamsplitter(0,0, wires=[3,4])

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
    qml.Displacement(x[3], 0.0, wires=3)
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
            qml.expval(qml.X(2)),
            qml.expval(qml.X(3))]
            # qml.expval(qml.X(4))]

def predict(inputs, bs=False):
    '''
        Loop through each of the training data and apply it to the quantum network to get a prediction for each value.

        Will need to somehow make the QNN shape the values to output 5 values for the softmax function. Not sure how to do this since the network only updates with scalar values and the output is the size of the number of inputs.
    '''
    preds = np.array([quantum_neural_net(weights, x=x ,bs=bs) for x in inputs.T])

    return [np.mean(p) for p in preds.T]

# %%

if __name__ == '__main__'
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    import pandas as pd

    import os
    os.chdir("D:/Github/QuantumResearch/NES_Meta_Trading/")

    from updated_NES_google_deterministic import load_data, get_state, softmax

    num_days = 30
    close, names = load_data("dataset/train/",num_days)
    num_stocks = len(names) # This will need to be used to calculate the iterations and input layer sizes along with num_days
    num_stocks
    np.shape(close)

    window_size = 10
    cur_state = get_state(close, 10, window_size + 1, num_days, num_stocks)
    # act(model, np.array([0.,0.,0.,0.]))

    cur_state = cur_state.reshape(num_stocks, window_size)
    np.shape(cur_state)

    dev.reset()
    preds = predict(cur_state, bs=True) # This is all we need..

    softmax([preds])
# softmax(np.array([preds]))

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
