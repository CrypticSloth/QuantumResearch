import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import matplotlib.pyplot as plt

import os
os.chdir("C:/Github/QuantumResearch/NES_Meta_Trading")

# NOTE: 5 wires gives a memory error so 4 seems to be the max
num_wires = 1
bs = False
dev = qml.device("strawberryfields.fock", wires=num_wires, cutoff_dim=7)

# %%

num_layers = 4

if bs == False:
    w = 0.05 * np.random.randn(num_layers, num_wires*5)
if bs == True:
    w = 0.05 * np.random.randn(num_layers, num_wires*9)

print(w)
# %%
# There weights
# np.random.seed(0)
# num_layers = 4
# var_init = 0.05 * np.random.randn(num_layers, 5)
# print(var_init)

# %%
def layer_mod(w, num_wires):
    for i in range(num_wires):
        qml.Rotation(w[i], wires=i)

    for i in range(num_wires):
        qml.Squeezing(w[i + num_wires], 0.0, wires=i)

    for i in range(num_wires):
        qml.Rotation(w[i + (num_wires*2)], wires=i)

    for i in range(num_wires):
        # Bias
        qml.Displacement(w[i + (num_wires*3)], 0.0, wires=i)

    for i in range(num_wires):
        # Element-wise nonlinear transformation
        qml.Kerr(w[i + (num_wires*4)], wires=i)

# %%
@qml.qnode(dev)
def quantum_neural_net(weights, x=None, bs=False, num_wires=num_wires):
    '''
        For each layer, apply the inputs to the gates to update the weights

        weights: list of lists of scalar weights (of length 5)
        x: list of stock closing values for 5 stocks
    '''

    # Encode input x into quantum state
    for i in range(num_wires):
        qml.Displacement(x, 0.0, wires=i)

    # "layer" subcircuits
    if bs == True:
        for w in weights:
            layer_bs_mod(w, num_wires)
    else:
        for w in weights:
            layer_mod(w, num_wires)

    output = []
    for i in range(num_wires):
        output.append(qml.expval(qml.X(i)))

    return output

# %%
def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

# %%
def cost(var, features, labels):
    preds = [quantum_neural_net(var, x=x, bs=False, num_wires=num_wires) for x in features]
        return square_loss(labels, preds)

# %%
data = np.loadtxt("dataset/sine.txt")
X = data[:, 0]
Y = data[:, 1]

# %%
plt.figure()
plt.scatter(X, Y)
plt.xlabel("x", fontsize=18)
plt.ylabel("f(x)", fontsize=18)
plt.tick_params(axis="both", which="major", labelsize=16)
plt.tick_params(axis="both", which="minor", labelsize=16)
plt.show()

# %%

opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

var = w
for it in range(25):
    var = opt.step(lambda v: cost(v, X, Y), var)
    print("Iter: {} | Cost: {} ".format(it + 1, cost(var, X, Y)))

# %%

x_pred = np.linspace(-1, 1, 50)
predictions = [quantum_neural_net(var, x=x_) for x_ in x_pred]

# %%

plt.figure()
plt.scatter(X, Y)
plt.scatter(x_pred, predictions, color="green")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tick_params(axis="both", which="major")
plt.tick_params(axis="both", which="minor")
plt.show()

# %%
variance = 1.0

plt.figure()
x_pred = np.linspace(-2, 2, 50)
for i in range(7):
    rnd_var = variance * np.random.randn(num_layers, 7)
    predictions = [quantum_neural_net(rnd_var, x=x_) for x_ in x_pred]
    plt.plot(x_pred, predictions, color="black")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.tick_params(axis="both", which="major")
plt.tick_params(axis="both", which="minor")
plt.show()

# %%



#########################################

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
