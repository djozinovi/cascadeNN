import pandas as pd

from tensorflow import keras
from keras import backend as K

from tensorflow.keras import layers
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns

sns.set_style("darkgrid")


def fully_connected_model(input_size):
    """
    Creates a fully connected neural network model.

    Args:
        input_size: The size of the input layer.

    Returns:
        A Keras model.
    """
    activation_func = "elu"

    paramInput = layers.Input(shape=input_size)

    layer1 = layers.Dense(64, activation=activation_func)(paramInput)
    layer1 = layers.Dropout(0.2)(layer1)
    layer1 = layers.Dense(40, activation=activation_func)(layer1)
    output = layers.Dense(32)(layer1)

    model = keras.Model(inputs=paramInput, outputs=output)

    optimizerChosen = optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizerChosen, loss="mse")

    return model


def split_input_output(data):
    """
    Splits the data into input and output variables.

    Args:
        data: The data to split.
        nameStart: The data output type.

    Returns:
        A tuple of the input and output variables.
    """
    inputs = data.iloc[:, 2:7].to_numpy()
    outputs = data.iloc[:, 8:].to_numpy()

    outputs = np.log10(outputs + 1)

    return inputs, outputs


def load_data(dataFolder):
    """
    Loads the data from the CSV files.

    Args:
        nameStart: The prefix of the CSV file names.

    Returns:
        A tuple of the train parameters, train outputs, validation parameters, validation outputs, test parameters, and test outputs.
    """
    trainData = pd.read_csv(dataFolder + "Train.csv")
    valData = pd.read_csv(dataFolder + "Val.csv")
    testData = pd.read_csv(dataFolder + "Test.csv")

    trainParams, trainOutputs = split_input_output(trainData)
    valParams, valOutputs = split_input_output(valData)
    testParams, testOutputs = split_input_output(testData)

    return trainParams, trainOutputs, valParams, valOutputs, testParams, testOutputs


trainParams, trainOutputs, valParams, valOutputs, testParams, testOutputs = load_data(
    "data/"
)

# The line `model = fully_connected_model(trainParams.shape[1])` is creating a fully connected neural
# network model using the `fully_connected_model` function. The `trainParams.shape[1]` is the size of
# the input layer, which is determined by the number of columns in the `trainParams` data. The
# function `fully_connected_model` defines the architecture of the model, including the number of
# layers, the activation function, and the optimizer.
model = fully_connected_model(trainParams.shape[1])

# `earlystop` is an instance of the `EarlyStopping` callback class in Keras. It is used to stop the
# training process early if the validation loss does not improve for a certain number of epochs
# (`patience`). The `restore_best_weights` parameter is set to `True`, which means that the weights of
# the model will be restored to the best weights found during training.
earlystop = keras.callbacks.EarlyStopping(
    patience=30, restore_best_weights=True, monitor="val_loss"
)
# `reduce_lr` is an instance of the `ReduceLROnPlateau` callback class in Keras. It is used to reduce
# the learning rate of the optimizer when the validation loss does not improve for a certain number of
# epochs (`patience`).
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=15, min_lr=0.0000001
)

# The `model.fit()` function is used to train the neural network model. It takes in the training data
# (`trainParams` and `trainOutputs`) and trains the model for a specified number of epochs (500 in
# this case) with a batch size of 32.
history = model.fit(
    x=trainParams,
    y=trainOutputs,
    epochs=500,
    batch_size=32,
    callbacks=[earlystop, reduce_lr],
    validation_data=(valParams, valOutputs),
)

model.save("trainedDNNmodel.tf")

testPredictions = model(testParams).numpy()

# Calculating the mean squared error (MSE) between the predicted outputs of the model on
# the test data (`testPredictions`) and the actual test outputs (`testOutputs`).
testMSE = np.array(keras.metrics.mean_squared_error(testPredictions, testOutputs))
print("Test MSE: " + str(np.mean(testMSE)))


# As we took the log of the outputs when training the model we are now
# transforming it back for easier visualization
testPredictions = 10 ** (testPredictions) / 20000
testOutputs = 10 ** (testOutputs) / 20000

# Loading the discretized time bins values and dividing the outputs by the length of the time bins
# As to get a correct event rates per day
bins = np.power(10, np.arange(-4, 4.01, 0.25))
timeBins = np.load("data/timeBins.npy")
testPredictions = testPredictions / np.diff(bins)
testOutputs = testOutputs / np.diff(bins)


xticks = bins[:-1] + np.diff(bins) / 2
# Plotting a random example
k = np.random.randint(len(testPredictions))

plt.plot(xticks, testPredictions[k], label="DNN Prediction")
plt.plot(xticks, testOutputs[k], label="Simulation")

plt.ylabel("G-F")
plt.yscale("log")

plt.xscale("log")
plt.xlabel("Days")

plt.legend()
