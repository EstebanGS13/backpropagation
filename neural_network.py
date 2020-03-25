import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(s):
    return s * (1 - s)


class NeuralNetwork:

    def __init__(self, X, Y):
        self.inputs = X
        self.Y = Y
        self.weights1 = np.random.rand(self.inputs.shape[1], 4)
        self.weights2 = np.random.rand(self.Y.shape[1], 4)
        self.outputs = np.zeros(Y.shape)
        self.error_history = []
        self.epoch_list = []

    def feed_forward(self):
        self.hidden = sigmoid(np.dot(self.inputs, self.weights1))
        self.outputs = sigmoid(np.dot(self.hidden, self.weights2))

    def backpropagation(self):
        errors = self.Y - self.outputs
        derivatives2 = sigmoid_derivative(self.outputs)
        deltas2 = errors * derivatives2
        self.weights2 += np.dot(self.hidden.T, deltas2)

        derivatives1 = sigmoid_derivative(self.hidden)
        deltas1 = np.dot(deltas2, self.weights2.T) * derivatives1
        self.weights1 += np.dot(self.inputs.T, deltas1)

    def train(self, epochs=25000):
            for epoch in range(epochs):
                self.feed_forward()
                self.backpropagation()
                #
                # self.error_history.append(np.average(np.abs((self.error))))
                # self.epoch_list.append(epoch)

    def predict(self, new_input):
        hidden = sigmoid(np.dot(new_input, self.weights1))
        prediction = sigmoid(np.dot(hidden, self.weights2))
        return np.round(prediction, 4)


if __name__ == '__main__':
    inputs = np.array([[1, 1, 1, 1, 1, 1, 0],
                       [0, 1, 1, 0, 0, 0, 0],
                       [1, 1, 0, 1, 1, 0, 1],
                       [1, 1, 1, 1, 0, 0, 1],
                       [0, 1, 1, 0, 0, 1, 1],
                       [1, 0, 1, 1, 0, 1, 1],
                       [1, 0, 1, 1, 1, 1, 1],
                       [1, 1, 1, 0, 0, 0, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 0, 1, 1]])

    outputs = np.array([[0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 1],
                        [0, 1, 1, 0],
                        [0, 1, 1, 1],
                        [1, 0, 0, 0],
                        [1, 0, 0, 1]])

    NN = NeuralNetwork(inputs, outputs)
    NN.train()

    example = np.array([[1, 1, 1, 1, 0, 0, 1]]) #3
    example_2 = np.array([[1, 1, 1, 0, 0, 0, 1]]) #7

    print(NN.predict(example), ' - Correct: 3 ', [0, 0, 1, 1])
    print(NN.predict(example_2), ' - Correct: 7 ', [0, 1, 1, 1])
