###############################################################################
# Run code by typing "python3 part4.py" in same directory as train.txt and
# test.txt
#
# Program will automatically train 2 neurons (one with simple learning,
# another with error correction learning) using data from train.txt and test
# using test.txt
#
# Test results will be written to output.txt
###############################################################################

# make a network class to keep track of layers of neurons
# then take input from one to another
# at each layer, add a dummy node(x0) that outputs y = 1 only
#
# Neuron will initialize with weight[n], n = number of inputs
# auto add random weight
#
# calc_a(x) & calc_output(x) so it can be called by network
# maybe add a learn???

# need 10 output neurons for 10 classes {0...9}
# output layer neurons have weights too, just the same neuron,
# but output is final y, to be compared to d.

import random
import math


class Neuron:

    def __init__(self, n=64, lrate=0.5, alpha=0, epsilon=0.1):
        self.weight = []
        for i in range(n):
            self.weight.append(random.uniform(-1,1))
        self.learnrate = lrate

        self.alpha = alpha                              #for momentum
        self.prev_delta_weight = [0]*len(self.weight)   #for momentum

        self.epsilon = epsilon                          #sigmoid/output
        self.y = 0                                      #output
        self.prev_weight = [0]*len(self.weight)
        self.de = 0

    def sigmoid(self, a):
        return 1/(1 + math.exp(-a))


    def calc_a(self, x):
        a = 0
        for i in range(len(x)):
            a += self.weight[i] * x[i]
        return a


    def calc_output(self, x):
        a = self.calc_a(x)
        self.y = self.sigmoid(a)
        if self.y >= (1-self.epsilon):
            self.y = 1
        elif self.y <= self.epsilon:
            self.y = 0
        return self.y

    #backprop_error = backpropagated values from next layer (d-y) for output layer
    def learn(self, backprop_error, x):
        delta_weight = [0]*len(self.weight)
        f_prime = (self.y * (1 - self.y))
        self.de = f_prime * backprop_error

        # print ("weight length:" + str(len(self.weight)))
        # print ("Input length:" + str(len(x)))

        for i in range(len(self.weight)):
            delta_weight[i] = self.learnrate * x[i] * f_prime * backprop_error
            self.prev_weight[i] = self.weight[i]

            self.weight[i] +=   delta_weight[i] + \
                                self.alpha * self.prev_delta_weight[i]

            self.prev_delta_weight[i] = delta_weight[i]



    def print_weight(self):
        print("\n")
        for i in range(len(self.weight)):
            print("Weight " + str(i) + ": ", end="")
            print("%.3f" % self.weight[i])
        # print("\n")



class DummyNeuron:
    def __init__(self):
        self.y = 1

    def calc_output(self, x):
        return self.y



class InputNeuron:
    def __init__(self):
        self.y = 0

    def calc_output(self, x):
        return self.y

    def input_data(self, x):
        self.y = x



class NeuralNetwork:
    def __init__(self, n):
        self.layer = []
        for i in range(n+1):
            self.layer.append([])
            # add dummy neuron for x0 to all layers except output layer
            if i < n:
                self.layer[i].append(DummyNeuron())

    def add_layer(self, layer_n, node_n, lrate, alpha, epsilon):
        for i in range(node_n):
            if layer_n == 0:
                self.layer[layer_n].append(InputNeuron())
            else:
                self.layer[layer_n].append(Neuron(len(self.layer[layer_n-1]), \
                                                lrate, alpha, epsilon))


    def input_train_data(self, data, d_output):
        # print(len(self.layer[0]))
        # put data into input layer (except x0)
        for i in range(1, len(self.layer[0])):
            self.layer[0][i].input_data(data[i-1])

        # for every layer after input layer
        for i in range(1, len(self.layer)):
            x = self.get_layer_output(i-1)
            #for every node in layer
            for j in range(1, len(self.layer[i])):
                self.layer[i][j].calc_output(x)

        # check for error and adjust weight for output layer
        print("Starting to train output layer..")
        i_output = len(self.layer)-1
        for i in range(len(self.layer[i_output])):
            d = 0
            if i == d_output:
                d = 1
            else:
                d = 0
            de = d - self.layer[i_output][i].y
            if de == 0:
                continue
            else:
                x = self.get_layer_output(i_output-1)
                self.layer[i_output][i].learn(de, x)

        print("Starting to train hidden layer..")
        for i in reversed(range(1, len(self.layer)-1)):
            print("Training hidden layer " + str(i))
            for j in range(1, len(self.layer[i])):
                print("Training node no. " + str(j))
                de = self.get_layer_error(i+1, j)
                x = self.get_layer_output(i-1)
                self.layer[i][j].learn(de, x)


    def get_layer_output(self, layer_n):
        y = []
        for i in range(len(self.layer[layer_n])):
            y.append(self.layer[layer_n][i].y)
        return y


    # get dj of all layers above it
    def get_layer_error(self, layer_n, node_n):
        sum = 0
        for i in range(1, len(self.layer[layer_n])):
            sum +=  self.layer[layer_n][i].de * \
                    self.layer[layer_n][i].prev_weight[node_n]

        return sum



def read_data(filename):
    data = []
    d_output = []
    f = open(filename, 'r')
    for line in f:
        data_line = []
        data_line += line.split(',')

        last_element = len(data_line)-1
        if (data_line[last_element].endswith("\n")):
            data_line[last_element] = data_line[last_element][:-1]

        for i in range(len(data_line)):
            data_line[i] = float(data_line[i])

        d_output.append(data_line.pop())
        data.append(data_line)
    f.close()

    return (data, d_output)


def output_textfile(test_name, output_name):
    print("hello")


###############################################################################

if __name__ == '__main__':
    # print (output)
    nn = NeuralNetwork(2)
    nn.add_layer(0, 64, 0.5, 0, 0.1)
    nn.add_layer(1, 40, 0.5, 0, 0.1)
    nn.add_layer(2, 10, 0.5, 0, 0.1)
    dat, output = read_data("./training.txt")
    # print(len(dat[1]))
    for i in range(len(dat)):
        nn.input_train_data(dat[i], output[i])

    # make function in NeuralNetwork to get data



    print("Hello world!")
