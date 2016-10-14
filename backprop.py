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

    def __init__(self, n=65, lrate=0.5, alpha=0, epsilon=0.1):
        self.weight = []
        for i in range(n+1):
            self.weight.append(random.uniform(-1,1))
        self.learnrate = lrate
        self.alpha = alpha              #for momentum
        self.last_delta_weight = 0      #for momentum
        self.epsilon = epsilon          #sigmoid/output
        self.y = 0                      #output


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
        if self.y >= (1-epsilon):
            self.y = 1
        elif self.y <= epsilon:
            self.y = 0
        return self.y



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
        for i in range(len(n)):
            self.layer.append([])
            # add dummy neuron for x0 to all layers except output layer
            if i <= n-1:
                self.layer[i].append(DummyNeuron())

    def add_layer(self, i, j, lrate, alpha, epsilon):
        for j in range(j):
            if i != 0:
                self.layer[i].append(Neuron(len(self.layer[i-1]), \
                                            lrate, alpha, epsilon))
            else:
                self.layer[i].append(InputNeuron())

    def input_data(self, filename):
        data = []
        d_output = 0





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
    dat, output = read_data("./training.txt")
    print (output)



    print("Hello world!")
