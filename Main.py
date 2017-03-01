#
#
#   Andrew Carlson
#

#


import numpy as np
import math
import pandas as pd
from scipy.optimize import check_grad


from scipy.special import expit as sigmoid #ignore unresolved error
import warnings

warnings.filterwarnings("ignore")


class MLP_Neural_Network:
    # TODO generalize this to allow any number of layers

    def __init__(self,*args):
        self.layers = args
        self.theta = []
        self.a = []


        # # shape of weights should be (number of nodes in next layer, num nodes in current layer +1)
        # first column is the bias weight
        for i in range(len(self.layers)-1):
            self.theta.append(np.random.randn(self.layers[i+1], self.layers[i]+1))

        # dry run of feed forward to initialize self.a and check shapes are correct
        self.a.append(np.zeros((self.layers[0]+1,1)))
        for i in range(1,len(self.layers)):
            temp = np.matmul(self.theta[i-1], self.a[i-1])
            temp = sigmoid(temp)
            a = np.ones((self.layers[i]+1, 1))
            a[1:] = temp
            self.a.append(a)


        # self.theta_1 = np.random.randn(self.hidden, self.input+1)             # first col is bias weight
        # self.theta_2 = np.random.randn(self.output, self.hidden+1)             # first col is bias weight
        #
        # # Initialize/declare activations and check shapes are all compatible
        # self.a1 = np.zeros((self.input+1,1))
        #
        # self.a2 = np.matmul(self.theta_1, self.a1)
        # # add bias neuron
        # temp = self.a2
        # self.a2 = np.ones((self.hidden + 1, 1))
        # self.a2[1:] = temp
        #
        # #calculate a3
        #
        # self.a3 = np.matmul(self.theta_2, self.a2)


    def feed_forward(self, x, theta = None):
        # allow use of alternative weights for gradient checking
        if theta == None:
            theta = self.theta

        # initialize a0 with x to begin forward propagation
        self.a[0] = np.ones(self.a[0].shape)
        self.a[0][1:] = x.reshape((self.layers[0], 1))
        # same code used to initialize self.a
        for i in range(1,len(self.layers)):
            # calculate activations for current layer
            temp = np.matmul(self.theta[i-1], self.a[i-1])
            temp = sigmoid(temp)
            # add bias neuron
            a = np.ones((self.layers[i]+1, 1))
            a[1:] = temp
            self.a[i] = a
        # return last actiation layer excluding the bias neuron
        return self.a[-1][1:]

    def cost(self,regularization_term, input, expected_output,theta=None):
        if theta is None:
            theta = self.theta

        prediction = self.feed_forward(input,theta)
        cost = 0
        for k in range(self.layers[-1]):
            if prediction == 0:
                prediction = 0.000000000000000001
            cost = cost + -1 * expected_output * math.log(prediction) - (1 - expected_output) * math.log(1 - prediction)

        # calculate regularization term
        weights = 0
        for weight in self.theta:
            squares = np.square(weight)
            squares = np.sum(squares)
            weights += squares
        weights *= regularization_term/2
        cost = np.add(cost,weights)

        return cost
    # todo  update grad checking for flexible layers
    def grad_check(self, input, expected_output,epsilon, regularization_term):
        dtheta1 = np.zeros(self.theta_1.shape).flatten()
        dtheta2 = np.zeros(self.theta_2.shape).flatten()

        for i in range(len(dtheta1)+len(dtheta2)):
            if i < len(dtheta1):
                flat_theta1min = self.theta_1.flatten()
                flat_theta1max = self.theta_1.flatten()
                flat_theta1min[i] -= epsilon
                flat_theta1max[i] += epsilon
                theta1min = flat_theta1min.reshape(self.theta_1.shape)
                theta1max = flat_theta1max.reshape(self.theta_1.shape)
                cost_high = self.cost(regularization_term,input,expected_output,theta1max,self.theta_2)
                cost_low = self.cost(regularization_term,input,expected_output,theta1min,self.theta_2)
                dtheta1[i] = (cost_high - cost_low)/(2*epsilon)
            else:
                j = i-len(dtheta1)


                flat_theta2min = self.theta_2.flatten()
                flat_theta2max = self.theta_2.flatten()
                flat_theta2min[j] -= epsilon
                flat_theta2max[j] += epsilon
                theta2min = flat_theta2min.reshape(self.theta_2.shape)
                theta2max = flat_theta2max.reshape(self.theta_2.shape)
                cost_high = self.cost(regularization_term,input,expected_output,self.theta_1,theta2max)
                cost_low = self.cost(regularization_term,input,expected_output,self.theta_1,theta2min)
                dtheta2[j] = (cost_high - cost_low)/(2*epsilon)


        dtheta1 = dtheta1.reshape(self.theta_1.shape)
        dtheta2 = dtheta2.reshape(self.theta_2.shape)


        return dtheta1,dtheta2

    def backprop(self, x, y, learning_rate= .001, regularization_term = .0001):

        # create array d to help build delta.
        d = [0 for x in range(len(self.theta))]
        delta = [0 for x in range(len(self.theta))]

        self.feed_forward(x)
        temp = y
        y = np.ones(self.a[-1].shape)
        y[1:] = temp.reshape((self.layers[-1], 1))
        d[-1] = np.subtract(self.a[-1],y)
        # -1  is because the last value is hardcoded above and the inputs dont contribute any error to the next layer
        # so with  the zero being exclusive it is not iterated over
        for i in range(len(self.theta)-1,0,-1):

            temp = np.matmul(self.theta[i].T, d[i][1:])   #remove d for bias term
            gz = np.multiply(self.a[i], 1-self.a[i])
            d[i-1] = np.multiply(temp, gz)

        for i in range(len(d)):
            delta[i] = np.matmul(d[i],self.a[i].T)[1:]    # calculate delta and remove bias layer because bias nodes have
                                                        #   no weights feeding into them to update

        # # calculate and add regularization terms without regularizing bias neuron
        # regularization = (np.dot(regularization_term, self.theta_2[:, 1:]))
        # delta_2[:, 1:] = np.add(delta_2[:, 1:],regularization)
        #
        for i in range(len(self.theta)):
            regularization = (np.multiply(regularization_term, self.theta[i][:, 1:]))
            delta[i][:, 1:] = np.add(delta[i][:, 1:], regularization)

        for i in range(len(self.theta)):
            self.theta[i] = np.subtract(self.theta[i], np.multiply(learning_rate,delta[i]))

    def train(self, epochs, inputs, expected_outputs, learning_rate = .01, regularization_term=.0001):
        for j in range(epochs):
            for i in range(len(inputs)):
                self.backprop(inputs[i],expected_outputs[i],learning_rate,regularization_term)
            if j % 500 == 0:
                print("epoch done:", j)
                print("feed forward results. should be 0", nn.feed_forward(np.array([0,0,0,1])))
            #
            #     cost = 0
            #     for i in range(len(inputs)):
            #         cost += self.cost(regularization_term, inputs[i], expected_outputs[i])
            #     cost /= len(inputs)
            #     print("{0} accuracy: {1} cost: {2}".format(j, self.test(inputs, expected_outputs), cost))

    def test(self, inputs, expected_outputs):
        """


        :param inputs:
        :param expected_outputs:
        :return: Average accuracy over given inputs
        """
        results = np.zeros(len(inputs))

        for i in range(len(inputs)):
            results[i] = self.feed_forward(inputs[i])
        results = np.greater(results,.5).astype(expected_outputs.dtype)
        correct = np.equal(results,expected_outputs).astype(expected_outputs.dtype)

        return correct.mean()

if __name__ == "__main__":
    #create Network
    nn = MLP_Neural_Network(4,6,2, 1)

    # set up training and test data
    data = pd.read_csv("testdata.txt")
    data = data.sample(frac=1)
    cutpoint = int(.8*len(data))
    traindata = data[:cutpoint]
    testdata = data[cutpoint:]
    test_inputs = testdata.drop(['y'], axis = 1).as_matrix()
    test_labels = testdata["y"].as_matrix()

    #run desired functions


    #print(check_grad(nn.cost,nn.backprop,test_inputs[0],test_labels[0]))
    nn.backprop(test_inputs[0],test_labels[0],.01,.0001)

   # print(nn.feed_forward(test_inputs[0]))

    nn.train(10000,test_inputs, test_labels,.001,.01)
    print("feed forward results. should be 0", nn.feed_forward(np.array([0,0,0,1])))


   # print(nn.test(testdata.drop('y', axis=1).as_matrix(),testdata['y'].as_matrix()))















