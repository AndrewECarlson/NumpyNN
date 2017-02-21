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
from scipy.stats import logistic as sigmoid2
import warnings

warnings.filterwarnings("ignore")


class MLP_Neural_Network:
    # TODO generalize this to allow any number of layers

    def __init__(self,input, hidden, output):
        self.input = input
        self.hidden = hidden
        self.output = output

        # shape of weights should be (number of nodes in next layer, num nodes in current layer +1)
        self.theta_1 = np.random.randn(self.hidden, self.input+1)             # first col is bias weight
        self.theta_2 = np.random.randn(self.output, self.hidden+1)             # first col is bias weight

        # Initialize/declare activations and check shapes are all compatible
        self.a1 = np.zeros((self.input+1,1))

        self.a2 = np.matmul(self.theta_1, self.a1)
        # add bias neuron
        temp = self.a2
        self.a2 = np.ones((self.hidden + 1, 1))
        self.a2[1:] = temp

        #calculate a3

        self.a3 = np.matmul(self.theta_2, self.a2)


    def feed_forward(self, input,t1=None,t2=None):

        if t1==None or t2 == None:
            t1 = self.theta_1
            t2 = self.theta_2

        # put input into a1 with bias neuron
        self.a1 = np.ones((self.input+1, 1))
        self.a1[1:] = input.reshape((self.input, 1))

        # calculate a2
        self.a2 = np.matmul(t1, self.a1)
        self.a2 = sigmoid(self.a2)

        # add bias neuron
        temp = self.a2
        self.a2 = np.ones((self.hidden + 1, 1))
        self.a2[1:] = temp

        # calculate a3
        self.a3 = np.matmul(t2, self.a2)
        self.a3 = sigmoid(self.a3)

        return self.a3
    feed_forward_vec = np.vectorize(feed_forward)

    def cost(self,regularization_term, input, expected_output,theta1=None,theta2= None):
        if theta1 is None:
            theta1 = self.theta_1
        if theta2 is None:
            theta2 = self.theta_2

        prediction = self.feed_forward(input,theta1,theta2)
        cost = 0

        for k in range(self.output):
            if prediction == 0:
                prediction = 0.000000000000000001
            cost = cost + -1 * expected_output * math.log(prediction) - (1 - expected_output) * math.log(1 - prediction)

        h_weights = np.square(self.theta_1)
        o_weights = np.square(self.theta_2)
        h_weights = np.sum(h_weights)
        o_weights = np.sum(o_weights)

        weights = np.add(h_weights,o_weights)
        weights = weights*regularization_term/2
        cost = np.add(cost,weights)


        return cost

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





    def backprop(self, input, expected, learning_rate= .001, regularization_term = .0001):

        d3 = np.subtract(self.feed_forward(input), expected.reshape((self.output,1)))

        d2 = np.matmul(self.theta_2.T,d3)
        gz = np.multiply(self.a2,1-self.a2)
        d2 = np.multiply(d2,gz)

        # calculate derivative of cost function, apparently there is a proof to show this is equal to it

        delta_2 = np.matmul(d3,self.a2.T)
        delta_1 = np.matmul(d2, self.a1.T)[1:]              # getting rid of bias neuron because there is no weight
                                                            # feeding into it to update


        # calculate and add regularization terms without regularizing bias neuron
        regularization = (np.dot(regularization_term, self.theta_2[:, 1:]))
        delta_2[:, 1:] = np.add(delta_2[:, 1:],regularization)


        regularization = (np.dot(regularization_term, self.theta_1[:, 1:]))

        delta_1[:, 1:] = np.add(delta_1[:, 1:], regularization)


        #run gradient checking
        gradAppx1,gradAppx2 = self.grad_check(input,expected,.0001,regularization_term)

        diff1 = delta_1 - gradAppx1
        # diff1 = np.square(diff1)
        # diff1 = np.sum(diff1)

        diff2 = delta_2 - gradAppx2
        # diff2 = np.square(diff2)
        # diff2 = np.sum(diff2)


        print("diff between descent and appx for theta 1\n",diff1)
        print("diff between descent and appx for theta 2\n",diff2)

        #
        #Update Weights
        self.theta_2 = np.subtract(self.theta_2, np.dot(learning_rate, delta_2))
        self.theta_1 = np.subtract(self.theta_1, np.dot(learning_rate, delta_1))
        return delta_2

    def train(self, epochs, inputs, expected_outputs, learning_rate, regularization_term):
        for j in range(epochs):
            for i in range(len(inputs)):
                self.backprop(inputs[i],expected_outputs[i],learning_rate,regularization_term)
            if j % 50 == 0:

                cost = 0
                for i in range(len(inputs)):
                    cost += self.cost(regularization_term, inputs[i], expected_outputs[i])
                cost /= len(inputs)
                print("{0} accuracy: {1} cost: {2}".format(j, self.test(inputs, expected_outputs), cost))

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
    nn = MLP_Neural_Network(4, 2, 1)

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
    nn.backprop(test_inputs[0],test_labels[0],.001,.00001)

   # print(nn.feed_forward(test_inputs[0]))

    #nn.train(300,test_inputs, test_labels,.001,.01)
   # print("feed forward results. should be 0", nn.feed_forward(np.array([0,0,0,1])))


   # print(nn.test(testdata.drop('y', axis=1).as_matrix(),testdata['y'].as_matrix()))















