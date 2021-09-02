# Feed Forward Neural Network
# Based on the work of Stephen Welch and his videos on "Neural Networks Demystified"
#
# This adaptation of Stephen Welch's code allows for scalable neural networks
#
# Joseph Artura
# @Joey2001

import numpy as np

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalize data to be between 0 and 1
X /= np.amax(X, axis=0)
y /= 100

class Neural_Network(object):
    def __init__(self, layerSizes, Lambda, input):
        # Adding a column of ones to the input to act as the bias term;
        # incorporating the bias into the weight matricies allows the term to be tuned in training.
        self.input = np.hstack(((np.ones((input.shape[0], 1))), input))

        # Defines a term to limit the model overfitting the data
        self.Lambda = Lambda

        # Defines how many layers there are and how many neurons in each layer
        self.layerSizes = layerSizes

        # Creates an array of matricies, one less weight matrix than layers
        self.W = [0] * (len(self.layerSizes) - 1)

        # Iterates through the array and initializes the weights to random weights and biases to random biases
        for i in range(len(self.W) - 1):
            self.W[i] = np.random.randn(self.layerSizes[i] + 1, self.layerSizes[i + 1] + 1)
            self.W[i][::, 0] = 1
        self.W[-1] = np.random.randn(self.layerSizes[-2] + 1, self.layerSizes[-1])
        
    def forward(self):
        # Creates the variables 'z' and 'a' to hold the data as 'X' is passed through the matricies
        self.z = [0] * len(self.layerSizes)
        self.a = [0] * len(self.layerSizes)

        # Initializes the first variable for 'z' and 'a' to be the input
        self.z[0] = self.input
        self.a[0] = self.input

        # Propogates the inputs through the network
        for i in range(1, len(self.layerSizes)):
            self.z[i] = np.matmul(self.a[i - 1], self.W[i - 1])
            self.a[i] = self.sigmoid(self.z[i])
            if(i < len(self.layerSizes) - 2):
                self.a[i][0, ::] = 1
        
        # Returns the result of feeding the input through the matrix operations
        return self.a[-1]

        
    def sigmoid(self, z):
        # Applies the sigmoid activation function to scalar, vector, or matrix
        # Maps any input down to between 0 and 1. {(-inf, inf) => [0, 1]}
        return 1.0 / (1.0 + np.exp(-z))
    
    def sigmoidPrime(self, z):
        # Derivative of the sigmoid function
        return 1.0 / (2 + np.exp(z) + np.exp(-z))
    
    def costFunction(self, y):
        # Computes the cost for given X and y, weights stored in class are used
        self.yHat = self.forward()

        # Sums the squares of the weight matricies and applies to cost to reduce overfitting
        sumWeights = 0
        for i in range(len(self.W)):
            sumWeights += np.sum((self.W[i]**2))
        
        return 0.5 * sum((y - self.yHat)**2) / self.input.shape[0] + (self.Lambda / 2) * sumWeights
        
    def costFunctionPrime(self, y):
        # Computes the derivatives of the cost with respect to the weights for given X and y
        self.yHat = self.forward()

        # Creates arrays 'delta' and 'dJdW' to find how much
        # the weights need to change to converge to a solution
        delta = [0] * len(self.W)
        dJdW = [0] * len(self.W)
        
        # Last element is the derivative with respect to yHat mulitplied by the derivative of the
        # activation function of 'z' right after being passed from the last weight matrix
        delta[-1] = np.multiply(self.yHat - y, self.sigmoidPrime(self.z[-1]))
        dJdW[-1] = np.dot(self.a[-2].T, delta[-1]) + (self.Lambda * self.W[-1])

        # From the second to last element, derivative of the previous weights
        # is calculated using the previous 'delta' and weight multiplied by
        # the derivative of the activation function of the previous 'z'
        for i in reversed(range(len(dJdW) - 1)):
            delta[i] = np.dot(delta[i + 1], self.W[i + 1].T) * self.sigmoidPrime(self.z[i + 1])
            dJdW[i] = np.dot(self.z[i].T, delta[i]) + (self.Lambda * self.W[i])
        
        # Returns the derivative with respect to the weights
        return dJdW
    
    def getParams(self):
        # Creates a variable 'params' to store the first weight matrix as a vector
        params = self.W[0].ravel()

        # Goes through all of the weights and appends the vector of that weight to 'params'
        for i in range(1, len(self.W)):
            params = np.concatenate((params, self.W[i].ravel()))

        # Returns the weights in a vector formate
        return params
    
    def setParams(self, params):
        # Has an input 'params' that resets all of the weight values in a vector format
        # Creates 'start' and 'end' to retrieve the correct indices to change the weights
        start = 0
        for i in range(len(self.W) - 1):
            end = start + ((self.layerSizes[i] + 1) * (self.layerSizes[i + 1] + 1))
            self.W[i] = np.reshape(params[start:end], (self.layerSizes[i] + 1, self.layerSizes[i + 1] + 1))
            self.W[i][::,0] = 1
            start = end
        end = start + ((self.layerSizes[-2] + 1) * self.layerSizes[-1])
        self.W[-1] = np.reshape(params[start: end], (self.layerSizes[-2] + 1, self.layerSizes[-1]))
        
    def computeGradients(self, y):
        # Retrieves the changes in weights
        dJdW = self.costFunctionPrime(y)

        # Turns first change in weight into vector
        params = dJdW[0].ravel()

        # Goes through all of the changes in weights and
        # appends the change of that weight to 'params'
        for i in range(1, len(self.W)):
            params = np.concatenate((params, dJdW[i].ravel()))

        # Returns the changes in weights in a vector format
        return params

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 

from scipy import optimize


class trainer(object):
    def __init__(self, N):
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.y))   
        
    def costFunctionWrapper(self, params, y):
        self.N.setParams(params)
        cost = self.N.costFunction(y)
        grad = self.N.computeGradients(y)
        return cost, grad
        
    def train(self, y):
        self.y = y

        self.J = []
        
        params0 = self.N.getParams()
        print('init params\n', params0)

        options = {'maxiter': 500, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

        print('final params\n', self.N.getParams())

layers = [2, 3, 1]
errorCorrectingTerm = 0
NN = Neural_Network(layers, errorCorrectingTerm, X)
T = trainer(NN)
T.train(y)
print(NN.forward())
