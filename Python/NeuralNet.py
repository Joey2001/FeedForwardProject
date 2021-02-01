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
    def __init__(self, layerSizes, Lambda):
        # Defines a term to limit the model overfitting the data
        self.Lambda = Lambda

        # Defines how many layers there are and how many neurons in each layer
        self.layerSizes = layerSizes

        # Creates an array of matricies, one less weight matrix than layers
        self.W = [0] * (len(self.layerSizes) - 1)

        # Iterates through the array and initializes the weights to random weights
        for i in range(len(self.W)):
            self.W[i] = np.random.randn(self.layerSizes[i],self.layerSizes[i + 1])
        
    def forward(self, X):
        # Creates the variables 'z' and 'a' to hold the data as 'X' is passed through the matricies
        self.z = [0] * len(self.layerSizes)
        self.a = [0] * len(self.layerSizes)

        # Initializes the first variable for 'z' and 'a' to be the input
        self.z[0] = X
        self.a[0] = X

        # Propogates the inputs through the network
        for i in range(1, len(self.layerSizes)):
            self.z[i] = np.matmul(self.a[i - 1], self.W[i - 1])
            self.a[i] = self.sigmoid(self.z[i])

        yHat = self.a[len(self.a) - 1]

        # Returns the result of feeding the input through the matrix operations
        return yHat
        
    def sigmoid(self, z):
        # Applies the sigmoid activation function to scalar, vector, or matrix
        # Maps any input down to between 0 and 1. {(-inf, inf) -> (0, 1)}
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self, z):
        # Derivative of the sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costFunction(self, X, y):
        # Computes the cost for given X and y, weights stored in class are used
        self.yHat = self.forward(X)

        # Sums the squares of the weight matricies and applies to cost to reduce overfitting
        sumWeights = 0
        for i in range(len(self.W)):
            sumWeights += np.sum((self.W[i]**2))

        J = 0.5 * sum((y - self.yHat)**2) / X.shape[0] + (self.Lambda / 2) * sumWeights
        return J
        
    def costFunctionPrime(self, X, y):
        # Computes the derivatives of the cost with respect to the weights for given X and y
        self.yHat = self.forward(X)

        # Creates arrays 'delta' and 'dJdW' to find how much
        # the weights need to change to converge to a solution
        delta = [0] * len(self.W)
        dJdW = [0] * len(self.W)
        
        # Last element is the derivative with respect to yHat mulitplied by the derivative of the
        # activation function of 'z' right after being passed from the last weight matrix
        delta[len(delta) - 1] = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z[len(self.z) - 1]))
        dJdW[len(dJdW) - 1] = np.dot(self.a[len(self.a) - 2].T, delta[len(delta) - 1]) + (self.Lambda * self.W[len(self.W) - 1])

        # From the second to last element, derivative of the previous weights
        # is calculated using the previous 'delta' and weight multiplied by
        # the derivative of the activation function of the previous 'z'
        for i in reversed(range(len(dJdW) - 1)):
            delta[i] = np.dot(delta[i + 1], self.W[i + 1].T)*self.sigmoidPrime(self.z[i + 1])
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
        for i in range(len(self.W)):
            end = start + (self.layerSizes[i + 1] * self.layerSizes[i])
            self.W[i] = np.reshape(params[start:end], (self.layerSizes[i] , self.layerSizes[i + 1]))
            start = end
        
    def computeGradients(self, X, y):
        # Retrieves the changes in weights
        dJdW = self.costFunctionPrime(X, y)

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
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        self.X = X
        self.y = y

        self.J = []
        
        params0 = self.N.getParams()
        print('init params\n', params0)

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

        print('final params\n', self.N.getParams())

layers = [2, 3, 1]
errorCorrectingTerm = 0
NN = Neural_Network(layers, errorCorrectingTerm)
T = trainer(NN)
T.train(X,y)
print(NN.forward(X))

# numgrad = computeNumericalGradient(NN, X, y)
# grad = NN.computeGradients(X, y)

# diffNorm = np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)

# print(diffNorm)