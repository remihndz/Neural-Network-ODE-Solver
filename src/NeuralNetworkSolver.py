import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


''' 
Create an instance of the class Model which solves a differential equation.
The initialization necessits:
   _a function f defining the dynamic of the system (i.e. y'=f(x,y))
   _the trial solution used 
   _the size of the hidden layers, given as a tuple hidden_layer=(l1,l2,...,lN)
      Currently, it only works with one layer, i.e. hidden_layer=(l1,)!!
Optionnal arguments are:
   _max_iter the maximum number of iterations of the minimization algorithm
   _method is the algorithm used to train the network, see scipy.optimize.minimize
     for a list of available methods. 
'''


''' 
TODO:
     _add the possibility to solve PDE 
     _add a method to compute second order derivatives of the network 
     _add the possibility to solve second order differential equations
     _make better use of the linear algebra packages to improve speed
'''


class Model:

    def __init__(self, f, trial_solution, hidden_layers=(5,), max_iter=100, method='BFGS'):

        self._hidden_layers = hidden_layers
        self._max_iter = max_iter
        self._f = f
        self._trial = trial_solution
        self._method=method
        
    def _logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def _D_logistic(self, x):
        return self._logistic(x)*(1.0-self._logistic(x))

    def _make_thetas(self, layers):

        self.thetas = []
        self.thetas_sizes = []

        for i in range(len(layers))[:-1]:
            rows = layers[i]
            if i == len(layers)-2:
                rows -= 1
            cols = layers[i+1]
            self.thetas.extend(np.random.randn(rows+1, cols).flatten(order='C'))
            self.thetas_sizes.append((rows+1, cols))

    def _reshape_thetas(self, flattened_thetas):
        
        reshaped_thetas = []
        elements_count  = 0
        for theta_size in self.thetas_sizes:
            rows, cols = theta_size
            elements = rows * cols
            reshaped_thetas.append(np.reshape(
                flattened_thetas[elements_count:elements+elements_count],
                newshape=(rows, cols), order='C'))
            elements_count += elements
        return reshaped_thetas

    def _forward_propagation(self, x, thetas):

        rows = 1            # One dimensional input x\in\R
        layers = []

        current_layer = np.array([1.0,x]).reshape((1,2))
        #current_layer = np.hstack((np.ones((rows,1)), x))

        theta_matrix1, theta_matrix2 = thetas
        hidden_layer = np.matmul(current_layer, theta_matrix1)
        layers.append(hidden_layer)
        output = np.matmul(self._logistic(hidden_layer), theta_matrix2)
        layers.append(output)
        
        return output[0]

    def _derivative_forward_propagation(self, x, thetas):
        rows   = 1
            
        current_layer = np.array([1,x])        
        theta_matrix1, theta_matrix2 = thetas
        theta_matrix2 = np.multiply(theta_matrix1[1:].T, theta_matrix2)
        
        hidden_layer = self._D_logistic(np.matmul(current_layer, theta_matrix1))
        output = np.matmul(hidden_layer, theta_matrix2)
        
        return output
        
    def _phi(self, x, thetas):
        ''' 
        The approximated solution to the ODE and its first derivative
        Method used for training
        For computation, use self.phi
        '''
        reshaped_thetas = self._reshape_thetas(thetas)
        output_net = self._forward_propagation( x, reshaped_thetas)[-1]
        D_output_net = self._derivative_forward_propagation(x, reshaped_thetas)

        return self._trial(x,output_net, D_output_net)

    def _define_dynamics(self, f):
        self._f = f
        
    def _cost(self, thetas, training_set):

        cost = 0.0
        for x in training_set:
            phi,dphi = self._phi(x,thetas)
            cost += (dphi-self._f(x,phi))**2
        return cost

    def fit(self, training_set):
        self._make_thetas((1,)+self._hidden_layers + (1,))

        optimized = minimize(fun=self._cost, x0=self.thetas, args=(training_set),
                             method=self._method, options={'maxiter':self._max_iter})

        self.thetas = optimized.x
        self.Optimized = optimized

    def phi(self, X):

        reshaped_thetas = self._reshape_thetas(self.thetas)    
        f,fp = [],[]
        if X.size>1:
            for x in np.array(X):
                output_net = self._forward_propagation( x, reshaped_thetas)[-1]
                D_output_net = self._derivative_forward_propagation(x, reshaped_thetas)
                aux = self._trial(x,output_net,D_output_net)
                f.append(aux[0])
                fp.append(aux[1])
            return f,fp
        output_net = self._forward_propagation(X, reshaped_thetas)[-1]
        D_output_net = self._derivative_forward_propagation(X, reshaped_thetas)
        aux = self._trial(x,output_net,D_output_net)        
        return np.array(aux[0]), np.array(aux[1])
                
