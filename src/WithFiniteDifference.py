import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class Network:

    def __init__(self, f, trial_solution, hidden_layers=(5,), max_iter=100, method='BFGS'):

        self._hidden_layers = hidden_layers
        self._max_iter = max_iter
        self._f = f
        self._trial = trial_solution
        self._method=method
        
    def _logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

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

    def _Net(self, X, thetas):

        if type(X)!=float and len(X.shape)!=0:
            rows = X.shape
            current_layer = np.hstack((np.ones((rows, 1)), X.reshape((rows, -1))))     
        else:
            rows = 1
            current_layer = np.array([1,X]).reshape((rows,2))
        
        # go through the thetas matrixes and calculate each layer
        for index, theta_matrix in enumerate(thetas[:-1]):
            # ith + 1 layer = matrix multiplication of the ith layer and the ith thetas
            current_layer = self._logistic(
                np.matmul(current_layer, theta_matrix))
            # add bias to the calculated layer except for the output layer
            if (index + 1) != len(thetas)-1:
                current_layer = np.hstack((np.ones((rows, 1)), current_layer))
        return np.matmul(current_layer, thetas[-1])

    def _DNet(self, x, thetas, order=1, fdm='centered'):
        '''
        # Compute derivatives using finite difference
        # The theoretical optimal value (up to a scaling f/(f'')
           for the step is \sqrt(4ε) with ε the machine epsilon
        # 1.4901161193847656e-08 is the value used by
           scipy's minimize routine 
        '''
        # eps = np.finfo(np.float64).eps # Machine epsilon
        eps = 1.4901161193847656e-08 ## scipy.optimize.minimize's eps 
        h = np.float64(2.0)*np.sqrt(eps)
        if order==1:
            if fdm=='centered':
                f = self._Net(x-h, thetas)
                h/=2.0
            else:
                f = self._Net(x, thetas)
            ff = self._Net(x+h, thetas)

            return (ff-f)/h

        if order==2:
            ff = self._Net(x+h, thetas)
            ff = self._Net(x+h, thetas)
            fb = self._Net(x-h, thetas)
            
            return (fb - 2*f + ff)/(h**2)

    def _phi(self, x, thetas):
        ''' 
        * The approximated solution to the ODE 
           and its first derivative
        * This method is used for training only
        * To compute the approximated solution
           after fitting the weights, use self.phi
        '''
        reshaped_thetas = self._reshape_thetas(thetas)
        N  = self._Net( x, reshaped_thetas)[0][0]
        DN = self._DNet(x, reshaped_thetas)[0][0]
        return self._trial(x, N, DN)

    def _define_dynamics(self, f):
        '''
        * Function defining the dynamics of the ODE:
           y'(x)=f(x,y(x))
        '''
        self._f = f

    def _cost(self, thetas, training_set):
        cost = 0.0
        for x in training_set:
            phi, dphi = self._phi(x, thetas)
            cost += (dphi-self._f(x, phi))**2
        return cost

    def fit(self, training_set, w0=None):
        
        if w0 is None:
            self._make_thetas((1,)+self._hidden_layers + (1,))
            w0 = self.thetas            

        optimized = minimize(fun=self._cost, x0=w0, args=(training_set),
                             method=self._method, options={'maxiter':self._max_iter})
        self.thetas = optimized.x
        self.Optimized = optimized

    def phi(self, X):
        
        reshaped_thetas = self._reshape_thetas(self.thetas)
        if type(X)==float:
            N  = self._Net(X, reshaped_thetas)
            dN = self._DNet(X, reshaped_thetas)
            return self._trial(x, N, dN)

        f, df = [], []
        for x in np.array(X):
            N  = self._Net(x, reshaped_thetas)
            dN = self._DNet(x, reshaped_thetas)
            aux = self._trial(x, N, dN)
            f.append(aux[0][0])
            df.append(aux[1][0])
        return np.array(f), np.array(df)
    

        
        
            
        
    

