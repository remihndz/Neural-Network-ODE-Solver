import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworkSolver import Model

 
def Pb1(x,y):
    return x**3 + 2*x + x**2*(1+3*x*x)/(1+x+x**3) - y*(x+(1+3*x*x)/(1+x+x**3))
def Sol1(x):
    return np.exp(-x**2/2)/(1+x+x**3) + x**2
def Trial1(x,N,dN):
    return 1 + x*N, N+x*dN

def Pb2(x,y):
    return np.exp(-x/5)*np.cos(x) - 1./5.*y
def Sol2(x):
    return np.exp(-x/5)*np.sin(x)
def Trial2(x,N,dN):
    return x*N, N + x*dN

''' Second order problem, currently not supported '''
def Pb3(x,y, yp):
    return -np.exp(-x/5.)*np.cos(x)/5. - y - yp/5.
def Sol3(x,y):
    return np.exp(-x/5.)*np.sin(x)
def Trial3(x,N,dN,ddN):
    return x+ x**2*N, 1 + 2*x*N + x**2*dN, 2*N + 4*x*dN + x**2*ddN


Problems = {
    1: (Pb1,Trial1,Sol1, (0,1)),
    2: (Pb2,Trial2,Sol2, (0,2)),
    3: (Pb3,Trial3,Sol3, (0,2))
    }


''' 
Create an instance of the class Model which solves a differential equation.
The initialization necessits:
   _a function f defining the dynamic of the system (i.e. y'=f(x,y))
   _the trial solution used 
Optionnal arguments are:
   _the size of the hidden layers, given as a tuple hidden_layer=(l1,l2,...,lN)
      Currently, it only works with one layer, i.e. hidden_layer=(l1,)!!

   _max_iter the maximum number of iterations of the minimization algorithm
   _method is the algorithm used to train the network, see scipy.optimize.minimize
     for a list of available methods. 
'''


Problem_number = 2
Number_of_Training_Points = 50

architecture = (20,)
max_iter = 100


Problem  = Problems[Problem_number]
F, Trial, Solution, Interval = Problem

x = np.linspace(Interval[0], Interval[1], 1000)
training_set = np.linspace(Interval[0], Interval[1], Number_of_Training_Points)
Sol = Solution(x)


model = Model(f = F, trial_solution = Trial, hidden_layers=architecture, max_iter=max_iter)
model.fit(training_set)
Approx, DApprox = model.phi(x)

Approx = np.array(Approx).reshape((x.shape))
DApprox = np.array(DApprox).reshape((x.shape))

plt.subplot(121)
plt.plot(x,Approx, label="Neural Network's solution")
plt.plot(x,Sol, label="Exact Solution")
plt.legend()

h = np.finfo(np.float64).eps
h = 2.0*np.sqrt(h)
dSol = (Solution(x+h)-Solution(x-h))/(2*h)
plt.subplot(122)
plt.plot(x,DApprox, label="Neural Network's solution")
plt.plot(x,dSol, label="Exact Solution")
plt.legend()

plt.show()

print(model.Optimized)

