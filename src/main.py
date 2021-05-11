
''' 
Create an instance of the class Model which solves a differential equation
using a feed forward neural network.

The initialization necessits:
   _ a function f defining the dynamic of the system (i.e. y'=f(x,y))
   _ the trial solution used 
Optionnal arguments are:
   _ the size of the hidden layers, given as a tuple hidden_layer=(l1,l2,...,lN)
      Currently, it only works with one layer, i.e. Layers=(l1,)!!
   _ max_iter the maximum number of iterations of the minimization algorithm
   _ method is the algorithm used to train the network, see scipy.optimize.minimize
      for a list of available methods. 
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
Problem = 2
F, Trial, Solution, Interval = Problems[Problem]

# Network's parameters
Training_Pts = 30
Layers       = (20,)
max_iter     = 30
Training_Set = np.linspace(Interval[0], Interval[1], Training_Pts)
Regs = [0, 0.5, 1, 5]    # Regularization parameter



# To be compared with the numerical solution
x = np.linspace(Interval[0], Interval[1], 1000)
Sol = Solution(x)               # Exact Solution
NormSol = np.sum(Sol**2)



# Train networks with different regularization parameters
Phis = []                       # Container for the numerical solutions
L2_Error = []                   # Container for the L2 errors
PtWise_Squared_Error = []       # Container for pointwise squared errors
Legend = []

for reg in Regs:
    Net = Model(F, Trial, Layers, max_iter=max_iter,
                activation='sigmoid',
                reg=reg)
    Net.fit(Training_Set)
    Phi, _ = Net.phi(x)
    
    Phis.append(Phi)
    E = (Phi-Sol)**2/NormSol
    PtWise_Squared_Error.append(E)
    L2_Error.append(np.sum(E))
    Legend.append(r'$\lambda$ = {}'.format(reg))


gs = gridspec.GridSpec(2,2)
    
fig = plt.figure(figsize=(12,8))

ax = plt.subplot(gs[0,:])
plt.plot(x, Phis[0], label=r'$\lambda = 0$')
plt.plot(x, Sol, label='Analytical solution')
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\phi(x)$')

ax = plt.subplot(gs[1,0])
for i in range(len(PtWise_Squared_Error)):
    plt.semilogy(x, PtWise_Squared_Error[i])
plt.legend(Legend)
plt.xlabel('x')
plt.title('Pointwise squared error') 

ax = plt.subplot(gs[1,1])
plt.semilogy(L2_Error, 'k+', markersize=16)
locs, labels = plt.xticks()
ticks = ['']
ticks.extend(Regs)
ticks.append('')
plt.xticks(locs, ticks)
plt.xlabel(r'$\lambda$')
plt.title(r'Normalized $L^2$ norm of the error function')
plt.legend()

plt.show()
    




