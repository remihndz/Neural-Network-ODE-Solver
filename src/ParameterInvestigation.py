import numpy as np
from NeuralNetworkSolver import Model
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Set the problem and trial solution (model)
def Pb1(x,y):
    return x**3 + 2*x + x**2*(1+3*x*x)/(1+x+x**3) - y*(x+(1+3*x*x)/(1+x+x**3))
def Sol1(x):
    return np.exp(-x**2/2)/(1+x+x**3) + x**2
def Trial1(x,N,dN):
    return 1 + x*N, N+x*dN    

Problems = {
    1: (Pb1,Trial1,Sol1, (0,1))}

Problem_number = 1
Number_of_Training_Points = 20

max_iter = 50

Problem  = Problems[Problem_number]
F, Trial, Solution, Interval = Problem

x = np.linspace(Interval[0], Interval[1], 1000)
training_set = np.linspace(Interval[0], Interval[1], Number_of_Training_Points)
Phi_a = Solution(x)



'''
-------------------------------------------------------------------
|  Investigate the size of the hidden layer in a shallow network  |
-------------------------------------------------------------------
'''

print('========================================\nInvestigating the width of the network\n========================================')

L2_error = [] # for ||phi_t-phi_a||
Loss_error = [] # for Loss function

N_Init = 30 # Number of random initialisation of the parameter guess for fitting
reg = 0     # L2 penalization for the weigths

width = [2,5,8,10,15,20,30]
for q in width:

    print('Training with q = {}....'.format(q))
    
    layers = (q,)
    model = Model(f = F, trial_solution = Trial, Layers=layers, max_iter=max_iter, reg=reg)

    # Try fitting with different initial guesses
    # to avoid local minima
    
    thetas_container = [] # for fitted parameters
    losses_container = [] # for the associated loss
    errors_container = [] # for the L2 error

    for i in range(N_Init):
        model.fit(training_set) # The fit method randomly initialize the parameters
    
        # thetas_container.append(model.thetas)
        losses_container.append(model.residual)

        Phi_t, _ = model.phi(x)
        L2_norm = np.sqrt(np.sum(np.abs(Phi_t-Phi_a)**2))
        errors_container.append(L2_norm)

    best_fit = min(losses_container)
    index_best_fit = [i for i, j in enumerate(losses_container) if j == best_fit][0]
    Loss_error.append(best_fit)
    L2_error.append(errors_container[index_best_fit])




# Comparison with theoretical value
#plt.subplot(2,1,1)
plt.semilogy(width, L2_error, 'bo--', linewidth=2, markersize=8)
q_plot = np.linspace(width[0]-1, max(width)+1,100)

def theoretical_convergence_rate(q, a,b):
    return a + b*q**(-0.5)
p_opt, _ = curve_fit(theoretical_convergence_rate, width, L2_error)

error_th = theoretical_convergence_rate(q_plot, p_opt[0], p_opt[1])
plt.semilogy(q_plot, error_th, 'k', linewidth=2)
plt.legend([r'Neural network solver', r'Theoretical $O(q^{-1/2})$'])
plt.title(r'$||\Psi_t-\Psi_a||_{L^2}$')
plt.xlabel('q')

# Accuracy w.r.t. the differential operator
# plt.subplot(2,1,2)
# plt.semilogy(width, Loss_error, 'bo--', linewidth=2, markersize=12)
# plt.xlabel('q')
# plt.title('r')



# Save results
L2_array = np.array(L2_error)
Loss_array = np.array(Loss_error)
width_array = np.array(width)

array_errors = np.stack((width_array, L2_array, Loss_array), axis=0).T 
np.savetxt('Error_width.txt', array_errors)

plt.show()






'''
----------------------------------------------------
|  Investigation of the number of training points  |
----------------------------------------------------
'''
print('\n\n========================================\nInvestigating the number of training points\n========================================')

Number_of_Training_Pts = [5,10,15,20,30,50]
q = 20 # Size of hidden layer

# Build the network
layers = (q,)
model = Model(f = F, trial_solution = Trial, Layers=layers, max_iter=max_iter, reg=reg)

# thetas_container = []
L2_error = []
Loss_error = []

for n in Number_of_Training_Pts:

    print('Training with n = {}....'.format(n))
    
    training_set = np.linspace(Interval[0], Interval[1], n) # training set

    thetas_container = [] # for fitted parameters
    losses_container = [] # for the associated loss
    errors_container = [] # for the L2 error

    for i in range(N_Init):

        model.fit(training_set) # The fit method randomly initialize the parameters
    
        # thetas_container.append(model.thetas)
        losses_container.append(model.residual)

        Phi_t, _ = model.phi(x)
        L2_norm = np.sqrt(np.sum(np.abs(Phi_t-Phi_a)**2))
        errors_container.append(L2_norm)

    best_fit = min(losses_container)
    index_best_fit = [i for i, j in enumerate(losses_container) if j == best_fit][0]
    Loss_error.append(best_fit)
    L2_error.append(errors_container[index_best_fit])


plt.semilogy(Number_of_Training_Pts, L2_error, 'bo--',
             linewidth=2, markersize=8)
plt.semilogy(Number_of_Training_Pts, Loss_error,
             'r+--', linewidth=2, markersize=8)

plt.xlabel('# of training points')
plt.legend([r'$L^2$ norm', r'$\mathcal{L}\Psi_t-f$'])
plt.title('Convergence w.r.t the number of training points')
plt.show()

# Save results
L2_array = np.array(L2_error)
Loss_array = np.array(Loss_error)
N_array = np.array(Number_of_Training_Pts)

array_errors = np.stack((N_array, L2_array, Loss_array), axis=0).T 
np.savetxt('Error_training_set.txt', array_errors)

