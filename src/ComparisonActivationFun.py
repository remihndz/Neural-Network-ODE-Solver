'''
------------------------------------------------------------------------
|  Investigate the effects of the activation function on the accuracy  |
------------------------------------------------------------------------
'''


import numpy as np
from NeuralNetworkSolver import Model
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


print('=========================================================\nInvestigating the activation function of the hidden layer\n=========================================================')


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

Problem  = Problems[Problem_number]
F, Trial, Solution, Interval = Problem
x = np.linspace(Interval[0], Interval[1], 1000)
Phi_a = Solution(x)


N_Init = 30 # Number of random initialisation of the parameter guess for fitting 
Number_of_Training_Points = 20
training_set = np.linspace(Interval[0], Interval[1], Number_of_Training_Points)
max_iter = 50
layers = (15,)


activation_functions = ['sigmoid','tanh','softplus']
fitted_thetas = []
L2_errors = []
Loss_errors = []
Mean_L2 = []
Mean_Loss = []
Std_L2 = []
Std_Loss = []


for activation_fun in activation_functions:

    model = Model(f = F, trial_solution = Trial, Layers=layers, max_iter=max_iter,
                  activation=activation_fun)

    # Try fitting with different initial guesses
    # to avoid local minima
    
    thetas_container = [] # for fitted parameters
    losses_container = [] # for the associated loss
    errors_container = [] # for the L2 error

    for i in range(N_Init):
        model.fit(training_set) # The fit method randomly initialize the parameters
    
        thetas_container.append(model.thetas)
        losses_container.append(model.residual)

        Phi_t, _ = model.phi(x)
        L2_norm = np.sqrt(np.sum(np.abs(Phi_t-Phi_a)**2))
        errors_container.append(L2_norm)

    best_fit = min(losses_container)
    index_best_fit = [i for i, j in enumerate(losses_container) if j == best_fit][0]
    Loss_errors.append(best_fit)
    L2_errors.append(errors_container[index_best_fit])
    fitted_thetas.append(thetas_container[index_best_fit])

    
    Mean_L2.append(np.mean(L2_errors))
    Std_L2.append(np.std(L2_errors))
    Mean_Loss.append(np.mean(Loss_errors))
    Std_Loss.append(np.std(L2_errors))
    

print("Best fits for each activation function:")
print ("{:<10}   {:<18}   {:<18}".format('Function', 'L2 norm(+-std)', 'Loss(+-std)'))

for fun,norm,stdnorm, loss, stdloss in zip(activation_functions, Mean_L2, Std_L2, Mean_Loss, Std_Loss):
    print ("{:<10}:  {:<.2e}(+-{:<.2e}) {:<.2e}(+-{:<.2e})".format(fun, norm, stdnorm, loss, stdloss))

fig = plt.figure(figsize=(8,6))
ax  = plt.axes()
ax.set_yscale("log")
ax.set_xlabel('Activation function', fontsize=15)
ax.set_ylabel('Error (log scale)', fontsize=15)
x = np.arange(len(activation_functions))
width = 0.25

ax.bar(x-width/2, Mean_L2, width, label=r'$L^2$ error', yerr=Std_L2, capsize=4)
ax.bar(x+width/2, Mean_Loss, width, label=r'Loss', yerr=Std_Loss, capsize=4)

# ax.errorbar(x, Mean_L2, Std_L2, linestyle='None', label=r'$L^2$ accuracy', marker='s',
#              capsize=4, capthick=2, barsabove=True)
# ax.errorbar(x, Mean_Loss, Std_Loss, linestyle='None', label='Loss', marker='o',
#              capsize=4, capthick=2, barsabove=True)

plt.ylim(bottom=1e-8)
ax.set_xticks(x)
ax.set_xticklabels(activation_functions, fontsize=12)

plt.title('Comparison of different activation functions\nfor a single layered network', fontsize=18)
plt.legend()
plt.show()             
           
