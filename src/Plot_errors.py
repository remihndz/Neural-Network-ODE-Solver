import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

error_q = np.loadtxt('Error_width.txt')
error_TPts  = np.loadtxt('Error_training_set.txt')


# Comparison with theoretical value

q = error_q[:,0]
L2_error = error_q[:,1]
Loss = error_q[:,2]


fig1 = plt.figure(1)
plt.semilogy(q, L2_error, 'bo--', linewidth=2, markersize=8)
q_plot = np.linspace(q[0]-1, max(q)+1,100)

def theoretical_convergence_rate(q, a,b):
        return a + b*q**(-0.5)
p_opt, _ = curve_fit(theoretical_convergence_rate, q, L2_error)

error_th = theoretical_convergence_rate(q_plot, p_opt[0], p_opt[1])
plt.semilogy(q_plot, error_th, 'k', linewidth=2)
plt.semilogy(q, Loss,
             'r+--', linewidth=2, markersize=8)
plt.legend([r'Neural network solver', r'Theoretical $O(q^{-1/2})$',
            r'$\mathcal{L}\Psi_t-f$'])
plt.title(r'Sensitivity to the number of neurons in\na single layer network')
plt.xlabel('q')


# Comparison with suggested exponential convergence

fig2 = plt.figure(2)
n = error_TPts[:,0]
n_plot = np.linspace(n[0], n[-1], 1000)

L2_error = error_TPts[:,1]
Loss = error_TPts[:,2]

def exponential_convergence(n, a,b):
    return a*np.exp(-b*n)
error_th = exponential_convergence(n_plot, p_opt[0], p_opt[1])

plt.semilogy(n, L2_error, 'bo--',
             linewidth=2, markersize=8)
plt.semilogy(n_plot, error_th, 'k', linewidth=2)
plt.semilogy(n, Loss,
             'r+--', linewidth=2, markersize=8)
plt.xlabel('# of training points')
plt.legend([r'Normalized $L^2$ norm',r'$O(e^{-cn})$', r'$\mathcal{L}\Psi_t-f$'])
plt.title('Convergence w.r.t the number of training points')
plt.show()
