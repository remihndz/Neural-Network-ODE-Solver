import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

Error_q = np.loadtxt('Error_width.txt')
Error_n = np.loadtxt('Error_training_set.txt')


width = Error_q[:,0]
L2_error = Error_q[:,1]
Loss_error = Error_q[:,2]

fig1 = plt.figure(1)
plt.semilogy(width, L2_error, 'bo--', linewidth=2, markersize=8)
q_plot = np.linspace(width[0], max(width),100)

def theoretical_convergence_rate(q, a, b):
    return a + b*q**(-0.5)
p_opt, _ = curve_fit(theoretical_convergence_rate, width, L2_error, bounds=[[0,0],[np.inf,np.inf]])
p_opt[1]/=2.

error_th = theoretical_convergence_rate(q_plot, *p_opt)
plt.semilogy(q_plot, error_th, 'k', linewidth=2)

plt.semilogy(width, Loss_error, 'r+--', linewidth=2, markersize=8)

plt.legend([r'$||\varphi_t-\varphi_a||_{L^2}$', r'Theoretical $O(q^{-1/2})$', r'$\mathcal{L}\varphi_t-f$'], fontsize=12)
plt.title('Convergence w.r.t.\nthe width of the network', fontsize=15)
plt.xlabel('q', fontsize=15)



Number_of_Training_Pts = Error_n[:,0]
n_plot = np.linspace(Error_n[0,0]-1, Error_n[-1,0]+1, 100)

L2_error = Error_n[:,1]
Loss_error = Error_n[:,2]

fig2 = plt.figure(2)

def exponential_decay(n, a,b):
    return a*np.exp(b*np.sqrt(n))
# a, b = L2_error[0], np.log(L2_error[-1]/L2_error[0])/(Error_n[-1,0]-Error_n[0,0])
# p_opt = [a,b]

p_opt, _ = curve_fit(exponential_decay, Number_of_Training_Pts, L2_error)

plt.semilogy(Number_of_Training_Pts, L2_error, 'bo--',
             linewidth=2, markersize=8)
plt.semilogy(n_plot, exponential_decay(n_plot, *p_opt),
             'k', linewidth=2)
             
plt.semilogy(Number_of_Training_Pts, Loss_error,
             'r+--', linewidth=2, markersize=8)

plt.xlabel('n', fontsize=15)
plt.legend([r'$L^2$ norm', r'$O(e^{-c\sqrt{n}})$',r'$\mathcal{L}\varphi_t-f$'], fontsize=12)
plt.title('Convergence w.r.t. # of training points', fontsize=15)
plt.show()
