import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

n = 1
nb_training_points = 200
epochs = range(200)
learning_rate=0.1
X = np.linspace(0.0,1.0, nb_training_points)
X = X.reshape((-1,n))
X = tf.convert_to_tensor(X, np.float32)


def Pb1(x,y):
    return x**3 + 2*x + x**2*(1+3*x*x)/(1+x+x**3) - y*(x+(1+3*x*x)/(1+x+x**3))
def Sol1(x):
    return np.exp(-x**2/2)/(1+x+x**3) + x**2
def Trial1(x,N,dN):
    return 1 + x*N, N+x*dN


# Initialize the network
optimizer = tf.keras.optimizers.Adam(learning_rate)
net = keras.Sequential()
net.add(Dense(16, activation='sigmoid'))
#net.add(Dense(16, activation='sigmoid'))
net.add(Dense(1))

# Initializes the first layer's shape
x = tf.ones((1,1))
y = net(x)



# The network can now be used to make predictions
def Predictions(x):
    with tf.GradientTape(
            watch_accessed_variables=False) as tape:
        tape.watch(x)
        N = net(x)
    dN = tape.gradient(N,x)
    return N,dN

def Loss(X):
    N,dN = Predictions(X)
    y,dy = 1 + x*N, N+x*dN

    total_loss = 0.0
    for xi,yi,dyi in zip(X,y,dy):
        total_loss += (dyi-Pb1(xi,yi))**2
    return total_loss

def train(net, X):
    with tf.GradientTape() as tape:
        current_loss = Loss(X)

    grad = tape.gradient(current_loss, net.trainable_weights)

    optimizer.apply_gradients(zip(grad, net.trainable_weights)) 

def training_loop(net, X):    
    for epoch in epochs:
        train(net, X)

training_loop(net, X)



def Phi(x):
    N, dN = Predictions(x)
    return 1 + x*N, N+x*dN

test_x = np.linspace(0.0,1.0,30)
sol = Sol1(test_x)
dsol = 0*test_x
test_x = test_x.reshape((-1,n))
test_x = tf.convert_to_tensor(test_x, np.float32)

def Compare():
    
    phi, dphi = Phi(test_x)
    plt.plot(test_x.numpy(),phi,  label='Network solution')
    plt.plot(test_x.numpy(),dphi, label='Network derivative')
    plt.plot(test_x.numpy(),sol,  label='Exact solution')
    plt.plot(test_x.numpy(),dsol, label='Exact derivative')
    plt.legend()
    plt.xlabel('x')
    plt.show()

Compare()
