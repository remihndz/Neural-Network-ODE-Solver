# Neural Network Solver for Differential Equations


## Presentation of the method
------

Implementation of the solver of differential equations presented by Lagaris et. al. ([see original paper](https://doi.org/10.1109/72.712178)).  
This method has the advantages of:
  * providing a meshless solution to ordinary and partial differential equations;
  * the solution (and its derivatives) can easily be used for post-processing.  
  * the use of simple feedforward networks, even single layered with few parameters provides accurate solutions;
  * the computationnally intensive optimization can be fastened greatly by providing the (exact) derivatives of the loss function or by making use of algorithm such as backpropagation. 

In order to solve the general differential equation:  
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\mathcal&space;L\Psi&space;=&space;\mathbf&space;f,&space;\quad&space;\text{in&space;}&space;\Omega\subset{\mathbb&space;R^n}">
</p>

we first define an appropriate (regarding the boundary conditions) trial function:  
<p align="center">
<img src="https://latex.codecogs.com/svg.image?\varphi(\mathbf&space;x,\theta)&space;=&space;A(\mathbf&space;x)&space;&plus;&space;B(\mathbf&space;x)\Phi(\mathbf&space;x,&space;\theta)" title="\varphi(\mathbf x,\theta) = A(\mathbf x) + B(\mathbf x)\Phi(\mathbf x, \theta)">
</p>

where *A* is taken to satisfy the boundary conditions exactly and *B* is zero on the boundary. The function *Φ* is the output of a neural network.   
For a discrete set of points S in Ω, the network's parameters σ are trained to minimize:  
<p align="center">
<img src="https://latex.codecogs.com/svg.image?J(\theta)&space;=&space;\sum_{x_i\in\mathcal&space;S}\bigl(\mathcal&space;L\varphi(x_i,\theta)&space;-&space;\mathbf&space;f(x_i)\bigr)^2">
</p>

Solving the minimization problem can be problematic because of the presence of local minima. For more complicated problem or better accuracy, a random walk on the parameter space for the initial guess may be useful to prevent falling in such pitfall.


## Investigation of the (hyper-)parameters
-----

In abscence of theoretical results for the method, we want to empirically investigate if and how the trial function's accuracy converges to the analytical solution, *ψ*. For this purpose, we set a test problem of which we know the solution and compare it with the numerical solution, *φ*.
To evaluate this, we use both the *L<sup>2</sup>* norm of *ψ-φ* and how well *φ* satisfies the equation, namely *J(θ)*.

Letting *q* be the size of the (single) hidden layer in the network and *n* the number of points in the training set *S*, we suggested the following convergence rates of *q<sup>-0.5</sup>*
<p align="center">
<img src="/images/Error_width.png">
</p>

and *e<sup>-c√n</sup>*
<p align="center">
<img src="/images/Error_training_set.png">
</p>
