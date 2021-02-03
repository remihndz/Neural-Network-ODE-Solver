# Neural Network Solver for Differential Equations
Implementation of the solver of differential equations presented by Lagaris et. al. ([see original paper](https://doi.org/10.1109/72.712178)).  
This method has the advantages of:
  * providing a meshless solution to ordinary and partial differential equations;
  * the solution (and its derivatives) can easily be used for post-processing.  
  * the use of simple feedforward networks, even single layered with few parameters provides accurate solutions;
  * the computationnally intensive optimization can be fastened greatly by providing the (exact) derivatives of the loss function or by making use of algorithm such as backpropagation. 

In order to solve the general differential equation:  
<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Clarge%20%5CLarge%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20G%28%5Cvec%20x%2C%20f%28%5Cvec%20x%29%2C%20%5Cnabla%20f%28%5Cvec%20x%29%2C%20%5Cnabla%5E2%28%5Cvec%20x%29%29%20%3D%200%2C%20%26%20%5Cqquad%20%5Cvec%20x%5Cin%5COmega%5Csubset%5Cmathbb%20R%5Ed%2C%5C%5C%20f%28%5Cvec%20x%29%20%3D%20g%28%5Cvec%20x%29%2C%20%26%20%5Cqquad%20%5Cvec%20x%5Cin%5CGamma_D%2C%5C%5C%20%5Cnabla%20f%28%5Cvec%20x%29%5Ccdot%5Cvec%20n%20%3D%20h%28%5Cvec%20x%29%2C%20%26%20%5Cqquad%20%5Cvec%20x%5Cin%5CGamma_N%2C%20%5Cend%7Bmatrix%7D%5Cright.">
</p>

we first define an appropriate (regarding the boundary conditions) trial function:  
<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Clarge%20%5CLarge%20%5Cvarphi%28%5Cvec%20x%29%20%3D%20A%28%5Cvec%20x%29%20&plus;%20B%28%5Cvec%20x%29%5CPsi%28%5Cvec%20x%29">
</p>
where A is taken to satisfy the boundary conditions exactly and B is zero on the boundary. The function Ψ is the output of a neural network.   
For a discrete set of points S in Ω, the network's parameters σ are trained to minimize:  
<p align="center">
<img src="https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Clarge%20%5CLarge%20J%28%5Csigma%29%20%3D%20%5Csum_%7B%5Cvec%20x%5Cin%20S%7D%20G%28%5Cvec%20x%2C%20%5Cphi%28%5Cvec%20x%29%2C%20%5Cnabla%5Cphi%28%5Cvec%20x%29%2C%20%5Cnabla%5E2%5Cphi%28%5Cvec%20x%29%29%5E2">
</p>
