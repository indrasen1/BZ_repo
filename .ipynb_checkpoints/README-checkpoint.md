# Readme file for Belousov-Zhabotinsky(BZ) simulation:

Environment file: 'testEnvironment.yml'
Notebook showing example usage: 'BZ_solver_test.ipynb'

# Description of model and discretization:
The BZ_solver.py module implements a simple numerical testbed to approximate the BZ reaction.
The reaction-diffusion system is treated through the time evolution of 2D spatial distributions of three species.
At each point in space, the following model for the BZ system is used where A, B and C represent the concentration of the species:

$$ A + B \rightarrow 2A
B + C \rightarrow 2B
C + A \rightarrow 2C $$

This leads to the following ordinary differential equation:

$$ \frac{dA}{dt} = k_1 AB - k_2 AC $$
and so on for the other species

In addition, we have also included a spatial diffusion term based on a local concentration gradient:

$$ \frac{dA}{dt} = k_1 AB - k_2 AC - d_A \nabla^{2} A $$ 

(the concentration current is proportional to the local concentration gradient, and the rate of change is the negative of the divergence of the concentration current)

These three above equations are solved with regular time discretization and spatial discretization in a Cartesian grid. 
The Laplacian is implemented through convolution with a 3x3 stencil representation. The equations are propagated in time through a 4th order Runge Kutta method.

The functions are implemented through a BZ solver class and example inputs and execution 

# Description of inputs and visualization

The initial spatial distribution of the species can be set with arbitrarily chosen png images. 
Currently, these images are cropped to square. The images need to be placed in the working directory.
Additionally, the species can be forced to adhere to these images at specified time points (not just t=0).
The simulation time step, domain size (in grid points), rate constant and diffusion constant can all be specified.
The step and rate constants should be scalable together: a set of working values is included in the notebook implementation.

Once the system has been propagated, the values across time, space and species are all available. 
The time evolution of any one species can be saved as an mp4 video across the spatial domain or displayed inline. 
The time evolution of all species at a given point can be displayed as a time progressions.

# Potential future updates

Only periodic boundary conditions (in space) are currently available.
Other types of boundary conditions would be interesting to implement. 

Consider saving intermediate results to disk via hdf5 for further propagation, computations and visualization.