import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

s = CyClpSimplex()

# Add variables
x = s.addVariable('x', 3)
y = s.addVariable('y', 2)

# Create coefficients and bounds
A = np.matrix([[1., 2., 0],[1., 0, 1.]])
B = np.matrix([[1., 0, 0], [0, 0, 1.]])
D = np.matrix([[1., 2.],[0, 1]])
a = CyLPArray([5, 2.5])
b = np.array([5,2.5])
#b = CyLPArray(k)
x_u= CyLPArray([2., 3.5])

# Add constraints
s += A * x <= a
s += 2 <= B * x + D * y <= b
s += y >= 0
s += 1.1 <= x[1:3] <= x_u

# Set the objective function
c = np.array([1., -2., 3.])
c = CyLPArray(c)
s.objective = c * x + 2 * y.sum()

# Solve using primal Simplex
s.primal()
print (s.primalVariableSolution['x'])
print (s.primalVariableSolution['y'])
