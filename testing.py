#/usr/bin/env python3

import cplex
import numpy as np

# Create an instance of a linear problem to solve
problem = cplex.Cplex()

# We want to find a minimum of our objective function
problem.objective.set_sense(problem.objective.sense.minimize)
#test
j = 2  #Customer sub index
m = 3  #Manufacturer sub index
s = 1  #State sub index
r = 3  #Region sub index
d = 5  #District sub index
i = 10 #Clinic sub index
t = 4  #Time sub index

#It has been stored time wise. For a given time, we placed all the respective centers adjacently. 
#I : Inventory
#q : Delivery quantity
#s : shortages
#w : consumption

Ist = np.array([["I(s,t)("+str(S)+","+str(T)+")" for S in range(1,s+1)] for T in range(1,t+1)]) 
Irt = np.array([["I(r,t)("+str(R)+","+str(T)+")" for R in range(1,r+1)] for T in range(1,t+1)])
Idt = np.array([["I(d,t)("+str(D)+","+str(T)+")" for D in range(1,d+1)] for T in range(1,t+1)])
Iit = np.array([["I(i,t)("+str(I)+","+str(T)+")" for I in range(1,i+1)] for T in range(1,t+1)])

qdrt = np.array([[["q(d,r,t)("+str(D)+","+str(R)+","+str(T)+")" for D in range(1,d+1)] for R in range(1,r+1)] for T in range(1,t+1)])
qsmt = np.array([[["q(s,m,t)("+str(S)+","+str(M)+","+str(T)+")" for S in range(1,s+1)] for M in range(1,m+1)] for T in range(1,t+1)])
qrst = np.array([[["q(r,s,t)("+str(R)+","+str(S)+","+str(T)+")" for R in range(1,r+1)] for S in range(1,s+1)] for T in range(1,t+1)])
qidt = np.array([[["q(i,d,t)("+str(I)+","+str(D)+","+str(T)+")" for I in range(1,i+1)] for D in range(1,d+1)] for T in range(1,t+1)])

sijt = np.array([[["s(i,j,t)("+str(I)+","+str(J)+","+str(T)+")" for I in range(1,i+1)] for J in range(1,j+1)] for T in range(1,t+1)])
wijt = np.array([[["w(i,j,t)("+str(I)+","+str(J)+","+str(T)+")" for I in range(1,i+1)] for J in range(1,j+1)] for T in range(1,t+1)])

names = np.concatenate((Ist,Irt,Idt,Iit,qdrt,qsmt,qrst,qidt,sijt,wijt),axis=None)

#Transportation cost per unit
Ksm = [[3],[2.5],[5]]
Krs = [0.3,0.4,0.37]
Kdr = [[0.4,0.5,0.45],[0.8,0.75,0.78],[0.9,0.85,0.88],[0.4,0.55,0.51],[0.33,0.61,0.44]]
Kid = [[0.43,0.51,0.49,0.51,0.52],[0.79,0.77,0.78,0.72,0.71],[0.35,0.33,0.43,0.35,0.46],[1.07,1.07,1.11,1.12,1.12],[0.87,0.87,0.87,0.87,0.87],[0.36,0.43,0.4,0.42,0.38],[1.11,1.18,1.14,1.21,1.16],[0.95,0.92,0.94,0.98,0.99],[0.7,0.66,0.61,0.63,0.58],[0.92,0.93,0.95,0.89,1]]

#Shortage costs
Pjt = [[0 for J in range(j)] for T in range(t)]
for T in range(t):
    Pjt[T][0] = 750000
    Pjt[T][1] = 650000
Pjt_obj = np.array([[[Pjt[T][J] for I in range(i)] for J in range(j)] for T in range(t)])

#Inventory holding costs
hrt = [[0 for R in range(r)] for T in range(t)]
hdt = [[0 for D in range(d)] for T in range(t)]
hit = [[0 for I in range(i)] for T in range(t)]
hst = [[0.3],[0.3],[0.3],[0.3]] 
for T in range(t):
    hrt[T][0] = 0.3
    hrt[T][1] = 0.4
    hrt[T][2] = 0.35
for T in range(t):
    hdt[T][0] = 0.4
    hdt[T][1] = 0.44
    hdt[T][2] = 0.38
    hdt[T][3] = 0.48
    hdt[T][4] = 0.42
for T in range(t):
    hit[T][0] = 0.5
    hit[T][1] = 0.46
    hit[T][2] = 0.44
    hit[T][3] = 0.51
    hit[T][4] = 0.48
    hit[T][5] = 0.38
    hit[T][6] = 0.47
    hit[T][7] = 0.55
    hit[T][8] = 0.53
    hit[T][9] = 0.50

print(len(names))


# # The obective function. More precisely, the coefficients of the objective
# # function. Note that we are casting to floats.
objective = np.concatenate((hst,hrt,hdt,hit,np.array(Ksm*t),np.array(Krs*t),np.array(Kdr*t),np.array(Kid*t),Pjt_obj,np.array([0]*len(wijt.flatten()))),axis=None)
print(len(objective))


# # Lower bounds. Since these are all zero, we could simply not pass them in as
# # all zeroes is the default.
lower_bounds = [0 for p in range(len(objective))]

# # Upper bounds. The default here would be cplex.infinity, or 1e+20.
upper_bounds = [cplex.infinity for p in range(len(objective))]

problem.variables.add(obj = objective,
                       lb = lower_bounds,
                      ub = upper_bounds,
                      names = names)

# # Constraints

# # Constraints are entered in two parts, as a left hand part and a right hand
# # part. Most times, these will be represented as matrices in your problem. 

# # First, we name the constraints

# # The actual constraints are now added. Each constraint is actually a list
# # consisting of two objects, each of which are themselves lists. The first list
# # represents each of the variables in the constraint, and the second list is the
# # coefficient of the respective variable. Data is entered in this way as the
# # constraints matrix is often sparse.

# # The first constraint is entered by referring to each variable by its name
# # (which we defined earlier). This then represents "3x + y - z"
# first_constraint = [["x", "y", "z"], [3.0, 1.0, -1.0]]
# # In this second constraint, we refer to the variables by their indices. Since
# # "x" was the first variable we added, "y" the second and "z" the third, this
# # then represents 3x + 4y + 4z
# second_constraint = [[0, 1, 2], [3.0, 4.0, 4.0]]

#Inventory balance equations
c1 = np.array([["c1(s,t)("+str(S)+","+str(T)+")" for S in range(1,s+1)] for T in range(1,t+1)])
c2 = np.array([["c2(r,t)("+str(R)+","+str(T)+")" for R in range(1,r+1)] for T in range(1,t+1)])
c3 = np.array([["c3(d,t)("+str(D)+","+str(T)+")" for D in range(1,d+1)] for T in range(1,t+1)])
c4 = np.array([["c4(i,t)("+str(I)+","+str(T)+")" for I in range(1,i+1)] for T in range(1,t+1)])

#Consumption bounded by demand
c5 = np.array([[["c5(i,j,t)("+str(I)+","+str(J)+","+str(T)+")" for I in range(1,i+1)] for J in range(1,j+1)] for T in range(1,t+1)])

#Consumption balance
c6 = np.array([[["c6(i,j,t)("+str(I)+","+str(J)+","+str(T)+")" for I in range(1,i+1)] for J in range(1,j+1)] for T in range(1,t+1)])

#Inventory capacity constraints
c7 = np.array([["c7(s,t)("+str(S)+","+str(T)+")" for S in range(1,s+1)] for T in range(1,t+1)])
c8 = np.array([["c8(r,t)("+str(R)+","+str(T)+")" for R in range(1,r+1)] for T in range(1,t+1)])
c9 = np.array([["c9(d,t)("+str(D)+","+str(T)+")" for D in range(1,d+1)] for T in range(1,t+1)])
c10 = np.array([["c10(i,t)("+str(I)+","+str(T)+")" for I in range(1,i+1)] for T in range(1,t+1)])

#Production capacity constraints at manufacturer end
c11 = np.array([["c11(i,t)("+str(M)+","+str(T)+")" for M in range(1,m+1)] for T in range(1,t+1)])

constraint_names = np.concatenate((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11),axis=None)

#Defining the constraints

constraint_5 = (np.array([[["w(i,j,t)("+str(I)+","+str(J)+","+str(T)+")" for I in range(1,i+1)] for J in range(1,j+1)] for T in range(1,t+1)]),np.ones(i*j*t));
constraint_6 = ([np.array([[["w(i,j,t)("+str(I)+","+str(J)+","+str(T)+")" for I in range(1,i+1)] for J in range(1,j+1)] for T in range(1,t+1)]),np.array([[["s(i,j,t)("+str(I)+","+str(J)+","+str(T)+")" for I in range(1,i+1)] for J in range(1,j+1)] for T in range(1,t+1)])],[np.ones(i*j*t),np.ones(i*j*t)]);
constraint_7 = (np.array([["I(s,t)("+str(S)+","+str(T)+")" for S in range(1,s+1)] for T in range(1,t+1)]),np.ones(s*t));
constraint_8 = (np.array([["I(r,t)("+str(R)+","+str(T)+")" for R in range(1,r+1)] for T in range(1,t+1)]),np.ones(r*t));
constraint_9 = (np.array([["I(d,t)("+str(D)+","+str(T)+")" for D in range(1,d+1)] for T in range(1,t+1)]),np.ones(d*t));
constraint_10 = (np.array([["I(i,t)("+str(I)+","+str(T)+")" for I in range(1,i+1)] for T in range(1,t+1)]),np.ones(i*t));
# # So far we haven't added a right hand side, so we do that now. Note that the
# # first entry in this list corresponds to the first constraint, and so-on.
# rhs = [75.0, 160.0]

# # We need to enter the senses of the constraints. That is, we need to tell Cplex
# # whether each constrains should be treated as an upper-limit (≤, denoted "L"
# # for less-than), a lower limit (≥, denoted "G" for greater than) or an equality
# # (=, denoted "E" for equality)
# constraint_senses = [ "L", "L" ]

# # Note that we can actually set senses as a string. That is, we could also use
# #     constraint_senses = "LL"
# # to pass in our constraints

# # And add the constraints
# problem.linear_constraints.add(lin_expr = constraints,
#                                senses = constraint_senses,
#                                rhs = rhs,
#                                names = constraint_names)

# # Solve the problem
# problem.solve()

# # And print the solutions
# print(problem.solution.get_values())
