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


# # The obective function. More precisely, the coefficients of the objective
# # function. Note that we are casting to floats.
objective = np.concatenate((hst,hrt,hdt,hit,np.array(Ksm*t),np.array(Krs*t),np.array(Kdr*t),np.array(Kid*t),Pjt_obj,np.array([0]*len(wijt.flatten()))),axis=None)

# # Lower bounds. Since these are all zero, we could simply not pass them in as
# # all zeroes is the default.
lower_bounds = [0 for p in range(len(objective))]

# # Upper bounds. The default here would be cplex.infinity, or 1e+20.
upper_bounds = [cplex.infinity for p in range(len(objective))]

problem.variables.add(obj = objective,
                       lb = lower_bounds,
                      ub = upper_bounds,
                      names = names.tolist())

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

constraint_names = np.concatenate((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11),axis=None).tolist()

#Defining the constraints

constraint_2 = []
for T in range(1,t+1):
    for R in range(1,r+1):

        if(T==1):
            I = np.array([Irt[T-1][R-1]])
        else:
            I = np.array([Irt[T-2][R-1],Irt[T-1][R-1]])
        qs = np.array([qrst[T-1][S-1][R-1] for S in range(1,s+1)])
        qd = np.array([qdrt[T-1][R-1][D-1] for D in range(1,d+1)])
        list1 = np.concatenate((I,qs,qd),axis=None).tolist()
        if(T==1):
            list2 = np.concatenate((np.array([-1]),np.ones(s),-1*np.ones(d)),axis=None).tolist()
        else:
            list2 = np.concatenate((np.array([1,-1]),np.ones(s),-1*np.ones(d)),axis=None).tolist()
        constraint = [[list1,list2]]
        constraint_2.extend(constraint)

constraint_1 = []
for T in range(1,t+1):
    for S in range(1,s+1):

        if(T==1):
            I = np.array([Ist[T-1][S-1]])
        else:
            I = np.array([Ist[T-2][S-1],Ist[T-1][S-1]])
        qm = np.array([qsmt[T-1][M-1][S-1] for M in range(1,m+1)])
        qr = np.array([qrst[T-1][S-1][R-1] for R in range(1,r+1)])
        list1 = np.concatenate((I,qm,qr),axis=None).tolist()
        if(T==1):
            list2 = np.concatenate((np.array([-1]),np.ones(m),-1*np.ones(r)),axis=None).tolist()
        else:
            list2 = np.concatenate((np.array([1,-1]),np.ones(m),-1*np.ones(r)),axis=None).tolist()
        constraint = [[list1,list2]]
        constraint_1.extend(constraint)

constraint_3 = []
for T in range(1,t+1):
    for D in range(1,d+1):

        if(T==1):
            I = np.array([Idt[T-1][D-1]])
        else:
            I = np.array([Idt[T-2][D-1],Idt[T-1][D-1]])
        qr = np.array([qdrt[T-1][R-1][D-1] for R in range(1,r+1)])
        qi = np.array([qidt[T-1][D-1][I-1] for I in range(1,i+1)])
        list1 = np.concatenate((I,qr,qi),axis=None).tolist()
        if(T==1):
            list2 = np.concatenate((np.array([-1]),np.ones(r),-1*np.ones(i)),axis=None).tolist()
        else:
            list2 = np.concatenate((np.array([1,-1]),np.ones(r),-1*np.ones(i)),axis=None).tolist()
        constraint = [[list1,list2]]
        constraint_3.extend(constraint)

constraint_4 = []
for T in range(1,t+1):
    for I in range(1,i+1):

        if(T==1):
            II = np.array([Iit[T-1][I-1]])
        else:
            II = np.array([Iit[T-2][I-1],Iit[T-1][I-1]])
        qd = np.array([qidt[T-1][D-1][I-1] for D in range(1,d+1)])
        wj = np.array([wijt[T-1][J-1][I-1] for J in range(1,j+1)])
        list1 = np.concatenate((II,qd,wj),axis=None).tolist()
        if(T==1):
            list2 = np.concatenate((np.array([-1]),np.ones(d),-1*np.ones(j)),axis=None).tolist()
        else:
            list2 = np.concatenate((np.array([1,-1]),np.ones(d),-1*np.ones(j)),axis=None).tolist()
        constraint = [[list1,list2]]
        constraint_4.extend(constraint)


constraint_5 = []
for T in range(1,t+1):
    for J in range(1,j+1):
        for I in range(1,i+1):
            w=[[[wijt[T-1][J-1][I-1]],[1.0]]]
            constraint_5.extend(w)

constraint_6 = []
for T in range(1,t+1):
    for J in range(1,j+1):
        for I in range(1,i+1):
            list1=[wijt[T-1][J-1][I-1],sijt[T-1][J-1][I-1]]
            list2=[1.0,1.0]
            constraint=[[list1,list2]]
            constraint_6.extend(constraint)

constraint_7 = []
for T in range(1,t+1):
    for S in range(1,s+1):
        w=[[[Ist[T-1][S-1]],[1.0]]]
        constraint_7.extend(w)

constraint_8 = []
for T in range(1,t+1):
    for R in range(1,r+1):
        w=[[[Irt[T-1][R-1]],[1.0]]]
        constraint_8.extend(w)

constraint_9 = []
for T in range(1,t+1):
    for D in range(1,d+1):
        w=[[[Idt[T-1][D-1]],[1.0]]]
        constraint_9.extend(w)

constraint_10 = []
for T in range(1,t+1):
    for I in range(1,i+1):
        w=[[[Iit[T-1][I-1]],[1.0]]]
        constraint_10.extend(w)

constraint_11 = []
for T in range(1,t+1):
    for M in range(1,m+1):
        qs = np.array([qsmt[T-1][M-1][S-1] for S in range(1,s+1)]).tolist()
        qs_num = np.ones(s).tolist()
        constraint = [[qs,qs_num]]
        constraint_11.extend(constraint)

constraints = []
for constraint in [constraint_1,constraint_2,constraint_3,constraint_4,constraint_5,constraint_6,constraint_7,constraint_8,constraint_9,constraint_10,constraint_11]:
    constraints.extend(constraint)


c1_rhs = 0*np.ones(len(c1.flatten()))
c2_rhs = 0*np.ones(len(c2.flatten()))
c3_rhs = 0*np.ones(len(c3.flatten()))
c4_rhs = 0*np.ones(len(c4.flatten()))

dijt = [[[(1-J)*550+J*425 for I in range(1,i+1)] for J in range(j)] for T in range(1,t+1)]
c5_rhs =  np.array(dijt).flatten()
c6_rhs = c5_rhs
c7_rhs = 56250*np.ones(s*t)
c8_rhs = 28125*np.ones(r*t)
c9_rhs = 2031*np.ones(d*t)
c10_rhs = 250*np.ones(i*t)
Bmt = [[M for M in [12000,15000,11000]] for T in range(t)]
c11_rhs = np.array(Bmt).flatten()

rhs = np.concatenate((c1_rhs,c2_rhs,c3_rhs,c4_rhs,c5_rhs,c6_rhs,c7_rhs,c8_rhs,c9_rhs,c10_rhs,c11_rhs),axis=None).tolist()
# # So far we haven't added a right hand side, so we do that now. Note that the
# # first entry in this list corresponds to the first constraint, and so-on.
# rhs = [75.0, 160.0]

# # We need to enter the senses of the constraints. That is, we need to tell Cplex
# # whether each constraint should be treated as an upper-limit (≤, denoted "L"
# # for less-than), a lower limit (≥, denoted "G" for greater than) or an equality
# # (=, denoted "E" for equality)
# constraint_senses = [ "L", "L" ]

l1 = np.array(["E" for g in range((s+r+d+i)*t)])
l2 = np.array(["L" for g in range(i*j*t)])
l3 = np.array(["E" for g in range(i*j*t)])
l4 = np.array(["L" for g in range((s+r+d+i)*t)])
l5 = np.array(["L" for g in range(m*t)])

constraint_senses = np.concatenate((l1,l2,l3,l4,l5),axis=None).tolist()



# # Note that we can actually set senses as a string. That is, we could also use
# #     constraint_senses = "LL"
# # to pass in our constraints

# # And add the constraints
problem.linear_constraints.add(lin_expr = constraints,
                               senses = constraint_senses,
                               rhs = rhs,
                               names = constraint_names)

# # Solve the problem
problem.solve()
answers = problem.solution.get_values()
for i in range(len(names)):
    print(names[i]," - ", end =" ") 
    print(answers[i])
# # And print the solutions
#print(problem.solution.get_values())