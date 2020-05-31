# And print the solutions
#print(problem.solution.get_values())
#/usr/bin/env python3

import cplex
import numpy as np
import xlsxwriter
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

nc = (s+d+r+i)*t+(d*r+s*m+r*s+i*d)*t+2*i*j*t #Total number of continuous variables
nb = (d*r+s*m+r*s+i*d)*t #Total number of binary variables

type_c = np.array(["I" for NC in range(nc)])
type_b = np.array(["B" for NB in range(nb)])
type_for_N = np.array(["I" for j in range((s*m+r*s+d*r+i*d)*t)])
types = np.concatenate((type_c,type_b,type_for_N),axis = None)

#It has been stored time wise. For a given time, we placed all the respective centers adjacently. 
#I : Inventory
#q : Delivery quantity
#s : Shortages
#w : Consumption
#X : Binary variable

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

Xdrt = np.array([[["X(d,r,t)("+str(D)+","+str(R)+","+str(T)+")" for D in range(1,d+1)] for R in range(1,r+1)] for T in range(1,t+1)])
Xsmt = np.array([[["X(s,m,t)("+str(S)+","+str(M)+","+str(T)+")" for S in range(1,s+1)] for M in range(1,m+1)] for T in range(1,t+1)])
Xrst = np.array([[["X(r,s,t)("+str(R)+","+str(S)+","+str(T)+")" for R in range(1,r+1)] for S in range(1,s+1)] for T in range(1,t+1)])
Xidt = np.array([[["X(i,d,t)("+str(I)+","+str(D)+","+str(T)+")" for I in range(1,i+1)] for D in range(1,d+1)] for T in range(1,t+1)])

Nsmt = np.array([[["N(s,m,t)("+str(S)+","+str(M)+","+str(T)+")" for S in range(1,s+1)] for M in range(1,m+1)] for T in range(1,t+1)])
Nrst = np.array([[["N(r,s,t)("+str(R)+","+str(S)+","+str(T)+")" for R in range(1,r+1)] for S in range(1,s+1)] for T in range(1,t+1)])
Ndrt = np.array([[["N(d,r,t)("+str(D)+","+str(R)+","+str(T)+")" for D in range(1,d+1)] for R in range(1,r+1)] for T in range(1,t+1)])
Nidt = np.array([[["N(i,d,t)("+str(I)+","+str(D)+","+str(T)+")" for I in range(1,i+1)] for D in range(1,d+1)] for T in range(1,t+1)])
    
names = np.concatenate((Ist,Irt,Idt,Iit,qdrt,qsmt,qrst,qidt,sijt,wijt,Xdrt,Xsmt,Xrst,Xidt,Nsmt,Nrst,Ndrt,Nidt),axis=None)

#Transportation cost per unit
diesel_cost = 14
booking_cost = 10000
np.random.seed(133)
Dsm = np.random.normal(1000,250,s*m).reshape(m,s)
Drs = np.random.normal(400,75,r*s).reshape(s,r)
Ddr = np.random.normal(200,25,d*r).reshape(r,d)
Did = np.random.normal(100,25,d*i).reshape(d,i)
cap_veh_sm = 5000 
cap_veh_rs = 3000
cap_veh_dr = 3000
cap_veh_id = 1000

Ksmt = np.array([[[Dsm[M][S]*diesel_cost+booking_cost for S in range(0,s)] for M in range(0,m)] for T in range(0,t)])
Krst = np.array([[[Drs[S][R]*diesel_cost+booking_cost for R in range(0,r)] for S in range(0,s)] for T in range(0,t)])
Kdrt = np.array([[[Ddr[R][D]*diesel_cost+booking_cost for D in range(0,d)] for R in range(0,r)] for T in range(0,t)])
Kidt = np.array([[[Did[D][I]*diesel_cost+booking_cost for I in range(0,i)] for D in range(0,d)] for T in range(0,t)])
# Ksm = [[3],[2.5],[5]]   
# Krs = [0.3,0.4,0.37]
# #Kdr = [[0.4,0.5,0.45],[0.8,0.75,0.78],[0.9,0.85,0.88],[0.4,0.55,0.51],[0.33,0.61,0.44]]
# Kdr = [[0.4,0.8,0.9,0.4,0.33],[0.5,0.75,0.85,0.55,0.61],[0.45,0.78,0.88,0.51,0.44]]
# #Kid = [[0.43,0.51,0.49,0.51,0.52],[0.79,0.77,0.78,0.72,0.71],[0.35,0.33,0.43,0.35,0.46],[1.07,1.07,1.11,1.12,1.12],[0.87,0.87,0.87,0.87,0.87],[0.36,0.43,0.4,0.42,0.38],[1.11,1.18,1.14,1.21,1.16],[0.95,0.92,0.94,0.98,0.99],[0.7,0.66,0.61,0.63,0.58],[0.92,0.93,0.95,0.89,1]]
# Kid = [[0.43,0.79,0.35,1.07,0.87,0.36,1.11,0.95,0.7,0.92],[0.51,0.77,0.33,1.07,0.87,0.43,1.18,0.92,0.66,0.93],[0.49,0.78,0.43,1.11,0.87,0.4,1.41,0.94,0.61,0.95],[0.51,0.72,0.35,1.12,0.87,0.42,1.21,0.98,0.63,0.89],[0.52,0.71,0.46,1.12,0.87,0.38,1.16,0.99,0.58,1]]

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

#Ordering costs
Csmt = [[[25000 for S in range(s)] for M in range(m)] for T in range(t)]
Crst = [[[25000 for R in range(r)] for S in range(s)] for T in range(t)]
Cdrt = [[[25000 for D in range(d)] for R in range(r)] for T in range(t)]
Cidt = [[[25000 for I in range(i)] for D in range(d)] for T in range(t)]
# # The obective function. More precisely, the coefficients of the objective function. 
objective = np.concatenate((hst,hrt,hdt,hit,np.zeros(s*m*t),np.zeros(r*s*t),np.zeros(d*r*t),np.zeros(i*d*t),Pjt_obj,np.array([0]*len(wijt.flatten())),Cdrt,Csmt,Crst,Cidt,Ksmt,Krst,Kdrt,Kidt),axis=None)
#print(objective)

# # Lower bounds. Since these are all zero, we could simply not pass them in as all zeroes is the default.
lower_bounds = [0 for p in range(len(objective))]

# # Upper bounds. The default here would be cplex.infinity, or 1e+20.
upper_bounds = [cplex.infinity for p in range(len(objective))]

problem.variables.add(obj = objective,
                       lb = lower_bounds,
                      ub = upper_bounds,
                      names = names.tolist(),
                      types = types)

# # Constraints

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
c11 = np.array([["c11(m,t)("+str(M)+","+str(T)+")" for M in range(1,m+1)] for T in range(1,t+1)])

c12 = np.array([[["c12(m,s,t)("+str(M)+","+str(S)+","+str(T)+")" for M in range(1,m+1)] for S in range(1,s+1)] for T in range(1,t+1)])
c13 = np.array([[["c13(m,s,t)("+str(M)+","+str(S)+","+str(T)+")" for M in range(1,m+1)] for S in range(1,s+1)] for T in range(1,t+1)])
c14 = np.array([[["c14(s,r,t)("+str(S)+","+str(R)+","+str(T)+")" for S in range(1,s+1)] for R in range(1,r+1)] for T in range(1,t+1)])
c15 = np.array([[["c15(s,r,t)("+str(S)+","+str(R)+","+str(T)+")" for S in range(1,s+1)] for R in range(1,r+1)] for T in range(1,t+1)])
c16 = np.array([[["c16(r,d,t)("+str(R)+","+str(D)+","+str(T)+")" for R in range(1,r+1)] for D in range(1,d+1)] for T in range(1,t+1)])
c17 = np.array([[["c17(r,d,t)("+str(R)+","+str(D)+","+str(T)+")" for R in range(1,r+1)] for D in range(1,d+1)] for T in range(1,t+1)])
c18 = np.array([[["c18(d,i,t)("+str(D)+","+str(I)+","+str(T)+")" for D in range(1,d+1)] for I in range(1,i+1)] for T in range(1,t+1)])
c19 = np.array([[["c19(d,i,t)("+str(D)+","+str(I)+","+str(T)+")" for D in range(1,d+1)] for I in range(1,i+1)] for T in range(1,t+1)])
c20 = np.array([["c20(r,t)("+str(R)+","+str(T)+")" for R in range(1,r+1)] for T in range(1,t+1)])
c21 = np.array([["c21(d,t)("+str(D)+","+str(T)+")" for D in range(1,d+1)] for T in range(1,t+1)])
c22 = np.array([["c22(i,t)("+str(I)+","+str(T)+")" for I in range(1,i+1)] for T in range(1,t+1)])

c23 = np.array([[["c23(s,m,t)("+str(S)+","+str(M)+","+str(T)+")" for S in range(1,s+1)] for M in range(1,m+1)] for T in range(1,t+1)])
c24 = np.array([[["c24(s,m,t)("+str(S)+","+str(M)+","+str(T)+")" for S in range(1,s+1)] for M in range(1,m+1)] for T in range(1,t+1)])
c25 = np.array([[["c25(r,s,t)("+str(R)+","+str(S)+","+str(T)+")" for R in range(1,r+1)] for S in range(1,s+1)] for T in range(1,t+1)])
c26 = np.array([[["c26(r,s,t)("+str(R)+","+str(S)+","+str(T)+")" for R in range(1,r+1)] for S in range(1,s+1)] for T in range(1,t+1)])
c27 = np.array([[["c27(d,r,t)("+str(D)+","+str(R)+","+str(T)+")" for D in range(1,d+1)] for R in range(1,r+1)] for T in range(1,t+1)])
c28 = np.array([[["c28(d,r,t)("+str(D)+","+str(R)+","+str(T)+")" for D in range(1,d+1)] for R in range(1,r+1)] for T in range(1,t+1)])
c29 = np.array([[["c29(i,d,t)("+str(I)+","+str(D)+","+str(T)+")" for I in range(1,i+1)] for D in range(1,d+1)] for T in range(1,t+1)])
c30 = np.array([[["c30(i,d,t)("+str(I)+","+str(D)+","+str(T)+")" for I in range(1,i+1)] for D in range(1,d+1)] for T in range(1,t+1)])

constraint_names = np.concatenate((c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30),axis=None).tolist()                          

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

INF = cplex.infinity;

constraint_12 = []
for T in range(1,t+1):
    for S in range(1,s+1):
        for M in range(1,m+1):
            var = [Xsmt[T-1][M-1][S-1],qsmt[T-1][M-1][S-1]]
            coeff = [1,-INF]
            constraint = [[var,coeff]]
            constraint_12.extend(constraint)

constraint_13 = []
for T in range(1,t+1):
    for S in range(1,s+1):
        for M in range(1,m+1):
            var = [Xsmt[T-1][M-1][S-1],qsmt[T-1][M-1][S-1]]
            coeff = [-INF,1]
            constraint = [[var,coeff]]
            constraint_13.extend(constraint)

constraint_14 = []
for T in range(1,t+1):
    for R in range(1,r+1):
        for S in range(1,s+1):
            var = [Xrst[T-1][S-1][R-1],qrst[T-1][S-1][R-1]]
            coeff = [1,-INF]
            constraint = [[var,coeff]]
            constraint_14.extend(constraint)

constraint_15 = []
for T in range(1,t+1):
    for R in range(1,r+1):
        for S in range(1,s+1):
            var = [Xrst[T-1][S-1][R-1],qrst[T-1][S-1][R-1]]
            coeff = [-INF,1]
            constraint = [[var,coeff]]
            constraint_15.extend(constraint)

constraint_16 = []
for T in range(1,t+1):
    for D in range(1,d+1):
        for R in range(1,r+1):
            var = [Xdrt[T-1][R-1][D-1],qdrt[T-1][R-1][D-1]]
            coeff = [1,-INF]
            constraint = [[var,coeff]]
            constraint_16.extend(constraint)

constraint_17 = []
for T in range(1,t+1):
    for D in range(1,d+1):
        for R in range(1,r+1):
            var = [Xdrt[T-1][R-1][D-1],qdrt[T-1][R-1][D-1]]
            coeff = [-INF,1]
            constraint = [[var,coeff]]
            constraint_17.extend(constraint)

constraint_18 = []
for T in range(1,t+1):
    for I in range(1,i+1):
        for D in range(1,d+1):
            var = [Xidt[T-1][D-1][I-1],qidt[T-1][D-1][I-1]]
            coeff = [1,-INF]
            constraint = [[var,coeff]]
            constraint_18.extend(constraint)

constraint_19 = []
for T in range(1,t+1):
    for I in range(1,i+1):
        for D in range(1,d+1):
            var = [Xidt[T-1][D-1][I-1],qidt[T-1][D-1][I-1]]
            coeff = [-INF,1]
            constraint = [[var,coeff]]
            constraint_19.extend(constraint)

constraint_20 = []
for T in range(1,t+1):
    for R in range(1,r+1):
        xs = np.array([Xrst[T-1][S-1][R-1] for S in range(1,s+1)]).tolist()
        xs_num = np.ones(s).tolist()
        constraint = [[xs,xs_num]]
        constraint_20.extend(constraint)

constraint_21 = []
for T in range(1,t+1):
    for D in range(1,d+1):
        xr = np.array([Xdrt[T-1][R-1][D-1] for R in range(1,r+1)]).tolist()
        xr_num = np.ones(r).tolist()
        constraint = [[xr,xr_num]]
        constraint_21.extend(constraint)

constraint_22 = []
for T in range(1,t+1):
    for I in range(1,i+1):
        xd = np.array([Xidt[T-1][D-1][I-1] for D in range(1,d+1)]).tolist()
        xd_num = np.ones(d).tolist()
        constraint = [[xd,xd_num]]
        constraint_22.extend(constraint)


constraint_23 = []
for T in range(1,t+1):
    for M in range(1,m+1):
        for S in range(1,s+1):
            var = [Nsmt[T-1][M-1][S-1],qsmt[T-1][M-1][S-1]]
            coeff = [-1,1/cap_veh_sm]
            constraint = [[var,coeff]]
            constraint_23.extend(constraint)

c23_rhs = 0*np.ones(len(c23.flatten()))

constraint_24 = []
for T in range(1,t+1):
    for M in range(1,m+1):
        for S in range(1,s+1):
            var = [Nsmt[T-1][M-1][S-1],qsmt[T-1][M-1][S-1]]
            coeff = [1,-1/cap_veh_sm]
            constraint = [[var,coeff]]
            constraint_24.extend(constraint)

c24_rhs = ((cap_veh_sm-1)/cap_veh_sm)*np.ones(len(c24.flatten()))

constraint_25 = []
for T in range(1,t+1):
    for S in range(1,s+1):
        for R in range(1,r+1):
            var = [Nrst[T-1][S-1][R-1],qrst[T-1][S-1][R-1]]
            coeff = [-1,1/cap_veh_rs]
            constraint = [[var,coeff]]
            constraint_25.extend(constraint)

c25_rhs = 0*np.ones(len(c25.flatten()))

constraint_26 = []
for T in range(1,t+1):
    for S in range(1,s+1):
        for R in range(1,r+1):
            var = [Nrst[T-1][S-1][R-1],qrst[T-1][S-1][R-1]]
            coeff = [1,-1/cap_veh_rs]
            constraint = [[var,coeff]]
            constraint_26.extend(constraint)

c26_rhs = ((cap_veh_rs-1)/cap_veh_rs)*np.ones(len(c26.flatten()))

constraint_27 = []
for T in range(1,t+1):
    for R in range(1,r+1):
        for D in range(1,d+1):
            var = [Ndrt[T-1][R-1][D-1],qdrt[T-1][R-1][D-1]]
            coeff = [-1,1/cap_veh_dr]
            constraint = [[var,coeff]]
            constraint_27.extend(constraint)

c27_rhs = 0*np.ones(len(c27.flatten()))

constraint_28 = []
for T in range(1,t+1):
    for R in range(1,r+1):
        for D in range(1,d+1):
            var = [Ndrt[T-1][R-1][D-1],qdrt[T-1][R-1][D-1]]
            coeff = [1,-1/cap_veh_dr]
            constraint = [[var,coeff]]
            constraint_28.extend(constraint)

c28_rhs = ((cap_veh_dr-1)/cap_veh_dr)*np.ones(len(c28.flatten()))

constraint_29 = []
for T in range(1,t+1):
    for D in range(1,d+1):
        for I in range(1,i+1):
            var = [Nidt[T-1][D-1][I-1],qidt[T-1][D-1][I-1]]
            coeff = [-1,1/cap_veh_id]
            constraint = [[var,coeff]]
            constraint_29.extend(constraint)

c29_rhs = 0*np.ones(len(c29.flatten()))

constraint_30 = []
for T in range(1,t+1):
    for D in range(1,d+1):
        for I in range(1,i+1):
            var = [Nidt[T-1][D-1][I-1],qidt[T-1][D-1][I-1]]
            coeff = [1,-1/cap_veh_id]
            constraint = [[var,coeff]]
            constraint_30.extend(constraint)

c30_rhs = ((cap_veh_id-1)/cap_veh_id)*np.ones(len(c30.flatten()))


constraints = []
for constraint in [constraint_1,constraint_2,constraint_3,constraint_4,constraint_5,constraint_6,constraint_7,constraint_8,constraint_9,constraint_10,constraint_11,constraint_12,constraint_13,constraint_14,constraint_15,constraint_16,constraint_17,constraint_18,constraint_19,constraint_20,constraint_21,constraint_22,constraint_23,constraint_24,constraint_25,constraint_26,constraint_27,constraint_28,constraint_29,constraint_30]:
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
c10_rhs = 2000*np.ones(i*t)
Bmt = [[M for M in [12000,15000,11000]] for T in range(t)]
c11_rhs = np.array(Bmt).flatten()
c12_rhs = 0*np.ones(m*s*t);
c13_rhs = 0*np.ones(m*s*t);
c14_rhs = 0*np.ones(r*s*t);
c15_rhs = 0*np.ones(r*s*t);
c16_rhs = 0*np.ones(r*d*t);
c17_rhs = 0*np.ones(r*d*t);
c18_rhs = 0*np.ones(i*d*t);
c19_rhs = 0*np.ones(i*d*t);
c20_rhs = np.ones(r*t);
c21_rhs = np.ones(d*t);
c22_rhs = np.ones(i*t);

rhs = np.concatenate((c1_rhs,c2_rhs,c3_rhs,c4_rhs,c5_rhs,c6_rhs,c7_rhs,c8_rhs,c9_rhs,c10_rhs,c11_rhs,c12_rhs,c13_rhs,c14_rhs,c15_rhs,c16_rhs,c17_rhs,c18_rhs,c19_rhs,c20_rhs,c21_rhs,c22_rhs,c23_rhs,c24_rhs,c25_rhs,c26_rhs,c27_rhs,c28_rhs,c29_rhs,c30_rhs),axis=None).tolist()

#Adding constraint senses
l1 = np.array(["E" for g in range((s+r+d+i)*t)])
l2 = np.array(["L" for g in range(i*j*t)])
l3 = np.array(["E" for g in range(i*j*t)])
l4 = np.array(["L" for g in range((s+r+d+i)*t)])
l5 = np.array(["L" for g in range(m*t)])
l6 = np.array(["L" for g in range((m*s+s*r+r*d+d*i)*2*t)])
l7 = np.array(["L" for g in range((r+d+i)*t)])
l8 = np.array(["L" for g in range((m*s+s*r+r*d+d*i)*2*t)])


constraint_senses = np.concatenate((l1,l2,l3,l4,l5,l6,l7,l8),axis=None).tolist()
# print(len(constraint_names))
# print(len(rhs))
# print(len(constraint_senses))
# print(len(constraints))
# And add the constraints
problem.linear_constraints.add(lin_expr = constraints,
                               senses = constraint_senses,
                               rhs = rhs,
                               names = constraint_names)

#Solve the problem
problem.solve()
sol = problem.solution.get_values()
for x in range(len(sol)):
    print(names[x]," = ",round(sol[x]))

# workbook = xlsxwriter.Workbook('C:/Users/Ayush/Desktop/withConstraints.xlsx')
# worksheet = workbook.add_worksheet()
# merge_format = workbook.add_format({
#     'bold': 1,
#     'border': 1,
#     'align': 'center',
#     'valign': 'vcenter'})

# worksheet.merge_range('A1:B1', 'Inventory', merge_format)
# worksheet.merge_range('C1:D1', 'Consumption', merge_format)
# worksheet.merge_range('E1:F1', 'Shortage', merge_format)
# worksheet.merge_range('G1:H1', 'Delivery from M to S', merge_format)
# worksheet.merge_range('I1:J1', 'X from M to S', merge_format)
# worksheet.merge_range('K1:L1', 'Delivery from S to R', merge_format)
# worksheet.merge_range('M1:N1', 'X from S to R', merge_format)
# worksheet.merge_range('O1:P1', 'Delivery from R to D', merge_format)
# worksheet.merge_range('Q1:R1', 'X from R to D', merge_format)
# worksheet.merge_range('S1:T1', 'Delivery from D to I', merge_format)
# worksheet.merge_range('U1:V1', 'X from D to I', merge_format)
# x = 0
# y = 0
# z = 0
# row = 1
# col = 0
# for x in range((s+r+d+i)*t):
#     worksheet.write(row, col, names[x])
#     worksheet.write(row, col + 1, round(sol[x]))
#     row += 1
# row = 1
# for y in range(x+1,x+d*r*t+1):
#     worksheet.write(row, 14, names[y])
#     worksheet.write(row, 15, round(sol[y]))
#     worksheet.write(row, 16, names[y+(d*r+s*m+r*s+i*d)*t+2*i*j*t])
#     worksheet.write(row, 17, round(sol[y+(d*r+s*m+r*s+i*d)*t+2*i*j*t]))
#     row +=1
# row = 1
# for z in range(y+1,y+s*m*t+1):
#     worksheet.write(row, 6, names[z])
#     worksheet.write(row, 7, round(sol[z]))
#     worksheet.write(row, 8, names[z+(d*r+s*m+r*s+i*d)*t+2*i*j*t])
#     worksheet.write(row, 9, round(sol[z+(d*r+s*m+r*s+i*d)*t+2*i*j*t]))
#     row +=1
# row = 1
# x = z
# for y in range(x+1,x+r*s*t+1):
#     worksheet.write(row, 10, names[y])
#     worksheet.write(row, 11, round(sol[y]))
#     worksheet.write(row, 12, names[y+(d*r+s*m+r*s+i*d)*t+2*i*j*t])
#     worksheet.write(row, 13, round(sol[y+(d*r+s*m+r*s+i*d)*t+2*i*j*t]))
#     row +=1
# row = 1
# x = y
# for y in range(x+1,x+i*d*t+1):
#     worksheet.write(row, 18, names[y])
#     worksheet.write(row, 19, round(sol[y]))
#     worksheet.write(row, 20, names[y+(d*r+s*m+r*s+i*d)*t+2*i*j*t])
#     worksheet.write(row, 21, round(sol[y+(d*r+s*m+r*s+i*d)*t+2*i*j*t]))
#     row +=1
# row = 1
# x = y
# for y in range(x+1,x+i*j*t+1):
#     worksheet.write(row, 4, names[y])
#     worksheet.write(row, 5, round(sol[y]))
#     row +=1
# row = 1
# x = y
# for y in range(x+1,x+i*j*t+1):
#     worksheet.write(row, 2, names[y])
#     worksheet.write(row, 3, round(sol[y]))
#     row +=1
# workbook.close()