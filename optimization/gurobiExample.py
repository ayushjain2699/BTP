import numpy as np
import pandas as pd

import gurobipy as gp
from gurobipy import GRB

############INDEX################

j = 2  #Customer sub index
m = 3  #Manufacturer sub index
g = 1  #GMSD index
s = 1  #State sub index
r = 3  #Region sub index
d = 5  #District sub index
i = 10 #Clinic sub index
t = 4  #Time sub index

customers = list(range(1,j+1))
manufacturers = list(range(1,m+1))
gmsd = list(range(1,g+1))
svs = list(range(1,s+1))
rvs = list(range(1,r+1))
dvs = list(range(1,d+1))
clinics = list(range(1,i+1))
time = list(range(1,t+1))

#Transportation cost
diesel_cost = 14
booking_cost = 10000
np.random.seed(133)

Dgm = np.random.normal(1000,250,g*m).reshape(m,g)
Dsg = np.random.normal(1000,250,s*g).reshape(g,s)
Drs = np.random.normal(400,75,r*s).reshape(s,r)
Ddr = np.random.normal(200,25,d*r).reshape(r,d)
Did = np.random.normal(100,25,d*i).reshape(d,i)

cap_veh_gm = 5000
cap_veh_sg = 5000 
cap_veh_rs = 3000
cap_veh_dr = 3000
cap_veh_id = 1000

Kgmt = np.array([[[Dgm[M][G]*diesel_cost+booking_cost for G in range(0,g)] for M in range(0,m)] for T in range(0,t)])
Ksgt = np.array([[[Dsg[G][S]*diesel_cost+booking_cost for S in range(0,s)] for G in range(0,g)] for T in range(0,t)])
Krst = np.array([[[Drs[S][R]*diesel_cost+booking_cost for R in range(0,r)] for S in range(0,s)] for T in range(0,t)])
Kdrt = np.array([[[Ddr[R][D]*diesel_cost+booking_cost for D in range(0,d)] for R in range(0,r)] for T in range(0,t)])
Kidt = np.array([[[Did[D][I]*diesel_cost+booking_cost for I in range(0,i)] for D in range(0,d)] for T in range(0,t)])


#Shortage costs
Pjt = [[0 for J in range(j)] for T in range(t)]
for T in range(t):
    Pjt[T][0] = 750000
    Pjt[T][1] = 650000
#Pjt_obj = np.array([[[Pjt[T][J] for I in range(i)] for J in range(j)] for T in range(t)])

#Inventory holding costs
hgt = [[0.3],[0.3],[0.3],[0.3]] 
hst = [[0.3],[0.3],[0.3],[0.3]] 
hrt = [[0 for R in range(r)] for T in range(t)]
hdt = [[0 for D in range(d)] for T in range(t)]
hit = [[0 for I in range(i)] for T in range(t)]
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
Cgmt = [[[25000 for G in range(g)] for M in range(m)] for T in range(t)]
Csgt = [[[25000 for S in range(s)] for G in range(g)] for T in range(t)]
Crst = [[[25000 for R in range(r)] for S in range(s)] for T in range(t)]
Cdrt = [[[25000 for D in range(d)] for R in range(r)] for T in range(t)]
Cidt = [[[25000 for I in range(i)] for D in range(d)] for T in range(t)]

model = gp.Model('Vaccine_Distribution')
#Inventory
Igt = model.addVars(time,gmsd,vtype=GRB.INTEGER, name="Igt")
Ist = model.addVars(time,svs,vtype=GRB.INTEGER, name="Ist")
Irt = model.addVars(time,rvs,vtype=GRB.INTEGER, name="Irt")
Idt = model.addVars(time,dvs,vtype=GRB.INTEGER, name="Idt")
Iit = model.addVars(time,clinics,vtype=GRB.INTEGER, name="Iit")