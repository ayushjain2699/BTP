import cplex
import numpy as np
import xlsxwriter

j = 2  #Customer sub index
m = 3  #Manufacturer sub index
g = 1  #GMSD index
s = 1  #State sub index
r = 3  #Region sub index
d = 5  #District sub index
i = 10 #Clinic sub index
t = 4  #Time sub index

nc = (g+s+d+r+i)*t+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t #Total number of continuous variables
nb = (g*m+d*r+s*g+r*s+i*d)*t #Total number of binary variables

type_c = np.array(["I" for NC in range(nc)])
type_b = np.array(["B" for NB in range(nb)])
type_for_N = np.array(["I" for KK in range((g*m+s*g+r*s+d*r+i*d)*t)])
types = np.concatenate((type_c,type_b,type_for_N),axis = None)

np.random.seed(133)
Dgm = np.random.normal(1000,250,g*m).reshape(m,g)
Dsg = np.random.normal(1000,250,s*g).reshape(g,s)
Drs = np.random.normal(400,75,r*s).reshape(s,r)
Ddr = np.random.normal(200,25,d*r).reshape(r,d)
Did = np.random.normal(100,25,d*i).reshape(d,i)

Dgm_name = np.array([["D(g,m)("+str(G)+","+str(M)+")" for G in range(1,g+1)] for M in range(1,m+1)])
Dsg_name = np.array([["D(s,g)("+str(S)+","+str(G)+")" for S in range(1,s+1)] for G in range(1,g+1)]) 
Drs_name = np.array([["D(r,s)("+str(R)+","+str(S)+")" for R in range(1,r+1)] for S in range(1,s+1)])
Ddr_name = np.array([["D(d,r)("+str(D)+","+str(R)+")" for D in range(1,d+1)] for R in range(1,r+1)])
Did_name = np.array([["D(i,d)("+str(I)+","+str(D)+")" for I in range(1,i+1)] for D in range(1,d+1)])

distances = np.concatenate((Dgm,Dsg,Drs,Ddr,Did),axis=None).tolist()
D_names = np.concatenate((Dgm_name,Dsg_name,Drs_name,Ddr_name,Did_name),axis=None).tolist()

Igt = np.array([["I(g,t)("+str(G)+","+str(T)+")" for G in range(1,g+1)] for T in range(1,t+1)]) 
Ist = np.array([["I(s,t)("+str(S)+","+str(T)+")" for S in range(1,s+1)] for T in range(1,t+1)]) 
Irt = np.array([["I(r,t)("+str(R)+","+str(T)+")" for R in range(1,r+1)] for T in range(1,t+1)])
Idt = np.array([["I(d,t)("+str(D)+","+str(T)+")" for D in range(1,d+1)] for T in range(1,t+1)])
Iit = np.array([["I(i,t)("+str(I)+","+str(T)+")" for I in range(1,i+1)] for T in range(1,t+1)])

qgmt = np.array([[["q(g,m,t)("+str(G)+","+str(M)+","+str(T)+")" for G in range(1,g+1)] for M in range(1,m+1)] for T in range(1,t+1)])
qsgt = np.array([[["q(s,g,t)("+str(S)+","+str(G)+","+str(T)+")" for S in range(1,s+1)] for G in range(1,g+1)] for T in range(1,t+1)])
qrst = np.array([[["q(r,s,t)("+str(R)+","+str(S)+","+str(T)+")" for R in range(1,r+1)] for S in range(1,s+1)] for T in range(1,t+1)])
qdrt = np.array([[["q(d,r,t)("+str(D)+","+str(R)+","+str(T)+")" for D in range(1,d+1)] for R in range(1,r+1)] for T in range(1,t+1)])
qidt = np.array([[["q(i,d,t)("+str(I)+","+str(D)+","+str(T)+")" for I in range(1,i+1)] for D in range(1,d+1)] for T in range(1,t+1)])

sijt = np.array([[["s(i,j,t)("+str(I)+","+str(J)+","+str(T)+")" for I in range(1,i+1)] for J in range(1,j+1)] for T in range(1,t+1)])
wijt = np.array([[["w(i,j,t)("+str(I)+","+str(J)+","+str(T)+")" for I in range(1,i+1)] for J in range(1,j+1)] for T in range(1,t+1)])

Xgmt = np.array([[["X(g,m,t)("+str(G)+","+str(M)+","+str(T)+")" for G in range(1,g+1)] for M in range(1,m+1)] for T in range(1,t+1)])
Xsgt = np.array([[["X(s,g,t)("+str(S)+","+str(G)+","+str(T)+")" for S in range(1,s+1)] for G in range(1,g+1)] for T in range(1,t+1)])
Xrst = np.array([[["X(r,s,t)("+str(R)+","+str(S)+","+str(T)+")" for R in range(1,r+1)] for S in range(1,s+1)] for T in range(1,t+1)])
Xdrt = np.array([[["X(d,r,t)("+str(D)+","+str(R)+","+str(T)+")" for D in range(1,d+1)] for R in range(1,r+1)] for T in range(1,t+1)])
Xidt = np.array([[["X(i,d,t)("+str(I)+","+str(D)+","+str(T)+")" for I in range(1,i+1)] for D in range(1,d+1)] for T in range(1,t+1)])

Ngmt = np.array([[["N(g,m,t)("+str(G)+","+str(M)+","+str(T)+")" for G in range(1,g+1)] for M in range(1,m+1)] for T in range(1,t+1)])
Nsgt = np.array([[["N(s,g,t)("+str(S)+","+str(G)+","+str(T)+")" for S in range(1,s+1)] for G in range(1,g+1)] for T in range(1,t+1)])
Nrst = np.array([[["N(r,s,t)("+str(R)+","+str(S)+","+str(T)+")" for R in range(1,r+1)] for S in range(1,s+1)] for T in range(1,t+1)])
Ndrt = np.array([[["N(d,r,t)("+str(D)+","+str(R)+","+str(T)+")" for D in range(1,d+1)] for R in range(1,r+1)] for T in range(1,t+1)])
Nidt = np.array([[["N(i,d,t)("+str(I)+","+str(D)+","+str(T)+")" for I in range(1,i+1)] for D in range(1,d+1)] for T in range(1,t+1)])
    
names = np.concatenate((Igt,Ist,Irt,Idt,Iit,qgmt,qsgt,qrst,qdrt,qidt,sijt,wijt,Xgmt,Xsgt,Xrst,Xdrt,Xidt,Ngmt,Nsgt,Nrst,Ndrt,Nidt),axis=None)

x=0
y=0
z=0
a=0
b=0
c=0
e=0
f=0

sol1 = [0 for x in range((g+s+r+d+i)*t)]
sol2 = [1 for y in range(g*m*t)]
sol3 = [2 for y in range(s*g*t)]
sol4 = [3 for a in range(r*s*t)]
sol5 = [4 for b in range(d*r*t)]
sol6 = [5 for c in range(i*d*t)]

sol7 = [6 for e in range(i*j*t)]
sol8= [7 for f in range(i*j*t)]

sol9 = [1 for y in range(g*m*t)]
sol10 = [2 for y in range(s*g*t)]
sol11 = [3 for a in range(r*s*t)]
sol12 = [4 for b in range(d*r*t)]
sol13 = [5 for c in range(i*d*t)]

sol14 = [1 for y in range(g*m*t)]
sol15 = [2 for y in range(s*g*t)]
sol16 = [3 for a in range(r*s*t)]
sol17 = [4 for b in range(d*r*t)]
sol18 = [5 for c in range(i*d*t)]

sol = sol1+sol2+sol3+sol4+sol5+sol6+sol7+sol8+sol9+sol10+sol11+sol12+sol13+sol14+sol15+sol16+sol17+sol18

workbook = xlsxwriter.Workbook('C:/Users/Shanmukhi/Desktop/test.xlsx')
worksheet = workbook.add_worksheet()
merge_format = workbook.add_format({
    'bold': 1,
    'border': 1,
    'align': 'center',
    'valign': 'vcenter'})

worksheet.merge_range('A1:B1', 'Inventory', merge_format)
worksheet.merge_range('C1:D1', 'Consumption', merge_format)
worksheet.merge_range('E1:F1', 'Shortage', merge_format)
worksheet.merge_range('G1:H1', 'Delivery from M to G', merge_format)
worksheet.merge_range('I1:J1', 'X from M to G', merge_format)
worksheet.merge_range('K1:L1', 'No of Trucks from M to G', merge_format)
worksheet.merge_range('M1:N1', 'Distance from M to G', merge_format)
worksheet.merge_range('O1:P1', 'Delivery from G to S', merge_format)
worksheet.merge_range('Q1:R1', 'X from G to S', merge_format)
worksheet.merge_range('S1:T1', 'No of Trucks from G to S', merge_format)
worksheet.merge_range('U1:V1', 'Distance from G to S', merge_format)
worksheet.merge_range('W1:X1', 'Delivery from S to R', merge_format)
worksheet.merge_range('Y1:Z1', 'X from S to R', merge_format)
worksheet.merge_range('AA1:AB1', 'No of Trucks from S to R', merge_format)
worksheet.merge_range('AC1:AD1', 'Distance from S to R', merge_format)
worksheet.merge_range('AE1:AF1', 'Delivery from R to D', merge_format)
worksheet.merge_range('AG1:AH1', 'X from R to D', merge_format)
worksheet.merge_range('AI1:AJ1', 'No of Trucks from R to D', merge_format)
worksheet.merge_range('AK1:AL1', 'Distance from R to D', merge_format)
worksheet.merge_range('AM1:AN1', 'Delivery from D to I', merge_format)
worksheet.merge_range('AO1:AP1', 'X from D to I', merge_format)
worksheet.merge_range('AQ1:AR1', 'No of Trucks from D to I', merge_format)
worksheet.merge_range('AS1:AT1', 'Distance from D to I', merge_format)

x = 0
y = 0
z = 0
row = 1
col = 0

#Inventory
for x in range((g+s+r+d+i)*t):
    worksheet.write(row, col, names[x])
    worksheet.write(row, col + 1, round(sol[x]))
    row += 1

#Qgmt, Xgmt, Ngmt
row = 1
for y in range(x+1,x+g*m*t+1):
    worksheet.write(row, 6, names[y])
    worksheet.write(row, 7, round(sol[y]))
    worksheet.write(row, 8, names[y+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t])
    worksheet.write(row, 9, round(sol[y+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t]))
    worksheet.write(row, 10, names[y+2*(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t])
    worksheet.write(row, 11, round(sol[y+2*(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t]))
    row +=1

#Qsgt,Xsgt,Nsgt
row = 1
for z in range(y+1,y+s*g*t+1):
    worksheet.write(row, 14, names[z])
    worksheet.write(row, 15, round(sol[z]))
    worksheet.write(row, 16, names[z+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t])
    worksheet.write(row, 17, round(sol[z+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t]))
    worksheet.write(row, 18, names[z+2*(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t])
    worksheet.write(row, 19, round(sol[z+2*(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t]))
    row +=1

#Qrst,Xrst,Nrst
row = 1
x = z
for y in range(x+1,x+r*s*t+1):
    worksheet.write(row, 22, names[y])
    worksheet.write(row, 23, round(sol[y]))
    worksheet.write(row, 24, names[y+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t])
    worksheet.write(row, 25, round(sol[y+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t]))
    worksheet.write(row, 26, names[y+2*(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t])
    worksheet.write(row, 27, round(sol[y+2*(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t]))
    row +=1

#Qdrt,Xdrt,Ndrt
row = 1
x = y
for y in range(x+1,x+r*d*t+1):
    worksheet.write(row, 30, names[y])
    worksheet.write(row, 31, round(sol[y]))
    worksheet.write(row, 32, names[y+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t])
    worksheet.write(row, 33, round(sol[y+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t]))
    worksheet.write(row, 34, names[y+2*(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t])
    worksheet.write(row, 35, round(sol[y+2*(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t]))
    row +=1

#Qidt,Xidt,Nidt
row = 1
x = y
for y in range(x+1,x+i*d*t+1):
    worksheet.write(row, 38, names[y])
    worksheet.write(row, 39, round(sol[y]))
    worksheet.write(row, 40, names[y+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t])
    worksheet.write(row, 41, round(sol[y+(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t]))
    worksheet.write(row, 42, names[y+2*(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t])
    worksheet.write(row, 43, round(sol[y+2*(g*m+d*r+s*g+r*s+i*d)*t+2*i*j*t]))
    row +=1

#Shortage
row = 1
x = y
for y in range(x+1,x+i*j*t+1):
    worksheet.write(row, 4, names[y])
    worksheet.write(row, 5, round(sol[y]))
    row +=1

#Consumption
row = 1
x = y
for y in range(x+1,x+i*j*t+1):
    worksheet.write(row, 2, names[y])
    worksheet.write(row, 3, round(sol[y]))
    row +=1

#Distances from M to G
row = 1
for y in range(g*m):
    worksheet.write(row, 12, D_names[y])
    worksheet.write(row, 13, distances[y])
    row +=1

#Distances from G to S
row = 1
x = y
for y in range(x+1,x+s*g+1):
    worksheet.write(row, 20, D_names[y])
    worksheet.write(row, 21, distances[y])
    row +=1

#Distances from S to R
row = 1
x = y
for y in range(x+1,x+s*r+1):
    worksheet.write(row, 28, D_names[y])
    worksheet.write(row, 29, distances[y])
    row +=1

#Distances from R to D
row = 1
x = y
for y in range(x+1,x+d*r+1):
    worksheet.write(row, 36, D_names[y])
    worksheet.write(row, 37, distances[y])
    row +=1

#Distances from D to I
row = 1
x = y
for y in range(x+1,x+d*i+1):
    worksheet.write(row, 44, D_names[y])
    worksheet.write(row, 45, distances[y])
    row +=1

workbook.close()