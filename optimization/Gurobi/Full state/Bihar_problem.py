import numpy as np
import pandas as pd
import xlsxwriter

import gurobipy as gp
from gurobipy import GRB

####################### INDEX ################################

j = 1  #Customer sub index
m = 1  #Manufacturer sub index
g = 1  #GMSD index
s = 1  #State sub index
r = 9  #Region sub index
d = 38  #District sub index
i = 606 #Clinic sub index
t = 12  #Time sub index

customers = list(range(1,j+1))
manufacturers = list(range(1,m+1))
gmsd = list(range(1,g+1))
svs = list(range(1,s+1))
rvs = list(range(1,r+1))
dvs = list(range(1,d+1))
clinics = list(range(1,i+1))
time = list(range(1,t+1))

########################### PARAMETERS ################################
l = GRB.INFINITY #large number for consistency constraints
fraction_storage = 1 #Fraction of total capacity in cold chain points to be considered for COVID-19 vaccine
fraction_transport = 1 #Fraction of total capacity in vehicles to be considered for COVID-19 vaccine

#Transportation cost
diesel_cost = 14
booking_cost = 5000
np.random.seed(133)

#Distances
Dgm = [[1000]] #From M to G (confirm)
Dsg = [[550]] #From G to S (confirm)

#From S to R
df_Drs = pd.read_csv("distances_sr.csv")
Drs = [[0 for R in range(r)] for S in range(s)]
for index in df_Drs.index:
	Drs[df_Drs['s'][index]-1][df_Drs['r'][index]-1] = df_Drs['Distance'][index]

#From R to D
df_Ddr = pd.read_csv("distances_rd.csv")
Ddr = [[0 for D in range(d)] for R in range(r)]
for index in df_Ddr.index:
	Ddr[df_Ddr['r'][index]-1][df_Ddr['d'][index]-1] = df_Ddr['Distance'][index]

#From D to I
df_Did = pd.read_csv("distances_di.csv")
Did = [[0 for I in range(i)] for D in range(d)]
for index in df_Did.index:
	Did[df_Did['d'][index]-1][df_Did['i'][index]-1] = df_Did['Distance'][index]


#Capacity of trucks
cap_veh_gm = fraction_transport*2932075 #Refrigerated van
cap_veh_sg = fraction_transport*2932075 #Refrigerated van
cap_veh_rs = fraction_transport*290880 #Insulated van
cap_veh_dr = fraction_transport*290880 #Insulated van
cap_veh_id = fraction_transport*290880 #Insulated van


#Final transportation costs
Kgmt = np.array([[[Dgm[M][G]*diesel_cost+booking_cost for G in range(0,g)] for M in range(0,m)] for T in range(0,t)])
Ksgt = np.array([[[Dsg[G][S]*diesel_cost+booking_cost for S in range(0,s)] for G in range(0,g)] for T in range(0,t)])
Krst = np.array([[[Drs[S][R]*diesel_cost+booking_cost for R in range(0,r)] for S in range(0,s)] for T in range(0,t)])
Kdrt = np.array([[[Ddr[R][D]*diesel_cost+booking_cost for D in range(0,d)] for R in range(0,r)] for T in range(0,t)])
Kidt = np.array([[[Did[D][I]*diesel_cost+booking_cost for I in range(0,i)] for D in range(0,d)] for T in range(0,t)])


#Shortage costs
Pjt = [[700000 for J in range(j)] for T in range(t)]
# for T in range(t):
#     Pjt[T][0] = 750000
#     Pjt[T][1] = 650000
#Pjt_obj = np.array([[[Pjt[T][J] for I in range(i)] for J in range(j)] for T in range(t)])

#Inventory holding costs
hgt = [[0.3 for G in range(g)] for T in range(t)]
hst = [[0.3 for S in range(s)] for T in range(t)]
hrt = np.random.normal(0.4,0.05,r*t).reshape(t,r)
hdt = np.random.normal(0.4,0.05,d*t).reshape(t,d)
hit = np.random.normal(0.4,0.05,i*t).reshape(t,i)

#Ordering costs
Cgmt = [[[25000 for G in range(g)] for M in range(m)] for T in range(t)]
Csgt = [[[25000 for S in range(s)] for G in range(g)] for T in range(t)]
Crst = [[[25000 for R in range(r)] for S in range(s)] for T in range(t)]
Cdrt = [[[25000 for D in range(d)] for R in range(r)] for T in range(t)]
Cidt = [[[25000 for I in range(i)] for D in range(d)] for T in range(t)]

#Demand
wastage_factor = 0.5 #This value will depend on the vaccine, we are talking about. Here, it is BCG.

df_demand = pd.read_csv("demand_weekly.csv")
dijt = [[[0 for I in range(1,i+1)] for J in range(j)] for T in range(1,t+1)]
for index in df_demand.index:
    dijt[df_demand['t'][index]-1][df_demand['j'][index]-1][df_demand['i'][index]-1] = df_demand['demand'][index]

#Capacity of cold chain points
Bgt = [[fraction_storage*24545455 for G in range(g)] for T in range(t)]
Bst = [[fraction_storage*6818182 for S in range(s)] for T in range(t)]

df_brt = pd.read_csv("capacity_RVS.csv")
Brt = [[0 for R in range(r)] for T in range(t)]
for index in df_brt.index:
	Brt[df_brt['t'][index]-1][df_brt['r'][index]-1] = fraction_storage*df_brt['Capacity'][index]


df_bdt = pd.read_csv("capacity_DVS.csv")
Bdt = [[0 for D in range(d)] for T in range(t)]
for index in df_bdt.index:
	Bdt[df_bdt['t'][index]-1][df_bdt['d'][index]-1] = fraction_storage*df_bdt['Capacity'][index]

df_bit = pd.read_csv("capacity_clinics.csv")
Bit = [[0 for I in range(i)] for T in range(t)]
for index in df_bit.index:
	Bit[df_bit['t'][index]-1][df_bit['i'][index]-1] = fraction_storage*df_bit['Capacity'][index]



model = gp.Model('Vaccine_Distribution')

#Production Capacity
Bmt = [[1000000 for M in range(m)] for T in range(t)]

################### DECISION VARIABLES ##########################

#Inventory
Igt = model.addVars(time,gmsd,vtype=GRB.INTEGER, name="Igt")
Ist = model.addVars(time,svs,vtype=GRB.INTEGER, name="Ist")
Irt = model.addVars(time,rvs,vtype=GRB.INTEGER, name="Irt")
Idt = model.addVars(time,dvs,vtype=GRB.INTEGER, name="Idt")
Iit = model.addVars(time,clinics,vtype=GRB.INTEGER, name="Iit")

#Quantity
Qgmt = model.addVars(time,manufacturers,gmsd,vtype=GRB.INTEGER, name="Qgmt")
Qsgt = model.addVars(time,gmsd,svs,vtype=GRB.INTEGER, name="Qsgt")
Qrst = model.addVars(time,svs,rvs,vtype=GRB.INTEGER, name="Qrst")
Qdrt = model.addVars(time,rvs,dvs,vtype=GRB.INTEGER, name="Qdrt")
Qidt = model.addVars(time,dvs,clinics,vtype=GRB.INTEGER, name="Qidt")

#Shortage and Consumption
Sijt = model.addVars(time,customers,clinics,vtype=GRB.INTEGER, name="Sijt")
Wijt = model.addVars(time,customers,clinics,vtype=GRB.INTEGER, name="Wijt")

#Assignment Variables
Xgmt = model.addVars(time,manufacturers,gmsd,vtype=GRB.BINARY, name="Xgmt")
Xsgt = model.addVars(time,gmsd,svs,vtype=GRB.BINARY, name="Xsgt")
Xrst = model.addVars(time,svs,rvs,vtype=GRB.BINARY, name="Xrst")
Xdrt = model.addVars(time,rvs,dvs,vtype=GRB.BINARY, name="Xdrt")
Xidt = model.addVars(time,dvs,clinics,vtype=GRB.BINARY, name="Xidt")

#Number of trucks
Ngmt = model.addVars(time,manufacturers,gmsd,vtype=GRB.INTEGER, name="Ngmt")
Nsgt = model.addVars(time,gmsd,svs,vtype=GRB.INTEGER, name="Nsgt")
Nrst = model.addVars(time,svs,rvs,vtype=GRB.INTEGER, name="Nrst")
Ndrt = model.addVars(time,rvs,dvs,vtype=GRB.INTEGER, name="Ndrt")
Nidt = model.addVars(time,dvs,clinics,vtype=GRB.INTEGER, name="Nidt")


############################# OBJECTIVE FUNCTION ###########################
transport_part = gp.quicksum(Kgmt[T-1][M-1][G-1]*Ngmt[T,M,G] for G in gmsd for M in manufacturers for T in time)
transport_part += gp.quicksum(Ksgt[T-1][G-1][S-1]*Nsgt[T,G,S] for S in svs for G in gmsd for T in time)
transport_part += gp.quicksum(Krst[T-1][S-1][R-1]*Nrst[T,S,R] for R in rvs for S in svs for T in time)
transport_part += gp.quicksum(Kdrt[T-1][R-1][D-1]*Ndrt[T,R,D] for D in dvs for R in rvs for T in time)
transport_part += gp.quicksum(Kidt[T-1][D-1][I-1]*Nidt[T,D,I] for I in clinics for D in dvs for T in time)

inventory_part = gp.quicksum(hgt[T-1][G-1]*Igt[T,G] for G in gmsd for T in time)
inventory_part += gp.quicksum(hst[T-1][S-1]*Ist[T,S] for S in svs for T in time)
inventory_part += gp.quicksum(hrt[T-1][R-1]*Irt[T,R] for R in rvs for T in time)
inventory_part += gp.quicksum(hdt[T-1][D-1]*Idt[T,D] for D in dvs for T in time)
inventory_part += gp.quicksum(hit[T-1][I-1]*Iit[T,I] for I in clinics for T in time)

shortage_part = gp.quicksum(Pjt[T-1][J-1]*Sijt[T,J,I] for J in customers for I in clinics for T in time)

consumption_part = gp.quicksum(0*Wijt[T,J,I] for J in customers for I in clinics for T in time) #Do we even need this?

ordering_part = gp.quicksum(Cgmt[T-1][M-1][G-1]*Xgmt[T,M,G] for G in gmsd for M in manufacturers for T in time)
ordering_part += gp.quicksum(Csgt[T-1][G-1][S-1]*Xsgt[T,G,S] for S in svs for G in gmsd for T in time)
ordering_part += gp.quicksum(Crst[T-1][S-1][R-1]*Xrst[T,S,R] for R in rvs for S in svs for T in time)
ordering_part += gp.quicksum(Cdrt[T-1][R-1][D-1]*Xdrt[T,R,D] for D in dvs for R in rvs for T in time)
ordering_part += gp.quicksum(Cidt[T-1][D-1][I-1]*Xidt[T,D,I] for I in clinics for D in dvs for T in time)

model.setObjective(transport_part+inventory_part+shortage_part+consumption_part+ordering_part,GRB.MINIMIZE)


###################################### CONSTRAINTS ################################

#Inventory Balance
gmsd_inventory = model.addConstrs(((Igt[T-1,G] if T>1 else 0) + gp.quicksum(Qgmt[T,M,G] for M in manufacturers) 
                                    - Igt[T,G] == gp.quicksum(Qsgt[T,G,S] for S in svs) for G in gmsd for T in time),
                                     name="gmsd_inventory")
svs_inventory = model.addConstrs(((Ist[T-1,S] if T>1 else 0) + gp.quicksum(Qsgt[T,G,S] for G in gmsd) 
                                    - Ist[T,S] == gp.quicksum(Qrst[T,S,R] for R in rvs) for S in svs for T in time),
                                     name="svs_inventory")
rvs_inventory = model.addConstrs(((Irt[T-1,R] if T>1 else 0) + gp.quicksum(Qrst[T,S,R] for S in svs) 
                                    - Irt[T,R] == gp.quicksum(Qdrt[T,R,D] for D in dvs) for R in rvs for T in time),
                                     name="rvs_inventory")
dvs_inventory = model.addConstrs(((Idt[T-1,D] if T>1 else 0) + gp.quicksum(Qdrt[T,R,D] for R in rvs) 
                                    - Idt[T,D] == gp.quicksum(Qidt[T,D,I] for I in clinics) for D in dvs for T in time),
                                     name="dvs_inventory")
clinic_inventory = model.addConstrs(((Iit[T-1,I] if T>1 else 0) + gp.quicksum(Qidt[T,D,I] for D in dvs) 
                                    - Iit[T,I] == gp.quicksum(Wijt[T,J,I] for J in customers) for I in clinics for T in time),
                                     name="clinic_inventory")

#Consumption by demand
consumption_demand = model.addConstrs((Wijt[T,J,I] <= dijt[T-1][J-1][I-1] for I in clinics for J in customers for T in time),
                                    name = "consumption_demand")

#Consumption Balance
consumption_balance = model.addConstrs((Wijt[T,J,I] + Sijt[T,J,I] == dijt[T-1][J-1][I-1] for I in clinics for J in customers for T in time),
                                    name = "consumption_balance")

#Inventory Capacity constraints
gmsd_cap = model.addConstrs((Igt[T,G]<=Bgt[T-1][G-1] for G in gmsd for T in time),name = "gmsd_cap")
svs_cap = model.addConstrs((Ist[T,S]<=Bst[T-1][S-1] for S in svs for T in time),name = "svs_cap")
rvs_cap = model.addConstrs((Irt[T,R]<=Brt[T-1][R-1] for R in rvs for T in time),name = "rvs_cap")
dvs_cap = model.addConstrs((Idt[T,D]<=Bdt[T-1][D-1] for D in dvs for T in time),name = "dvs_cap")
clinic_cap = model.addConstrs((Iit[T,I]<=Bit[T-1][I-1] for I in clinics for T in time),name = "clinic_cap")

#Production capacity constraints
production_cap = model.addConstrs((gp.quicksum(Qgmt[T,M,G] for G in gmsd)<= Bmt[T-1][M-1] for M in manufacturers for T in time)
                                ,name = "production_cap")

#Facility selection constraints
fac_sg = model.addConstrs((gp.quicksum(Xsgt[T,G,S] for G in gmsd)<=1 for S in svs for T in time),name = "fac_sg")
fac_rs = model.addConstrs((gp.quicksum(Xrst[T,S,R] for S in svs)<=1 for R in rvs for T in time),name = "fac_rs")
fac_dr = model.addConstrs((gp.quicksum(Xdrt[T,R,D] for R in rvs)<=1 for D in dvs for T in time),name = "fac_dr")
fac_id = model.addConstrs((gp.quicksum(Xidt[T,D,I] for D in dvs)<=1 for I in clinics for T in time),name = "fac_id")

#Constraints for consistency of q and X
cons_1 = model.addConstrs((Xgmt[T,M,G]<=l*Qgmt[T,M,G] for G in gmsd for M in manufacturers for T in time),name = "cons_1")
cons_2 = model.addConstrs((Qgmt[T,M,G]<=l*Xgmt[T,M,G] for G in gmsd for M in manufacturers for T in time),name = "cons_2")
cons_3 = model.addConstrs((Xsgt[T,G,S]<=l*Qsgt[T,G,S] for S in svs for G in gmsd for T in time),name = "cons_3")
cons_4 = model.addConstrs((Qsgt[T,G,S]<=l*Xsgt[T,G,S] for S in svs for G in gmsd for T in time),name = "cons_4")
cons_5 = model.addConstrs((Xrst[T,S,R]<=l*Qrst[T,S,R] for R in rvs for S in svs for T in time),name = "cons_5")
cons_6 = model.addConstrs((Qrst[T,S,R]<=l*Xrst[T,S,R] for R in rvs for S in svs for T in time),name = "cons_6")
cons_7 = model.addConstrs((Xdrt[T,R,D]<=l*Qdrt[T,R,D] for D in dvs for R in rvs for T in time),name = "cons_7")
cons_8 = model.addConstrs((Qdrt[T,R,D]<=l*Xdrt[T,R,D] for D in dvs for R in rvs for T in time),name = "cons_8")
cons_9 = model.addConstrs((Xidt[T,D,I]<=l*Qidt[T,D,I] for I in clinics for D in dvs for T in time),name = "cons_9")
cons_10 = model.addConstrs((Qidt[T,D,I]<=l*Xidt[T,D,I] for I in clinics for D in dvs for T in time),name = "cons_10")

#Number of trucks constraints
num_trucks_1 = model.addConstrs((Qgmt[T,M,G]/cap_veh_gm<=Ngmt[T,M,G] for G in gmsd for M in manufacturers for T in time),name = "num_trucks_1")
num_trucks_2 = model.addConstrs((Ngmt[T,M,G]-Qgmt[T,M,G]/cap_veh_gm<=((cap_veh_gm-1)/cap_veh_gm) for G in gmsd for M in manufacturers for T in time),name = "num_trucks_2")
num_trucks_3 = model.addConstrs((Qsgt[T,G,S]/cap_veh_sg<=Nsgt[T,G,S] for S in svs for G in gmsd for T in time),name = "num_trucks_3")
num_trucks_4 = model.addConstrs((Nsgt[T,G,S]-Qsgt[T,G,S]/cap_veh_sg<=((cap_veh_sg-1)/cap_veh_sg) for S in svs for G in gmsd for T in time),name = "num_trucks_4")
num_trucks_5 = model.addConstrs((Qrst[T,S,R]/cap_veh_rs<=Nrst[T,S,R] for R in rvs for S in svs for T in time),name = "num_trucks_5")
num_trucks_6 = model.addConstrs((Nrst[T,S,R]-Qrst[T,S,R]/cap_veh_rs<=((cap_veh_rs-1)/cap_veh_rs) for R in rvs for S in svs for T in time),name = "num_trucks_6")
num_trucks_7 = model.addConstrs((Qdrt[T,R,D]/cap_veh_dr<=Ndrt[T,R,D] for D in dvs for R in rvs for T in time),name = "num_trucks_7")
num_trucks_8 = model.addConstrs((Ndrt[T,R,D]-Qdrt[T,R,D]/cap_veh_dr<=((cap_veh_dr-1)/cap_veh_dr) for D in dvs for R in rvs for T in time),name = "num_trucks_8")
num_trucks_9 = model.addConstrs((Qidt[T,D,I]/cap_veh_id<=Nidt[T,D,I] for I in clinics for D in dvs for T in time),name = "num_trucks_9")
num_trucks_10 = model.addConstrs((Nidt[T,D,I]-Qidt[T,D,I]/cap_veh_id<=((cap_veh_id-1)/cap_veh_id) for I in clinics for D in dvs for T in time),name = "num_trucks_10")

#################################Solving the problem##########################
model.optimize()

names = []
sol = []
for v in model.getVars():
    print(v.varName,"=", v.x)
    names.append(v.varName)
    sol.append(v.x)


###########################Printing the results to excel file################## 

np.random.seed(133)
# Dgm = np.random.normal(1000,250,g*m).reshape(m,g)
# Dsg = np.random.normal(1000,250,s*g).reshape(g,s)
# Drs = np.random.normal(400,75,r*s).reshape(s,r)
# Ddr = np.random.normal(200,25,d*r).reshape(r,d)
# Did = np.random.normal(100,25,d*i).reshape(d,i)

Dgm_name = np.array([["D(g,m)("+str(G)+","+str(M)+")" for G in range(1,g+1)] for M in range(1,m+1)])
Dsg_name = np.array([["D(s,g)("+str(S)+","+str(G)+")" for S in range(1,s+1)] for G in range(1,g+1)]) 
Drs_name = np.array([["D(r,s)("+str(R)+","+str(S)+")" for R in range(1,r+1)] for S in range(1,s+1)])
Ddr_name = np.array([["D(d,r)("+str(D)+","+str(R)+")" for D in range(1,d+1)] for R in range(1,r+1)])
Did_name = np.array([["D(i,d)("+str(I)+","+str(D)+")" for I in range(1,i+1)] for D in range(1,d+1)])

distances = np.concatenate((Dgm,Dsg,Drs,Ddr,Did),axis=None).tolist()
D_names = np.concatenate((Dgm_name,Dsg_name,Drs_name,Ddr_name,Did_name),axis=None).tolist()

workbook = xlsxwriter.Workbook('C:/Users/Ayush/Desktop/test1.xlsx')
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

