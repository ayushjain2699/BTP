import numpy as np
import pandas as pd
import xlsxwriter

import gurobipy as gp
from gurobipy import GRB

####################### INDEX ################################

j = 3  #Customer sub index
m = 1  #Manufacturer sub index
g = 1  #GMSD index
s = 1  #State sub index
r = 9  #Region sub index
d = 10  #District sub index
i = 59 #Clinic sub index
t = 12  #Time sub index

clinic_breakpoints = [10,16,28,40,59,76,92,103,123,151,178,194,205,213,226,249,256,266,274,290,312,322,340,361,376,413,432,451,462,483,504,511,517,535,555,568,585,606]   #CLinic breakpoints for each districts
clinic_breakpoints = clinic_breakpoints[0:d]

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
booking_cost = {
	"MG" : 40000,
	"GS" : 20000,
	"SR" : 12000,
	"RD" : 10000,
	"DI" : 5000
}
np.random.seed(133)

#Distances
Dgm = [[1000]] #From M to G (confirm)
Dsg = [[550]] #From G to S (confirm)

#From S to R
df_Drs = pd.read_csv("Input_data/distances_sr.csv")
Drs = [[0 for R in range(r)] for S in range(s)]
for index in df_Drs.index:
	Drs[df_Drs['s'][index]-1][df_Drs['r'][index]-1] = df_Drs['Distance'][index]

#From R to D
df_Ddr = pd.read_csv("Input_data/distances_rd.csv")
Ddr = [[0 for D in range(d)] for R in range(r)]
for index in df_Ddr.index:
	if (df_Ddr['d'][index] > d):
		continue
	Ddr[df_Ddr['r'][index]-1][df_Ddr['d'][index]-1] = df_Ddr['Distance'][index]

#From D to I
df_Did = pd.read_csv("Input_data/distances_di.csv")
Did = [[0 for I in range(i)] for D in range(d)]
for index in df_Did.index:
	if (df_Did['d'][index] > d or df_Did['i'][index] > i):
		continue
	Did[df_Did['d'][index]-1][df_Did['i'][index]-1] = df_Did['Distance'][index]

#Capacity of trucks
cap_veh_gm = round(fraction_transport*2932075) #Refrigerated van
cap_veh_sg = round(fraction_transport*2932075) #Refrigerated van
cap_veh_rs = round(fraction_transport*290880) #Insulated van
cap_veh_dr = round(fraction_transport*290880) #Insulated van
cap_veh_id = round(fraction_transport*290880) #Insulated van


#Final transportation costs
Kgmt = np.array([[[Dgm[M][G]*diesel_cost+booking_cost["MG"] for G in range(0,g)] for M in range(0,m)] for T in range(0,t)])
Ksgt = np.array([[[Dsg[G][S]*diesel_cost+booking_cost["GS"] for S in range(0,s)] for G in range(0,g)] for T in range(0,t)])
Krst = np.array([[[Drs[S][R]*diesel_cost+booking_cost["SR"] for R in range(0,r)] for S in range(0,s)] for T in range(0,t)])
Kdrt = np.array([[[Ddr[R][D]*diesel_cost+booking_cost["RD"] for D in range(0,d)] for R in range(0,r)] for T in range(0,t)])
Kidt = np.array([[[Did[D][I]*diesel_cost+booking_cost["DI"] for I in range(0,i)] for D in range(0,d)] for T in range(0,t)])


#Shortage costs;
Pjt = [[0 for J in range(j)] for T in range(t)]
for T in range(t):
    Pjt[T][0] = 75000       #children
    Pjt[T][1] = 12150       #Old
    Pjt[T][2] = 50000       #hcw


#Clinical cost per unit of vaccine
Vj = 225

#Inventory holding costs
hgt = [[0.3 for G in range(g)] for T in range(t)]
hst = [[0.3 for S in range(s)] for T in range(t)]
hrt = np.random.normal(0.4,0.05,r*t).reshape(t,r)
hdt = np.random.normal(0.4,0.05,d*t).reshape(t,d)
hit = np.random.normal(0.4,0.05,i*t).reshape(t,i)

#Ordering costs
Cgmt = [[[200000 for G in range(g)] for M in range(m)] for T in range(t)]
Csgt = [[[100000 for S in range(s)] for G in range(g)] for T in range(t)]
Crst = [[[75000 for R in range(r)] for S in range(s)] for T in range(t)]
Cdrt = [[[25000 for D in range(d)] for R in range(r)] for T in range(t)]
Cidt = [[[15000 for I in range(i)] for D in range(d)] for T in range(t)]

#Demand
wastage_factor = 0.5 #This value will depend on the vaccine, we are talking about. Here, it is BCG.

#Fraction of demand
Fr_d = 0.25
#Fr_d = 0.5
#Fr_d = 0.75

df_demand = pd.read_csv("Input_data/weekly_demand.csv")
dijt = [[[0 for I in range(1,i+1)] for J in range(j)] for T in range(1,t+1)]
for index in df_demand.index:
	if (df_demand['i'][index] > i):
		break

	dijt[df_demand['t'][index]-1][df_demand['j'][index]-1][df_demand['i'][index]-1] = round(df_demand['demand'][index]*Fr_d)

#Capacity of cold chain points
Bgt = [[round(fraction_storage*24545455) for G in range(g)] for T in range(t)]
Bst = [[round(fraction_storage*6818182) for S in range(s)] for T in range(t)]

df_brt = pd.read_csv("Input_data/capacity_RVS.csv")
Brt = [[0 for R in range(r)] for T in range(t)]
for index in df_brt.index:
	Brt[df_brt['t'][index]-1][df_brt['r'][index]-1] = fraction_storage*df_brt['Capacity'][index]


df_bdt = pd.read_csv("Input_data/capacity_DVS.csv")
Bdt = [[0 for D in range(d)] for T in range(t)]
for index in df_bdt.index:
	if(df_bdt['d'][index] > d):
		break
	Bdt[df_bdt['t'][index]-1][df_bdt['d'][index]-1] = fraction_storage*df_bdt['Capacity'][index]


df_bit = pd.read_csv("Input_data/capacity_clinics.csv")
Bit = [[0 for I in range(i)] for T in range(t)]
for index in df_bit.index:
	if(df_bit['i'][index] > i):
		break
	Bit[df_bit['t'][index]-1][df_bit['i'][index]-1] = fraction_storage*df_bit['Capacity'][index]



model = gp.Model('Vaccine_Distribution')

#Production Capacity
Bmt = [[1000000 for M in range(m)] for T in range(t)]

#Average time required to administer the vaccine (minutes)
No = 5

#Number of medical personnel hours available (minutes)
Nit = [[3360 for I in range(i)] for T in range(t)]

#Hiring Cost of nurses
hc = 25000

#Firing Cost of nurses
fc = 10000

#Weekly wages of nurses
wg = 6175

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

#Nurses
N_nurses_it = model.addVars(time,clinics,vtype=GRB.INTEGER, name = "N_nurses_it")
H_nurses_it = model.addVars(time,clinics,vtype=GRB.INTEGER, name = "H_nurses_it")
F_nurses_it = model.addVars(time,clinics,vtype=GRB.INTEGER, name = "F_nurses_it")


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

consumption_part = gp.quicksum(Vj*Wijt[T,J,I] for J in customers for I in clinics for T in time) #Do we even need this?

ordering_part = gp.quicksum(Cgmt[T-1][M-1][G-1]*Xgmt[T,M,G] for G in gmsd for M in manufacturers for T in time)
ordering_part += gp.quicksum(Csgt[T-1][G-1][S-1]*Xsgt[T,G,S] for S in svs for G in gmsd for T in time)
ordering_part += gp.quicksum(Crst[T-1][S-1][R-1]*Xrst[T,S,R] for R in rvs for S in svs for T in time)
ordering_part += gp.quicksum(Cdrt[T-1][R-1][D-1]*Xdrt[T,R,D] for D in dvs for R in rvs for T in time)
ordering_part += gp.quicksum(Cidt[T-1][D-1][I-1]*Xidt[T,D,I] for I in clinics for D in dvs for T in time)

nurses_part = gp.quicksum(wg*N_nurses_it[T,I] for I in clinics for T in time)
nurses_part += gp.quicksum(hc*H_nurses_it[T,I] for I in clinics for T in time)
nurses_part += gp.quicksum(fc*F_nurses_it[T,I] for I in clinics for T in time)

model.setObjective(transport_part+inventory_part+shortage_part+consumption_part+ordering_part+nurses_part,GRB.MINIMIZE)


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

#Medical personnel availability constraints
med_constraint = model.addConstrs((gp.quicksum(No*Wijt[T,J,I] for J in customers)<=Nit[T-1][I-1]*N_nurses_it[T,I] for I in clinics for T in time),name = "med_constraint")

#Nurses Balance Constraint
nurses_constraint = model.addConstrs((N_nurses_it[T,I] == (N_nurses_it[T-1,I] if T>1 else 0)+ H_nurses_it[T,I] - F_nurses_it[T,I] for I in clinics for T in time),name = "nurses_constraint")

#################################Solving the problem##########################
model.optimize()

names = []
sol = []
for v in model.getVars():
    #print(v.varName,"=", v.x)
    names.append(v.varName)
    sol.append(v.x)
print("Done")

# ###########################Code for full excel sheet results generation##########################
workbook = xlsxwriter.Workbook('fraction-1.xlsx')
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


#Inventory
row = 1
for T in time:
    for G in gmsd:
        worksheet.write(row,0,Igt[T,G].varName)
        worksheet.write(row,1,round(Igt[T,G].x))
        row += 1

for T in time:
    for S in svs:
        worksheet.write(row,0,Ist[T,S].varName)
        worksheet.write(row,1,round(Ist[T,S].x))
        row += 1

for T in time:
    for R in rvs:
        worksheet.write(row,0,Irt[T,R].varName)
        worksheet.write(row,1,round(Irt[T,R].x))
        row += 1
        
for T in time:
    for D in dvs:
        worksheet.write(row,0,Idt[T,D].varName)
        worksheet.write(row,1,round(Idt[T,D].x))
        row += 1

for T in time:
    for I in clinics:
        worksheet.write(row,0,Iit[T,I].varName)
        worksheet.write(row,1,round(Iit[T,I].x))
        row += 1

#Qgmt, Xgmt, Ngmt
row = 1
for T in time:
    for M in manufacturers:
        for G in gmsd:
            worksheet.write(row,6,Qgmt[T,M,G].varName)
            worksheet.write(row,7,round(Qgmt[T,M,G].x))
            worksheet.write(row,8,Xgmt[T,M,G].varName)
            worksheet.write(row,9,round(Xgmt[T,M,G].x))
            worksheet.write(row,10,Ngmt[T,M,G].varName)
            worksheet.write(row,11,round(Ngmt[T,M,G].x))
            row += 1
#Dgm
row = 1
for M in manufacturers:
    for G in gmsd:
        variable = "Dgm["+str(M)+","+str(G)+"]"
        worksheet.write(row,12,variable)
        worksheet.write(row,13,Dgm[M-1][G-1])
        row += 1
        

#Qsgt,Xsgt,Nsgt
row = 1
for T in time:
    for G in gmsd:
        for S in svs:
            worksheet.write(row,14,Qsgt[T,G,S].varName)
            worksheet.write(row,15,round(Qsgt[T,G,S].x))
            worksheet.write(row,16,Xsgt[T,G,S].varName)
            worksheet.write(row,17,round(Xsgt[T,G,S].x))
            worksheet.write(row,18,Nsgt[T,G,S].varName)
            worksheet.write(row,19,round(Nsgt[T,G,S].x))
            row += 1
#Dsg
row = 1
for G in gmsd:
    for S in svs:
        variable = "Dsg["+str(G)+","+str(S)+"]"
        worksheet.write(row,20,variable)
        worksheet.write(row,21,Dsg[G-1][S-1])
        row += 1

#Qrst,Xrst,Nrst
row = 1
for T in time:
    for S in svs:
        for R in rvs:
            worksheet.write(row,22,Qrst[T,S,R].varName)
            worksheet.write(row,23,round(Qrst[T,S,R].x))
            worksheet.write(row,24,Xrst[T,S,R].varName)
            worksheet.write(row,25,round(Xrst[T,S,R].x))
            worksheet.write(row,26,Nrst[T,S,R].varName)
            worksheet.write(row,27,round(Nrst[T,S,R].x))
            row += 1
#Drs
row = 1
for S in svs:
    for R in rvs:
        variable = "Drs["+str(S)+","+str(R)+"]"
        worksheet.write(row,28,variable)
        worksheet.write(row,29,Drs[S-1][R-1])
        row += 1

#Qdrt,Xdrt,Ndrt
row = 1
for T in time:
    for R in rvs:
        for D in dvs:
            worksheet.write(row,30,Qdrt[T,R,D].varName)
            worksheet.write(row,31,round(Qdrt[T,R,D].x))
            worksheet.write(row,32,Xdrt[T,R,D].varName)
            worksheet.write(row,33,round(Xdrt[T,R,D].x))
            worksheet.write(row,34,Ndrt[T,R,D].varName)
            worksheet.write(row,35,round(Ndrt[T,R,D].x))
            row += 1
#Ddr
row = 1
for R in rvs:
    for D in dvs:
        variable = "Ddr["+str(R)+","+str(D)+"]"
        worksheet.write(row,36,variable)
        worksheet.write(row,37,Ddr[R-1][D-1])
        row += 1
        
#Qidt,Xidt,Nidt
row = 1
for T in time:
    for D in dvs:
        for I in clinics:
            worksheet.write(row,38,Qidt[T,D,I].varName)
            worksheet.write(row,39,round(Qidt[T,D,I].x))
            worksheet.write(row,40,Xidt[T,D,I].varName)
            worksheet.write(row,41,round(Xidt[T,D,I].x))
            worksheet.write(row,42,Nidt[T,D,I].varName)
            worksheet.write(row,43,round(Nidt[T,D,I].x))
            row += 1
#Did
row = 1
for D in dvs:
    for I in clinics:
        variable = "Did["+str(D)+","+str(I)+"]"
        worksheet.write(row,44,variable)
        worksheet.write(row,45,Did[D-1][I-1])
        row += 1

#Shortage
row = 1
for T in time:
    for J in customers:
        for I in clinics:
            worksheet.write(row,4,Sijt[T,J,I].varName)
            worksheet.write(row,5,round(Sijt[T,J,I].x))
            row += 1


#Consumption
row = 1
for T in time:
    for J in customers:
        for I in clinics:
            worksheet.write(row,2,Wijt[T,J,I].varName)
            worksheet.write(row,3,round(Wijt[T,J,I].x))
            row += 1



workbook.close()

###########################Code for generating CSV files for graphical analysis################## 

import pandas as pd
start = (g+s+r+d+i)*t 
T = 1
for T in range(1,13):
    x = start + (T-1)*m*g
    address = "Excel files/"
    address += str(T)
    address += "-M to G"
    address += ".xlsx"
    #print(address)

    #M to G
    workbook = xlsxwriter.Workbook(address)
    worksheet = workbook.add_worksheet("M to G")
    row = 0
    count = 0
    for M in range(m):
        for G in range(g):
            if (round(sol[x+count]) != 0):
                worksheet.write(row,0,M+1)
                worksheet.write(row,1,G+1)
                worksheet.write(row,2,sol[x+count])
                count += 1
                row += 1
            else:
                count += 1
    #print("M to G done")
    workbook.close()
    read_file = pd.read_excel (address) 
    csv_address ="CSV files/"
    csv_address += str(T)
    csv_address += "-M to G"
    csv_address += ".csv"
    read_file.to_csv (csv_address, index = None, header=True) 
    
    #G to S
    x = x + (t-T+1)*m*g +(T-1)*s*g
    address = "Excel files/"
    address += str(T)
    address += "-G to S"
    address += ".xlsx"
    #print(address)
    workbook = xlsxwriter.Workbook(address)
    worksheet = workbook.add_worksheet("G to S")
    row = 0
    count = 0
    for G in range(g):
        for S in range(s):
            if (round(sol[x+count]) != 0):
                worksheet.write(row,0,G+1)
                worksheet.write(row,1,S+1)
                worksheet.write(row,2,sol[x+count])
                row += 1
                count += 1
            else:
                count += 1
    #print("G to S done")
    workbook.close()
    read_file = pd.read_excel (address) 
    csv_address ="CSV files/"
    csv_address += str(T)
    csv_address += "-G to S"
    csv_address += ".csv"
    read_file.to_csv (csv_address, index = None, header=True) 

    #S to R
    x = x + (t-T+1)*g*s + (T-1)*r*s
    address = "Excel files/"
    address += str(T)
    address += "-S to R"
    address += ".xlsx"
    #print(address)
    workbook = xlsxwriter.Workbook(address)
    worksheet = workbook.add_worksheet("S to R")
    row = 0
    count = 0
    for S in range(s):
        for R in range(r):
            if (round(sol[x+count]) != 0):
                worksheet.write(row,0,S+1)
                worksheet.write(row,1,R+1)
                worksheet.write(row,2,sol[x+count])
                row += 1
                count += 1
            else:
                count += 1
    #print("S to R done")
    workbook.close()
    read_file = pd.read_excel (address) 
    csv_address ="CSV files/"
    csv_address += str(T)
    csv_address += "-S to R"
    csv_address += ".csv"
    read_file.to_csv (csv_address, index = None, header=True) 

    #R to D
    x = x + (t-T+1)*s*r + (T-1)*r*d
    address = "Excel files/"
    address += str(T)
    address += "-R to D"
    address += ".xlsx"
    #print(address)
    workbook = xlsxwriter.Workbook(address)
    worksheet = workbook.add_worksheet("R to D")
    row = 0
    count = 0
    for R in range(r):
        for D in range(d):
            if (round(sol[x+count]) != 0):
                worksheet.write(row,0,R+1)
                worksheet.write(row,1,D+1)
                worksheet.write(row,2,sol[x+count])
                row += 1
                count += 1
            else:
                count += 1
    #print("R to D done")
    workbook.close()
    read_file = pd.read_excel (address) 
    csv_address ="CSV files/"
    csv_address += str(T)
    csv_address += "-R to D"
    csv_address += ".csv"
    read_file.to_csv (csv_address, index = None, header=True) 

    #D to I
    x = x + (t-T+1)*r*d + (T-1)*d*i
    address = "Excel files/"
    address += str(T)
    address += "-D to I"
    address += ".xlsx"
    #print(address)
    workbook = xlsxwriter.Workbook(address)
    worksheet = workbook.add_worksheet("D to I")
    row = 0
    count = 0
    for D in range(d):
        for I in range(i):
            if (round(sol[x+count]) != 0):
                worksheet.write(row,0,D+1)
                worksheet.write(row,1,I+1)
                worksheet.write(row,2,sol[x+count])
                row += 1
                count += 1
            else:
                count += 1
    #print("D to I done")
    workbook.close()
    read_file = pd.read_excel (address) 
    csv_address ="CSV files/"
    csv_address += str(T)
    csv_address += "-D to I"
    csv_address += ".csv"
    read_file.to_csv (csv_address, index = None, header=True)


############################################ Summary #####################################################

############## transport part ###############
no_of_times_M = ""
cost_M = ""
total_cost_M = 0
for M in manufacturers:
    number = 0
    cost = 0
    for G in gmsd:
        for T in time:
            number =  number + Xgmt[T,M,G].x
            cost = cost + Kgmt[T-1][M-1][G-1]*Ngmt[T,M,G].x
    cost = round(cost)
    if(number!=0):
        if(no_of_times_M==""):
            no_of_times_M += (str(M)+"("+str(number)+" times"+")")
            cost_M += (str(M)+"("+str(cost)+" Rs"+")")
        else:
            no_of_times_M += (", "+str(M)+"("+str(number)+" times"+")")
            cost_M += (", "+str(M)+"("+str(cost)+" Rs"+")")
        total_cost_M += cost


no_of_times_G = ""
cost_G = ""
total_cost_G = 0
for G in gmsd:
    number = 0
    cost = 0
    for S in svs:
        for T in time:
            number =  number + Xsgt[T,G,S].x
            cost = cost + Ksgt[T-1][G-1][S-1]*Nsgt[T,G,S].x
    cost = round(cost)
    if(number!=0):
        if(no_of_times_G==""):
            no_of_times_G += (str(G)+"("+str(number)+" times"+")")
            cost_G += (str(G)+"("+str(cost)+" Rs"+")")
        else:
            no_of_times_G += (", "+str(G)+"("+str(number)+" times"+")")
            cost_G += (", "+str(G)+"("+str(cost)+" Rs"+")")
        total_cost_G += cost


no_of_times_S = ""
cost_S = ""
total_cost_S = 0
for S in svs:
    number = 0
    cost = 0
    for R in rvs:
        for T in time:
            number =  number + Xrst[T,S,R].x
            cost = cost + Krst[T-1][S-1][R-1]*Nrst[T,S,R].x
    cost = round(cost)
    if(number!=0):
        if(no_of_times_S==""):
            no_of_times_S += (str(S)+"("+str(number)+" times"+")")
            cost_S += (str(S)+"("+str(cost)+" Rs"+")")
        else:
            no_of_times_S += (", "+str(S)+"("+str(number)+" times"+")")
            cost_S += (", "+str(S)+"("+str(cost)+" Rs"+")")
        total_cost_S += cost


no_of_times_R = ""
cost_R = ""
total_cost_R = 0
for R in rvs:
    number = 0
    cost = 0
    for D in dvs:
        for T in time:
            number =  number + Xdrt[T,R,D].x
            cost = cost + Kdrt[T-1][R-1][D-1]*Ndrt[T,R,D].x
    cost = round(cost)
    if(number!=0):
        if(no_of_times_R==""):
            no_of_times_R += (str(R)+"("+str(number)+" times"+")")
            cost_R += (str(R)+"("+str(cost)+" Rs"+")")
        else:
            no_of_times_R += (", "+str(R)+"("+str(number)+" times"+")")
            cost_R += (", "+str(R)+"("+str(cost)+" Rs"+")")
        total_cost_R += cost

no_of_times_D = ""
cost_D = ""
total_cost_D = 0
for D in dvs:
    number = 0
    cost = 0
    for I in clinics:
        for T in time:
            number =  number + Xidt[T,D,I].x
            cost = cost + Kidt[T-1][D-1][I-1]*Nidt[T,D,I].x
    cost = round(cost)
    if(number!=0):
        if(no_of_times_D==""):
            no_of_times_D += (str(D)+"("+str(number)+" times"+")")
            cost_D += (str(D)+"("+str(cost)+" Rs"+")")
        else:
            no_of_times_D += (", "+str(D)+"("+str(number)+" times"+")")
            cost_D += (", "+str(D)+"("+str(cost)+" Rs"+")")
        total_cost_D += cost


transport_summary = {
    "From":["M->G","G->S","S->R","R->D","D->I"],
    "Number of times transport occurs over the entire planning horizon":[no_of_times_M,no_of_times_G,no_of_times_S,no_of_times_R,no_of_times_D],
    "Cost Incurred":[cost_M,cost_G,cost_S,cost_R,cost_D],
    "Total Cost":[total_cost_M,total_cost_G,total_cost_S,total_cost_R,total_cost_D]
}


################ ordering part ##############

average_quantity_M = ""
cost_M = ""
total_cost_M = 0
for M in manufacturers:
    number = 0
    cost = 0
    quantity = 0
    for G in gmsd:
        for T in time:
            number =  number + Xgmt[T,M,G].x
            cost = cost + Cgmt[T-1][M-1][G-1]*Xgmt[T,M,G].x
            quantity = quantity + Qgmt[T,M,G].x
    cost = round(cost)
    if(number!=0):
        average_quantity = quantity/number
        if(average_quantity_M==""):
            average_quantity_M += (str(M)+"("+str(round(average_quantity))+" units"+")")
            cost_M += (str(M)+"("+str(cost)+" Rs"+")")
        else:
            average_quantity_M += (", "+str(M)+"("+str(round(average_quantity))+" units"+")")
            cost_M += (", "+str(M)+"("+str(cost)+" Rs"+")")
        total_cost_M += cost


average_quantity_G = ""
cost_G = ""
total_cost_G = 0
for G in gmsd:
    number = 0
    cost = 0
    quantity = 0
    for S in svs:
        for T in time:
            number =  number + Xsgt[T,G,S].x
            cost = cost + Csgt[T-1][G-1][S-1]*Xsgt[T,G,S].x
            quantity = quantity + Qsgt[T,G,S].x
    cost = round(cost)
    if(number!=0):
        average_quantity = quantity/number
        if(average_quantity_G==""):
            average_quantity_G += (str(G)+"("+str(round(average_quantity))+" units"+")")
            cost_G += (str(G)+"("+str(cost)+" Rs"+")")
        else:
            average_quantity_G += (", "+str(G)+"("+str(round(average_quantity))+" units"+")")
            cost_G += (", "+str(G)+"("+str(cost)+" Rs"+")")
        total_cost_G += cost


average_quantity_S = ""
cost_S = ""
total_cost_S = 0
for S in svs:
    number = 0
    cost = 0
    quantity = 0
    for R in rvs:
        for T in time:
            number =  number + Xrst[T,S,R].x
            cost = cost + Crst[T-1][S-1][R-1]*Xrst[T,S,R].x
            quantity = quantity + Qrst[T,S,R].x
    cost = round(cost)
    if(number!=0):
        average_quantity = quantity/number
        if(average_quantity_S==""):
            average_quantity_S += (str(S)+"("+str(round(average_quantity))+" units"+")")
            cost_S += (str(S)+"("+str(cost)+" Rs"+")")
        else:
            average_quantity_S += (", "+str(S)+"("+str(round(average_quantity))+" units"+")")
            cost_S += (", "+str(S)+"("+str(cost)+" Rs"+")")
        total_cost_S += cost


average_quantity_R = ""
cost_R = ""
total_cost_R = 0
for R in rvs:
    number = 0
    cost = 0
    quantity = 0
    for D in dvs:
        for T in time:
            number =  number + Xdrt[T,R,D].x
            cost = cost + Cdrt[T-1][R-1][D-1]*Xdrt[T,R,D].x
            quantity = quantity + Qdrt[T,R,D].x
    cost = round(cost)
    if(number!=0):
        average_quantity = quantity/number
        if(average_quantity_R==""):
            average_quantity_R += (str(R)+"("+str(round(average_quantity))+" units"+")")
            cost_R += (str(R)+"("+str(cost)+" Rs"+")")
        else:
            average_quantity_R += (", "+str(R)+"("+str(round((average_quantity)))+" units"+")")
            cost_R += (", "+str(R)+"("+str(cost)+" Rs"+")")
        total_cost_R += cost

average_quantity_D = ""
cost_D = ""
total_cost_D = 0
for D in dvs:
    number = 0
    cost = 0
    quantity = 0
    for I in clinics:
        for T in time:
            number =  number + Xidt[T,D,I].x
            cost = cost + Cidt[T-1][D-1][I-1]*Xidt[T,D,I].x
            quantity = quantity + Qidt[T,D,I].x
    cost = round(cost)
    if(number!=0):
        average_quantity = quantity/number
        if(average_quantity_D==""):
            average_quantity_D += (str(D)+"("+str(round(average_quantity))+" units"+")")
            cost_D += (str(D)+"("+str(cost)+" Rs"+")")
        else:
            average_quantity_D += (", "+str(D)+"("+str(round(average_quantity))+" units"+")")
            cost_D += (", "+str(D)+"("+str(cost)+" Rs"+")")
        total_cost_D += cost


ordering_summary = {
    "From":["M->G","G->S","S->R","R->D","D->I"],
    "Average quantities ordered over the entire planning horizon":[average_quantity_M,average_quantity_G,average_quantity_S,average_quantity_R,average_quantity_D],
    "Cost Incurred":[cost_M,cost_G,cost_S,cost_R,cost_D],
    "Total Cost":[total_cost_M,total_cost_G,total_cost_S,total_cost_R,total_cost_D]
}

################ inventory part #############

no_of_times_G = ""
avg_inv_G = ""
cost_G = ""
total_cost_G = 0
for G in gmsd:
    number = 0
    cost = 0
    total_inventory_G = 0
    for T in time:
        if(Igt[T,G].x>0):
            number += 1
            total_inventory_G += Igt[T,G].x
            cost += hgt[T-1][G-1]*Igt[T,G].x
    cost = round(cost)
    if(number!=0):
        if(no_of_times_G==""):
            no_of_times_G += (str(G)+"("+str(number)+" times"+")")
            cost_G += (str(G)+"("+str(cost)+" Rs"+")")
            avg_inv_G += (str(G)+"("+str(total_inventory_G/number)+" units"+")")

        else:
            no_of_times_G += (", "+str(G)+"("+str(number)+" times"+")")
            cost_G += (", "+str(G)+"("+str(cost)+" Rs"+")")
            if(number!=0):
                avg_inv_G += (", "+str(G)+"("+str(total_inventory_G/number)+" units"+")")
            else:
                avg_inv_G += (", "+str(G)+"("+str(0)+" units"+")")
    total_cost_G += cost
        

no_of_times_S = ""
avg_inv_S = ""
cost_S = ""
total_cost_S = 0
for S in svs:
    number = 0
    cost = 0
    total_inventory_S = 0
    for T in time:
        if(Ist[T,S].x>0):
            number += 1
            total_inventory_S += Ist[T,S].x
            cost += hst[T-1][S-1]*Ist[T,S].x
    cost = round(cost)
    if(number!=0):
        if(no_of_times_S==""):
            no_of_times_S += (str(S)+"("+str(number)+" times"+")")
            cost_S += (str(S)+"("+str(cost)+" Rs"+")")
            avg_inv_S += (str(S)+"("+str(total_inventory_S/number)+" units"+")")
    
        else:
            no_of_times_S += (", "+str(S)+"("+str(number)+" times"+")")
            cost_S += (", "+str(S)+"("+str(cost)+" Rs"+")")
            if(number!=0):
                avg_inv_S += (", "+str(S)+"("+str(total_inventory_S/number)+" units"+")") 
            else: 
                avg_inv_S += (", "+str(S)+"("+str(0)+" units"+")") 
    total_cost_S += cost


no_of_times_R = ""
avg_inv_R = ""
cost_R = ""
total_cost_R = 0
for R in rvs:
    number = 0
    cost = 0
    total_inventory_R = 0
    for T in time:
        if(Irt[T,R].x>0):
            number += 1
            total_inventory_R += Irt[T,R].x
            cost += hrt[T-1][R-1]*Irt[T,R].x
    cost = round(cost)
    if(number!=0):
        if(no_of_times_R==""):
            no_of_times_R += (str(R)+"("+str(number)+" times"+")")
            cost_R += (str(R)+"("+str(cost)+" Rs"+")")
            if(number!=0):
                avg_inv_R += (str(R)+"("+str(total_inventory_R/number)+" units"+")")
            else:
                avg_inv_R += (str(R)+"("+str(0)+" units"+")")
        else:
            no_of_times_R += (", "+str(R)+"("+str(number)+" times"+")")
            cost_R += (", "+str(R)+"("+str(cost)+" Rs"+")")
            if(number!=0):
                avg_inv_R += (", "+str(R)+"("+str(total_inventory_R/number)+" units"+")")
            else:  
                avg_inv_R += (", "+str(R)+"("+str(0)+" units"+")")
    total_cost_R += cost


no_of_times_D = ""
avg_inv_D = ""
cost_D = ""
total_cost_D = 0
for D in dvs:
    number = 0
    cost = 0
    total_inventory_D = 0
    for T in time:
        if(Idt[T,D].x>0):
            number += 1
            total_inventory_D += Idt[T,D].x
            cost += hdt[T-1][D-1]*Idt[T,D].x
    cost = round(cost)
    if(number!=0):
        if(no_of_times_D==""):
                no_of_times_D += (str(D)+"("+str(number)+" times"+")")
                cost_D += (str(D)+"("+str(cost)+" Rs"+")")
                if(number!=0):
                    avg_inv_D += (str(D)+"("+str(total_inventory_D/number)+" units"+")")
                else:
                    avg_inv_D += (str(D)+"("+str(0)+" units"+")")
        else:
            no_of_times_D += (", "+str(D)+"("+str(number)+" times"+")")
            cost_D += (", "+str(D)+"("+str(cost)+" Rs"+")")
            if(number!=0):
                avg_inv_D += (", "+str(D)+"("+str(total_inventory_D/number)+" units"+")")  
            else:
                avg_inv_D += (", "+str(D)+"("+str(0)+" units"+")")
    total_cost_D += cost


I_s = 1
total_cost_I = 0
no_of_times_I = ""
D = 1
cost_I = ""
avg_inv_I = ""
for I_b in clinic_breakpoints:
    number_average_district = 0
    avg_inv_avg_district = 0
    cost_avg_district = 0
    for I in range(I_s,I_b+1):
        number = 0
        cost = 0
        total_inventory_I = 0
        for T in time:
            if(Iit[T,I].x>0):
                number += 1
                total_inventory_I += Iit[T,I].x
                cost += hit[T-1][I-1]*Iit[T,I].x
        number_average_district += number
        if(number!=0):
            avg_inv_avg_district += total_inventory_I/number
        else:
            avg_inv_avg_district += 0
        cost_avg_district += cost
        total_cost_I += cost

    number_average_district /= (I_b-I_s+1)
    number_average_district = round(number_average_district)

    avg_inv_avg_district /= (I_b-I_s+1)
    avg_inv_avg_district = round(avg_inv_avg_district)

    cost_avg_district /= (I_b-I_s+1)

    if(no_of_times_I==""):
            no_of_times_I += (str(D)+"("+str(number_average_district)+" times"+")")
            cost_I += (str(D)+"("+str(round(cost_avg_district))+" Rs"+")")
            avg_inv_I += (str(D)+"("+str(round(avg_inv_avg_district))+" units"+")")
    else:
        no_of_times_I += (", "+str(D)+"("+str(number_average_district)+" times"+")")
        cost_I += (", "+str(D)+"("+str(round(cost_avg_district))+" Rs"+")")
        avg_inv_I += (", "+str(D)+"("+str(round(avg_inv_avg_district))+" units"+")") 

    D = D+1
    I_s = I_b+1

inventory_summary = {
    "CCP": ["G","S","R","D","I"],
    "Number of times Inventory is non zero":[no_of_times_G,no_of_times_S,no_of_times_R,no_of_times_D,no_of_times_I],
    "Average Inventory carried":[avg_inv_G,avg_inv_S,avg_inv_R,avg_inv_D,avg_inv_I],
    "Cost Incurred":[cost_G,cost_S,cost_R,cost_D,cost_I],
    "Total Cost":[total_cost_G,total_cost_S,total_cost_R,total_cost_D,total_cost_I]
}


########################### Shortages summary ##############################


I_s = 1
num1 = 0
costs_list = []
average_costs_list = []
number_of_clinics = []
count = 0
clinic_num = []
for num in clinic_breakpoints:
	cost = 0
	avg_cost = 0
	num_clinics = num - num1
	count += 1
	for x in range(num_clinics):
		I = I_s + x
		for J in customers:
			for T in time:
				if(Sijt[T,J,I].x!=0):
					cost += Pjt[T-1][J-1]*Sijt[T,J,I].x
	avg_cost = cost/num_clinics
	I_s += num_clinics
	num1 = num
	costs_list.append(cost)
	average_costs_list.append(avg_cost)
	number_of_clinics.append(num_clinics)
	clinic_num.append(count)

shortage_summary = {
    "District Number": clinic_num,
    "Number of clinics": number_of_clinics,
    "Total shortage cost Incurred": costs_list,
    "Average shortage cost incurred per clinic":average_costs_list
}


transport_df = pd.DataFrame.from_dict(transport_summary)
inventory_df = pd.DataFrame.from_dict(inventory_summary)
ordering_df = pd.DataFrame.from_dict(ordering_summary)
shortage_df = pd.DataFrame.from_dict(shortage_summary)

# transport_df.to_excel("transport.xlsx")
# inventory_df.to_excel("inventory.xlsx")
# ordering_df.to_excel("ordering.xlsx")
# shortage_df.to_excel("shortage.xlsx")

###########################################Compiled Results##################################################
writer = pd.ExcelWriter('compiled.xlsx',engine='xlsxwriter')   
workbook=writer.book
worksheet=workbook.add_worksheet('Compiled')
writer.sheets['Compiled'] = worksheet
worksheet.write(0,0,"Transport part")
transport_df.to_excel(writer,sheet_name='Compiled',startrow=2 , startcol=0)

x = 4 + len(transport_df.index)
worksheet.write(x,0,"Inventory part")
x += 2
inventory_df.to_excel(writer,sheet_name='Compiled',startrow=x , startcol=0)

x += 4 + len(inventory_df.index)
worksheet.write(x,0,"Ordering part")
x += 2
ordering_df.to_excel(writer,sheet_name='Compiled',startrow=x , startcol=0)

x += 4 + len(ordering_df.index)
worksheet.write(x,0,"Shortage part")
x += 2
shortage_df.to_excel(writer,sheet_name='Compiled',startrow=x , startcol=0)
workbook.close()

################################################# The End ######################################################### 