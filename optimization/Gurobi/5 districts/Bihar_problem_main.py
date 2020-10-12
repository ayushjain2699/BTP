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
d = 5  #District sub index
i = 59 #Clinic sub index
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
fraction_storage = 0.1 #Fraction of total capacity in cold chain points to be considered for COVID-19 vaccine
fraction_transport = 0.1 #Fraction of total capacity in vehicles to be considered for COVID-19 vaccine

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
cap_veh_gm = fraction_transport*2932075 #Refrigerated van
cap_veh_sg = fraction_transport*2932075 #Refrigerated van
cap_veh_rs = fraction_transport*290880 #Insulated van
cap_veh_dr = fraction_transport*290880 #Insulated van
cap_veh_id = fraction_transport*290880 #Insulated van


#Final transportation costs
Kgmt = np.array([[[Dgm[M][G]*diesel_cost+booking_cost["MG"] for G in range(0,g)] for M in range(0,m)] for T in range(0,t)])
Ksgt = np.array([[[Dsg[G][S]*diesel_cost+booking_cost["GS"] for S in range(0,s)] for G in range(0,g)] for T in range(0,t)])
Krst = np.array([[[Drs[S][R]*diesel_cost+booking_cost["SR"] for R in range(0,r)] for S in range(0,s)] for T in range(0,t)])
Kdrt = np.array([[[Ddr[R][D]*diesel_cost+booking_cost["RD"] for D in range(0,d)] for R in range(0,r)] for T in range(0,t)])
Kidt = np.array([[[Did[D][I]*diesel_cost+booking_cost["DI"] for I in range(0,i)] for D in range(0,d)] for T in range(0,t)])


#Shortage costs
Pjt = [[1000000 for J in range(j)] for T in range(t)]
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
Cgmt = [[[200000 for G in range(g)] for M in range(m)] for T in range(t)]
Csgt = [[[100000 for S in range(s)] for G in range(g)] for T in range(t)]
Crst = [[[75000 for R in range(r)] for S in range(s)] for T in range(t)]
Cdrt = [[[25000 for D in range(d)] for R in range(r)] for T in range(t)]
Cidt = [[[15000 for I in range(i)] for D in range(d)] for T in range(t)]

#Demand
wastage_factor = 0.5 #This value will depend on the vaccine, we are talking about. Here, it is BCG.

df_demand = pd.read_csv("Input_data/demand_weekly.csv")
dijt = [[[0 for I in range(1,i+1)] for J in range(j)] for T in range(1,t+1)]
for index in df_demand.index:
	if (df_demand['i'][index] > i):
		break

	dijt[df_demand['t'][index]-1][df_demand['j'][index]-1][df_demand['i'][index]-1] = df_demand['demand'][index]

#Capacity of cold chain points
Bgt = [[fraction_storage*24545455 for G in range(g)] for T in range(t)]
Bst = [[fraction_storage*6818182 for S in range(s)] for T in range(t)]

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
    #print(v.varName,"=", v.x)
    names.append(v.varName)
    sol.append(v.x)
print("Done")

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
            cost = cost + Cgmt[T-1][M-1][G-1]*Qgmt[T,M,G].x
            quantity = quantity + Qgmt[T,M,G].x
    if(number!=0):
    	average_quantity = quantity/number
        if(average_quantity_M==""):
            average_quantity_M += (str(M)+"("+str(average_quantity)+" units"+")")
            cost_M += (str(M)+"("+str(cost)+" Rs"+")")
        else:
            average_quantity_M += (", "+str(M)+"("+str(average_quantity)+" units"+")")
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
            cost = cost + Csgt[T-1][G-1][S-1]*Qsgt[T,G,S].x
            quantity = quantity + Qsgt[T,G,S].x
    if(number!=0):
    	average_quantity = quantity/number
        if(average_quantity_G==""):
            average_quantity_G += (str(G)+"("+str(average_quantity)+" units"+")")
            cost_G += (str(G)+"("+str(cost)+" Rs"+")")
        else:
            average_quantity_G += (", "+str(G)+"("+str(average_quantity)+" units"+")")
            cost_G += (", "+str(G)+"("+str(cost)+" Rs"+")")
        total_cost_G += cost


average_quantity_S = ""
cost_S = ""
total_cost_S = 0
for S in svs:
    number = 0
    cost = 0
    quantity = 0
    for R in dvs:
        for T in time:
            number =  number + Xrst[T,S,R].x
            cost = cost + Crst[T-1][S-1][R-1]*Qrst[T,S,R].x
            quantity = quantity + Qrst[T,S,R].x
    if(number!=0):
    	average_quantity = quantity/number
        if(average_quantity_S==""):
            average_quantity_S += (str(S)+"("+str(average_quantity)+" units"+")")
            cost_S += (str(S)+"("+str(cost)+" Rs"+")")
        else:
            average_quantity_S += (", "+str(S)+"("+str(average_quantity)+" units"+")")
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
            cost = cost + Cdrt[T-1][R-1][D-1]*Qdrt[T,R,D].x
            quantity = quantity + Qdrt[T,R,D].x
    if(number!=0):
    	average_quantity = quantity/number
        if(average_quantity_R==""):
            average_quantity_R += (str(R)+"("+str(average_quantity)+" units"+")")
            cost_R += (str(R)+"("+str(cost)+" Rs"+")")
        else:
            average_quantity_R += (", "+str(R)+"("+str(average_quantity)+" units"+")")
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
            cost = cost + Cidt[T-1][D-1][I-1]*Qidt[T,D,I].x
            quantity = quantity + Qidt[T,D.I].x
    if(number!=0):
    	average_quantity = quantity/number
        if(average_quantity_D==""):
            average_quantity_D += (str(D)+"("+str(average_quantity)+" units"+")")
            cost_D += (str(D)+"("+str(cost)+" Rs"+")")
        else:
            average_quantity_D += (", "+str(D)+"("+str(average_quantity)+" units"+")")
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
    if(no_of_times_G==""):
        no_of_times_G += (str(G)+"("+str(number)+" times"+")")
        cost_G += (str(G)+"("+str(cost)+" Rs"+")")
        if(number!=0):
            avg_inv_G += (str(G)+"("+str(total_inventory_G/number)+" units"+")")
        else:
            avg_inv_G += (str(G)+"("+str(0)+" units"+")")
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
            total_inventory_G += Ist[T,S].x
            cost += hst[T-1][S-1]*Ist[T,S].x
    cost = round(cost)
    if(no_of_times_S==""):
        no_of_times_S += (str(S)+"("+str(number)+" times"+")")
        cost_S += (str(S)+"("+str(cost)+" Rs"+")")
        if(number!=0):
            avg_inv_S += (str(S)+"("+str(total_inventory_S/number)+" units"+")")
        else:
            avg_inv_S += (str(S)+"("+str(0)+" units"+")")
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


clinic_breakpoints = [10,16,28,40,59]
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
            cost_I += (str(D)+"("+str(cost_avg_district)+" Rs"+")")
            avg_inv_I += (str(D)+"("+str(avg_inv_avg_district)+" units"+")")
    else:
        no_of_times_I += (", "+str(D)+"("+str(number_average_district)+" times"+")")
        cost_I += (", "+str(D)+"("+str(cost_avg_district)+" Rs"+")")
        avg_inv_I += (", "+str(D)+"("+str(avg_inv_avg_district)+" units"+")") 

    D = D+1

inventory_summary = {
    "CCP": ["G","S","R","D","I"],
    "Number of times Inventory is non zero":[no_of_times_G,no_of_times_S,no_of_times_R,no_of_times_D,no_of_times_I],
    "Average Inventory carried":[avg_inv_G,avg_inv_S,avg_inv_R,avg_inv_D,avg_inv_I],
    "Cost Incurred":[cost_G,cost_S,cost_R,cost_D,cost_I],
    "Total Cost":[total_cost_G,total_cost_S,total_cost_R,total_cost_D,total_cost_I]
}


########################### Shortages summary ##############################

#For district 1
I_s = 1
num_clinics_1 = 10
cost_1 = 0
avg_cost_1 = 0

for x in range(num_clinics_1):
	I = I_s + x
	for J in customers:
		for T in time:
			if(Sijt[T,J,I]!=0):
				cost_1 += Pjt[T-1][J-1]*Sijt[T,J,I].x
avg_cost_1 = cost_1/num_clinics_1

#For district 2
I_s = 11
num_clinics_2 = 6
cost_2 = 0
avg_cost_2 = 0

for x in range(num_clinics_2):
	I = I_s + x
	for J in customers:
		for T in time:
			if(Sijt[T,J,I]!=0):
				cost_2 += Pjt[T-1][J-1]*Sijt[T,J,I].x
avg_cost_2 = cost_2/num_clinics_2


#For district 3
I_s = 17
num_clinics_3 = 12
cost_3 = 0
avg_cost_3 = 0

for x in range(num_clinics_3):
	I = I_s + x
	for J in customers:
		for T in time:
			if(Sijt[T,J,I]!=0):
				cost_3 += Pjt[T-1][J-1]*Sijt[T,J,I].x
avg_cost_3 = cost_3/num_clinics_3


#For district 4
I_s = 29
num_clinics_4 = 12
cost_4 = 0
avg_cost_4 = 0

for x in range(num_clinics_4):
	I = I_s + x
	for J in customers:
		for T in time:
			if(Sijt[T,J,I]!=0):
				cost_4 += Pjt[T-1][J-1]*Sijt[T,J,I].x
avg_cost_4 = cost_4/num_clinics_4


#For district 5
I_s = 41
num_clinics_5 = 19
cost_5 = 0
avg_cost_5 = 0

for x in range(num_clinics_5):
	I = I_s + x
	for J in customers:
		for T in time:
			if(Sijt[T,J,I]!=0):
				cost_5 += Pjt[T-1][J-1]*Sijt[T,J,I].x
avg_cost_5 = cost_5/num_clinics_5

shortage_summary = {
    "CCP": ["1","2","3","4","5"],
    "Number of clinics":[num_clinics_1,num_clinics_2,num_clinics_3,num_clinics_4,num_clinics_5],
    "Total shortage cost Incurred":[cost_1,cost_2,cost_3,cost_4,cost_5],
    "Average shortage cost incurred per clinic":[avg_cost_1,avg_cost_2,avg_cost_3,avg_cost_4,avg_cost_5]
}

################################################# The End ######################################################### 