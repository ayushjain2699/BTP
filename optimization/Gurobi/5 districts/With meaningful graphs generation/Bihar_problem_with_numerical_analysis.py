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
	if (df_Ddr['d'][index] > d):
		continue
	Ddr[df_Ddr['r'][index]-1][df_Ddr['d'][index]-1] = df_Ddr['Distance'][index]

#From D to I
df_Did = pd.read_csv("distances_di.csv")
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
	if (df_demand['i'][index] > i):
		break

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
	if(df_bdt['d'][index] > d):
		break
	Bdt[df_bdt['t'][index]-1][df_bdt['d'][index]-1] = fraction_storage*df_bdt['Capacity'][index]


df_bit = pd.read_csv("capacity_clinics.csv")
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
    print(v.varName,"=", v.x)
    names.append(v.varName)
    sol.append(v.x)


###########################Printing the results to excel file################## 

workbook_1 = xlsxwriter.Workbook("Numerical Analysis/CCPs ordering.xlsx")

import pandas as pd
start = (g+s+r+d+i)*t 
T = 1
for T in range(1,13):
    x = start + (T-1)*m*g
    address = "Excel files/"
    address += str(T)
    address += "-M to G"
    address += ".xlsx"
    print(address)

    sheet_name = str(T)
    worksheet_1 = workbook_1.add_worksheet(sheet_name)
    # worksheet_1.write(0,0,"Manufacturer")
    worksheet_1.write(0,0,"GMSD")
    worksheet_1.write(0,1,"SVS")
    worksheet_1.write(0,2,"RVS")
    worksheet_1.write(0,3,"DVS")
    worksheet_1.write(0,4,"Clinics")

    #M to G
    workbook = xlsxwriter.Workbook(address)
    worksheet = workbook.add_worksheet("M to G")
    row = 0
    row_1 = 1
    count = 0
    for M in range(m):
        for G in range(g):
            if (round(sol[x+count]) != 0):
                worksheet.write(row,0,M+1)
                worksheet.write(row,1,G+1)
                worksheet.write(row,2,sol[x+count])
                worksheet_1.write(row_1,0,G+1)
                count += 1
                row += 1
                row_1 += 1
            else:
                count += 1
    print("M to G done")
    workbook.close()
    read_file = pd.read_excel (address) 
    csv_address ="CSV files/"
    csv_address += str(T)
    csv_address += "-M to G"
    csv_address += ".csv"
    read_file.to_csv (csv_address, index = None, header=True) 
    

    x = x + (t-T+1)*m*g +(T-1)*s*g
    address = "Excel files/"
    address += str(T)
    address += "-G to S"
    address += ".xlsx"
    print(address)
    workbook = xlsxwriter.Workbook(address)
    worksheet = workbook.add_worksheet("G to S")
    row = 0
    count = 0
    row_1 = 1
    for G in range(g):
        for S in range(s):
            if (round(sol[x+count]) != 0):
                worksheet.write(row,0,G+1)
                worksheet.write(row,1,S+1)
                worksheet.write(row,2,sol[x+count])
                worksheet_1.write(row_1,1,S+1)
                row += 1
                count += 1
                row_1 += 1
            else:
                count += 1
    print("G to S done")
    workbook.close()
    read_file = pd.read_excel (address) 
    csv_address ="CSV files/"
    csv_address += str(T)
    csv_address += "-G to S"
    csv_address += ".csv"
    read_file.to_csv (csv_address, index = None, header=True) 

    x = x + (t-T+1)*g*s + (T-1)*r*s
    address = "Excel files/"
    address += str(T)
    address += "-S to R"
    address += ".xlsx"
    print(address)
    workbook = xlsxwriter.Workbook(address)
    worksheet = workbook.add_worksheet("S to R")
    row = 0
    count = 0
    row_1 = 1
    for S in range(s):
        for R in range(r):
            if (round(sol[x+count]) != 0):
                worksheet.write(row,0,S+1)
                worksheet.write(row,1,R+1)
                worksheet.write(row,2,sol[x+count])
                worksheet_1.write(row_1,2,R+1)
                row += 1
                count += 1
                row_1 += 1
            else:
                count += 1
    print("S to R done")
    workbook.close()
    read_file = pd.read_excel (address) 
    csv_address ="CSV files/"
    csv_address += str(T)
    csv_address += "-S to R"
    csv_address += ".csv"
    read_file.to_csv (csv_address, index = None, header=True) 

    x = x + (t-T+1)*s*r + (T-1)*r*d
    address = "Excel files/"
    address += str(T)
    address += "-R to D"
    address += ".xlsx"
    print(address)
    workbook = xlsxwriter.Workbook(address)
    worksheet = workbook.add_worksheet("R to D")
    row = 0
    count = 0
    row_1 = 1
    for R in range(r):
        for D in range(d):
            if (round(sol[x+count]) != 0):
                worksheet.write(row,0,R+1)
                worksheet.write(row,1,D+1)
                worksheet.write(row,2,sol[x+count])
                worksheet_1.write(row_1,3,D+1)
                row += 1
                count += 1
                row_1 += 1
            else:
                count += 1
    print("R to D done")
    workbook.close()
    read_file = pd.read_excel (address) 
    csv_address ="CSV files/"
    csv_address += str(T)
    csv_address += "-R to D"
    csv_address += ".csv"
    read_file.to_csv (csv_address, index = None, header=True) 

    x = x + (t-T+1)*r*d + (T-1)*d*i
    address = "Excel files/"
    address += str(T)
    address += "-D to I"
    address += ".xlsx"
    print(address)
    workbook = xlsxwriter.Workbook(address)
    worksheet = workbook.add_worksheet("D to I")
    row = 0
    count = 0
    row_1 = 1
    for D in range(d):
        for I in range(i):
            if (round(sol[x+count]) != 0):
                worksheet.write(row,0,D+1)
                worksheet.write(row,1,I+1)
                worksheet.write(row,2,sol[x+count])
                worksheet_1.write(row_1,4,I+1)
                row += 1
                count += 1
                row_1 += 1
            else:
                count += 1
    print("D to I done")
    workbook.close()
    # workbook.close()
    read_file = pd.read_excel (address) 
    csv_address ="CSV files/"
    csv_address += str(T)
    csv_address += "-D to I"
    csv_address += ".csv"
    read_file.to_csv (csv_address, index = None, header=True)
workbook_1.close()

###############################Shortage analysis########################
workbook_2 = xlsxwriter.Workbook("Numerical Analysis/Shortages.xlsx")
worksheet_2 = workbook_1.add_worksheet("Clinics with shortages")
worksheet_2.write(0,0,"T")
worksheet_2.write(0,1,"J")
worksheet_2.write(0,2,"I")
worksheet_2.write(0,3,"Shortage")
row = 1
for T in time:
	for J in customers:
		for I in clinics:
			if(round(Sijt[T,J,I].x)!=0):
				worksheet_2.write(row,0,T)
				worksheet_2.write(row,1,J)
				worksheet_2.write(row,2,I)
				worksheet_2.write(row,3,round(Sijt[T,J,I].x))
				row = row + 1


##################################### Total Cost ##############################################

############ Transportation Cost ##############
for T in time:
    for M in manufacturers:
        for G in gmsd:
            transport_cost = transport_cost + Kgmt[T-1][M-1][G-1]*Ngmt[T,M,G].x
for T in time:
    for G in gmsd:
        for S in svs:
            transport_cost = transport_cost + Ksgt[T-1][G-1][S-1]*Nsgt[T,G,S].x
for T in time:
    for S in svs:
        for R in rvs:
            transport_cost = transport_cost + Krst[T-1][S-1][R-1]*Nrst[T,S,R].x
for T in time:
    for R in rvs:
        for D in dvs:
            transport_cost = transport_cost + Kdrt[T-1][R-1][D-1]*Ndrt[T,R,D].x
for T in time:
    for D in dvs:
        for I in clinics:
            transport_cost = transport_cost + Kidt[T-1][D-1][I-1]*Nidt[T,D,I].x

########### Inventory COst #############

for T in time:
    for G in gmsd:
        inventory_cost = inventory_cost + hgt[T-1][G-1]*Igt[T,G].x
for T in time:
    for S in svs:
        inventory_cost = inventory_cost + hst[T-1][S-1]*Ist[T,S].x
for T in time:
    for R in rvs:
        inventory_cost = inventory_cost + hrt[T-1][R-1]*Irt[T,R].x
for T in time:
    for D in dvs:
        inventory_cost = inventory_cost + hdt[T-1][D-1]*Idt[T,D].x
for T in time:
    for I in clinics:
        inventory_cost = inventory_cost + hit[T-1][I-1]*Iit[T,I].x

########### Shortage Cost ################
for T in time:
    for I in clinics:
        for J in customers:
            shortage_cost = shortage_cost + Pjt[T-1][J-1]*Sijt[T,J,I].x

########### Consumption Cost ################
for T in time:
    for I in clinics:
        for J in customers:
            consumption_cost = consumption_cost + 0*Wijt[T,J,I].x

############ Ordering Cost ##############
for T in time:
    for M in manufacturers:
        for G in gmsd:
            ordering_cost = ordering_cost + Cgmt[T-1][M-1][G-1]*Xgmt[T,M,G].x
for T in time:
    for G in gmsd:
        for S in svs:
            ordering_cost = ordering_cost + Csgt[T-1][G-1][S-1]*Xsgt[T,G,S].x
for T in time:
    for S in svs:
        for R in rvs:
            ordering_cost = ordering_cost + Crst[T-1][S-1][R-1]*Xrst[T,S,R].x
for T in time:
    for R in rvs:
        for D in dvs:
            ordering_cost = ordering_cost + Cdrt[T-1][R-1][D-1]*Xdrt[T,R,D].x
for T in time:
    for D in dvs:
        for I in clinics:
            ordering_cost = ordering_cost + Cidt[T-1][D-1][I-1]*Xidt[T,D,I].x

################ Total ###############

total_cost =  transport_cost + inventory_cost + shortage_cost + consumption_cost + ordering_cost





