import numpy as np
import pandas as pd
import xlsxwriter

#import gurobipy as gp
#from gurobipy import GRB

####################### INDEX ################################

j = 1  #Customer sub index
m = 1  #Manufacturer sub index
g = 1  #GMSD index
s = 1  #State sub index
r = 9  #Region sub index
d = 6  #District sub index
i = 118 #Clinic sub index
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
#l = GRB.INFINITY #large number for consistency constraints
fraction_storage = 1 #Fraction of total capacity in cold chain points to be considered for COVID-19 vaccine
fraction_transport = 1 #Fraction of total capacity in vehicles to be considered for COVID-19 vaccine

#Transportation cost
diesel_cost = 14
booking_cost = 5000
np.random.seed(133)

#Distances
Dgm = [[1000]] #From M to G (confirm)
Dsg = [[550]] #From G to S (confirm)

#From R to D
# df_Ddr = pd.read_csv("distances_rd.csv")
# Ddr = [[0 for D in range(d)] for R in range(r)]
# for index in df_Ddr.index:
# 	if (df_Ddr['d'][index] > 5):
# 		continue
# 	Ddr[df_Ddr['r'][index]-1][df_Ddr['d'][index]-1] = df_Ddr['Distance'][index]

# print(Ddr)

# df_Did = pd.read_csv("distances_di.csv")
# Did = [[0 for I in range(i)] for D in range(d)]
# for index in df_Did.index:
# 	if (df_Did['d'][index] > 5 or df_Did['i'][index] > 59):
# 		continue
# 	Did[df_Did['d'][index]-1][df_Did['i'][index]-1] = df_Did['Distance'][index]
# print(Did)

# df_demand = pd.read_csv("demand_weekly.csv")
# dijt = [[[0 for I in range(1,i+1)] for J in range(j)] for T in range(1,t+1)]
# for index in df_demand.index:
# 	if (df_demand['i'][index] > 59):
# 		break

# 	dijt[df_demand['t'][index]-1][df_demand['j'][index]-1][df_demand['i'][index]-1] = df_demand['demand'][index]

# print(dijt)

# df_bdt = pd.read_csv("capacity_DVS.csv")
# Bdt = [[0 for D in range(d)] for T in range(t)]
# for index in df_bdt.index:
# 	if(df_bdt['d'][index] > 5):
# 		break
# 	Bdt[df_bdt['t'][index]-1][df_bdt['d'][index]-1] = fraction_storage*df_bdt['Capacity'][index]

# print(Bdt)

# df_bit = pd.read_csv("capacity_clinics.csv")
# Bit = [[0 for I in range(i)] for T in range(t)]
# for index in df_bit.index:
# 	if(df_bit['i'][index] > 59):
# 		break
# 	Bit[df_bit['t'][index]-1][df_bit['i'][index]-1] = fraction_storage*df_bit['Capacity'][index]

# print(Bit)

# #From R to D
# df_Ddr = pd.read_csv("distances_rd.csv")
# Ddr = [[0 for D in range(d)] for R in range(r)]
# for index in df_Ddr.index:
# 	if (df_Ddr['d'][index] > 12 or df_Ddr['d'][index] < 7):
# 		continue
# 	Ddr[df_Ddr['r'][index]-1][df_Ddr['d'][index]-7] = df_Ddr['Distance'][index]

# print(Ddr)

#From D to I
df_Did = pd.read_csv("distances_di.csv")
Did = [[0 for I in range(i)] for D in range(d)]
for index in df_Did.index:
	if (df_Did['d'][index] > 12 or df_Did['d'][index] < 7  or df_Did['i'][index] > 194 or df_Did['i'][index] < 77):
		continue
	Did[df_Did['d'][index]-7][df_Did['i'][index]-77] = df_Did['Distance'][index]

print(Did)