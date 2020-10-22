import xlrd
import xlsxwriter
import pandas as pd
import math

workbook = xlsxwriter.Workbook("weekly_demand.xls")
worksheet = workbook.add_worksheet()
worksheet.write(0,0,"i")
worksheet.write(0,1,"j")
worksheet.write(0,2,"t")
worksheet.write(0,3,"demand")

wb = xlrd.open_workbook("Vaccine Quantities(Bihar).xlsx") 
sheet = wb.sheet_by_index(0)

i = 606
j = 2
row = 1
t = 12
demand = 0

for I in range(1,i+1):
	for J in range(1,j+1):
		col = 5+J
		demand = sheet.cell_value(I,col)/12
		for T in range(1,t+1):
			worksheet.write(row,0,I)
			worksheet.write(row,1,J)
			worksheet.write(row,2,T)
			worksheet.write(row,3,math.ceil(demand))
			row += 1
workbook.close()

read_file = pd.read_excel ("weekly_demand.xls") 
  
read_file.to_csv ("weekly_demand.csv", index = None, header=True) 
