import xlrd
import xlsxwriter
import pandas as pd
import math

workbook = xlsxwriter.Workbook("weekly_demand_new.xls")
worksheet = workbook.add_worksheet()
worksheet.write(0,0,"i")
worksheet.write(0,1,"j")
#worksheet.write(0,2,"t")
worksheet.write(0,2,"demand")

wb = xlrd.open_workbook("Vaccine Quantities(Bihar).xlsx") 
sheet = wb.sheet_by_index(0)

i = 606
j = 3
row = 1
#t = 12
demand = 0

for I in range(1,i+1):
	for J in range(1,j+1):
		col = [6,7,8]
		demand = sheet.cell_value(I,col[J-1])
		worksheet.write(row,0,I)
		worksheet.write(row,1,J)
		#worksheet.write(row,2,T)
		worksheet.write(row,2,math.ceil(demand))
		row += 1
workbook.close()

read_file = pd.read_excel ("weekly_demand_new.xls") 
  
read_file.to_csv ("weekly_demand_new.csv", index = None, header=True) 
