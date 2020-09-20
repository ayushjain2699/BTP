import xlwt
from xlwt import Workbook
import xlrd

wb = Workbook()
loc = "C:/Users/Shanmukhi/Dropbox/Shanmukhi Ayush 2020/Data/Capacities of CCPs/Capacity of Cold chain Points(Bihar).xlsx"
workbook = xlrd.open_workbook(loc) 
sheet = workbook.sheet_by_index(3)

sheet1 = wb.add_sheet('Clinics')
sheet1.write(0,0,'i')
sheet1.write(0,1,'t')
sheet1.write(0,2,'Capacity')

i = 606
t = 12

row = 1
for I in range(1,i+1):
	for T in range(1,t+1):
		sheet1.write(row,0,I)
		sheet1.write(row,1,T)
		sheet1.write(row,2,sheet.cell_value(I,7))
		row = row + 1

wb.save("C:/Users/Shanmukhi/Dropbox/Shanmukhi Ayush 2020/Data/Capacities of CCPs/capacity_clinics.xls")