import xlwt
from xlwt import Workbook
import xlrd

wb = Workbook()
loc = "C:/Users/Shanmukhi/Dropbox/Shanmukhi Ayush 2020/Data/Capacities of CCPs/Capacity of Cold chain Points(Bihar).xlsx"
workbook = xlrd.open_workbook(loc) 
sheet = workbook.sheet_by_index(4)

sheet1 = wb.add_sheet('RVS')
sheet1.write(0,0,'r')
sheet1.write(0,1,'t')
sheet1.write(0,2,'Capacity')

r = 9
t = 12

row = 1
for R in range(1,r+1):
	for T in range(1,t+1):
		sheet1.write(row,0,R)
		sheet1.write(row,1,T)
		sheet1.write(row,2,sheet.cell_value(R-1,2))
		row = row + 1

wb.save("C:/Users/Shanmukhi/Dropbox/Shanmukhi Ayush 2020/Data/Capacities of CCPs/capacity_RVS.xls")
#wb.save("C:/Users/Shanmukhi/Desktop/test2.xls")