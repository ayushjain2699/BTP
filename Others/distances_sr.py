import xlwt
from xlwt import Workbook
import xlrd

wb = Workbook()
loc = "C:/Users/Shanmukhi/Dropbox/Shanmukhi Ayush 2020/Data/Distances between stores/Bihar/SVS-RVS.xlsx"
workbook = xlrd.open_workbook(loc) 
sheet = workbook.sheet_by_index(0)

sheet1 = wb.add_sheet('Distances')
sheet1.write(0,0,'s')
sheet1.write(0,1,'r')
sheet1.write(0,2,'Distance')

s = 1
r = 9

row = 1
for S in range(1,s+1):
	for R in range(1,r+1):
		sheet1.write(row,0,S)
		sheet1.write(row,1,R)
		sheet1.write(row,2,sheet.cell_value(1,R))
		row = row + 1

wb.save("C:/Users/Shanmukhi/Dropbox/Shanmukhi Ayush 2020/Data/Distances between stores/Bihar/distances_sr.xls")
#wb.save("C:/Users/Shanmukhi/Desktop/test3.xlsx")