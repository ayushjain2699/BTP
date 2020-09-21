import xlwt
from xlwt import Workbook
import xlrd

wb = Workbook()
loc = "C:/Users/Shanmukhi/Dropbox/Shanmukhi Ayush 2020/Data/Distances between stores/Bihar/RVS-DVS.xlsx"
workbook = xlrd.open_workbook(loc) 
sheet = workbook.sheet_by_index(0)

sheet1 = wb.add_sheet('Distances')
sheet1.write(0,0,'r')
sheet1.write(0,1,'d')
sheet1.write(0,2,'Distance')

d = 38
r = 9

row = 1
for R in range(1,r+1):
	for D in range(1,d+1):
		sheet1.write(row,0,R)
		sheet1.write(row,1,D)
		sheet1.write(row,2,sheet.cell_value(R,D))
		row = row + 1

wb.save("C:/Users/Shanmukhi/Dropbox/Shanmukhi Ayush 2020/Data/Distances between stores/Bihar/distances_rd.xls")
#wb.save("C:/Users/Shanmukhi/Desktop/test4.xls")