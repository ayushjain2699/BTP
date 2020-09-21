import xlwt
from xlwt import Workbook
import xlrd

wb = Workbook()
loc = "C:/Users/Shanmukhi/Dropbox/Shanmukhi Ayush 2020/Data/Distances between stores/Bihar/DVS-PHC.xlsx"
workbook = xlrd.open_workbook(loc) 
sheet = workbook.sheet_by_index(0)

sheet1 = wb.add_sheet('Distances')
sheet1.write(0,0,'d')
sheet1.write(0,1,'i')
sheet1.write(0,2,'Distance')

d = 38
i = 606

row = 1
for D in range(1,d+1):
	for I in range(1,i+1):
		sheet1.write(row,0,D)
		sheet1.write(row,1,I)
		sheet1.write(row,2,sheet.cell_value(D,I))
		row = row + 1

wb.save("C:/Users/Shanmukhi/Dropbox/Shanmukhi Ayush 2020/Data/Distances between stores/Bihar/distances_di.xls")
#wb.save("C:/Users/Shanmukhi/Desktop/test5.xls")