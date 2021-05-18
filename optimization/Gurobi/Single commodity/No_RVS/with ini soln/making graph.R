data = read.csv("no.csv",header = F,sep  ="")
l = grepl("H",data$V1)
data = data[!l,]
time = data$V10
obj = data$V6
obj = substr(obj,start = 1,stop = nchar(obj)-1)
time = substr(time,start = 1,stop = nchar(time)-1)
obj = as.numeric(obj)
time = as.numeric(time)
df = data.frame("Time" = time, "Obj" = obj)
library(xlsx)
write.xlsx(x = df,file = "no.xlsx",row.names = F)
