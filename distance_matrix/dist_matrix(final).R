f = "https://dev.virtualearth.net/REST/v1/Routes/DistanceMatrix?origins="
s = "&destinations="
t = "&travelMode=driving&key=AjPUy6L3ZouH31A3BNigLLscOSzXqecAAKedFQDMjFG3paD0-n5TkuCNBRiVqCiU"
library(data.table)
library(httr)
library(jsonlite)

data = read.csv("svs.csv") ###
data = data[,]
data$latitude = as.character(data$latitude)
data$longitude = as.character(data$longitude)
data$cor = paste(data$latitude,data$longitude,sep = ",")
origin = data$cor[1]
if(nrow(data)>1){
        for(i in 2:nrow(data)){
                origin = paste(origin,data[i,8],sep=";")
        }
}
index_names_origin = data$name


for (i in 1:21)
{
        print(30*i+1)
        data = read.csv("dvs.csv")   ####
        e = min(606,30*i)
        data = data[(30*(i-1)+1):e,]
        data$latitude = as.character(data$latitude)
        data$longitude = as.character(data$longitude)
        data$cor = paste(data$latitude,data$longitude,sep = ",")
        dest = data$cor[1]
        if(nrow(data)>1){
                for(j in 2:nrow(data)){
                        dest = paste(dest,data[j,8],sep=";")
                }
        }
        index_names_dest = data$name
        
        url = paste(f,origin,s,dest,t,sep = "")
        js = GET(url)
        js = content(js)
        json1 = jsonlite::fromJSON(toJSON(js))
        data = json1$resourceSets$resources[[1]]$results
        data = as.data.table(data)
        data$destinationIndex = as.numeric(data$destinationIndex)
        data$originIndex = as.numeric(data$originIndex)
        data$travelDistance = as.numeric(data$travelDistance)
        data = data[,c(1,2,4)]
        data = dcast(data,originIndex~destinationIndex,value.var = "travelDistance")
        
        
        colnames(data) = c("SVS/DVS",index_names_dest) ###
        data$`SVS/DVS` = index_names_origin    ###
        if(i==1)
        {
                write.csv(data,file = ".\\SVS-DVS.csv",row.names = F)
        }
        if(i>1)
        {
                final = read.csv(".\\DVS-PHC.csv")
                final = cbind(final,data[,-1])
                write.csv(final,file = ".\\DVS-PHC.csv",row.names = F) 
        }
        #write.csv(data,file = ".\\RVS-DVS.csv",row.names = F)   ###
}  
