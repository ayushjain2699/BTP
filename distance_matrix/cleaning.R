data = read.csv("PHC.csv",header = F)
data$V3 = gsub("\n"," ",data$V3)
data$V2 = gsub("\n"," ",data$V2)
rm = c("RH","URBAN","Urban","urban","ANM","APHC","-1","-2","AVD","SDH","CCE","CCH","CCP","CVS","CHC","DH","DVS","eVIN","PHC","RVS","SVS","SDH","SE","SS","SVS","UHC","UNDP","UPHC","HSC","Rural","rural")
for (i in 1:(length(rm))){
        data$V3 = gsub(rm[i],"",data$V3)
}
state = "Bihar"
data$V4 = paste(data$V3,data$V2,sep = ", ")
data$V4 = paste(data$V4,state,"India",sep = ", ")
write.csv(data,"PHC_Corrected.csv")
