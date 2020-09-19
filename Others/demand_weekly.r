time = 12
df <- data.frame(matrix(ncol = 4, nrow = 606*time))
demand = read.xlsx("Vaccine Quantities(Bihar).xlsx",sheetIndex = 1)
demand =  demand[,1:9]
names = c("i","j","t","demand")
colnames(df) = names
df$j = 1

demand = demand$Total.old.age.plus.health.workers.
for (i in 0:605)
{
     for (j in 1:12)
     {
        df[12*i+j,1] = i+1
        df[12*i+j,3] = j
        if(demand[i+1]>160*7)
        {
                put = 160*7
                demand[i+1] = demand[i+1]-160*7
        }
        else
        {
                put = demand[i+1]
                demand[i+1] = 0
        }
        df[12*i+j,4] = put
     }
}
write.csv(df,"./demand_weekly.csv")
