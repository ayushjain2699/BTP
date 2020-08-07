d1 = read.csv("data1.csv")
d2 = read.csv("data2.csv")
d2$District = gsub(" $","",d2$District)

for (i in 2:nrow(d1))
{
        dist = d1[i,1]
        temp = d2[d2$District==dist,]
        l = nrow(temp)
        ilr_l = d1[i,2]
        ilr_s = d1[i,3]
        df_l = d1[i,4]
        df_s = d1[i,5]
        
        ilr_l_whole = as.integer(ilr_l/l)
        ilr_l = ilr_l-(ilr_l_whole*l)
        temp[,4] = ilr_l_whole
        if(ilr_l<l)
        {
            r = sample(l,ilr_l)
            temp[r,4] = temp[r,4]+1
        }
        
        ilr_s_whole = as.integer(ilr_s/l)
        ilr_s = ilr_s-(ilr_s_whole*l)
        temp[,5] = ilr_s_whole
        if(ilr_s<l)
        {
                r = sample(l,ilr_s)
                temp[r,5] = temp[r,5]+1
        }
        
        df_l_whole = as.integer(df_l/l)
        df_l = df_l-(df_l_whole*l)
        temp[,6] = df_l_whole
        if(df_l<l)
        {
                r = sample(l,df_l)
                temp[r,6] = temp[r,6]+1
        }
        
        df_s_whole = as.integer(df_s/l)
        df_s = df_s-(df_s_whole*l)
        temp[,7] = df_s_whole
        if(df_s<l)
        {
                r = sample(l,df_s)
                temp[r,7] = temp[r,7]+1
        }
        
        
        final = rbind(final,temp)
}
coln = c("S.No","District","Clinic","ILR(L)","ILR(S)","DF(L)","DF(S)")
colnames(final) = coln
write.csv(final,"data.csv")
