data = read.csv("no-pre.csv")
delivery_from_D = list()
for (d in 1:38)
{
        l = grepl(paste("Xidt.*,",as.character(d),",.*",sep = ""),data$X.from.D.to.I)
        delivery_from_D[[d]] = data$X.from.D.to.I[l][data$D.to.I.X[l]==1]
}
delivery_to_I = list()
for (i in 1:606)
{
        l = grepl(paste("Xidt.*,.*,",as.character(i),sep = ""),data$X.from.D.to.I)
        delivery_to_I[[i]] = data$X.from.D.to.I[l][data$D.to.I.X[l]==1]
}
delivery_at_T = list()
for (t in 1:12)
{
        l = grepl(paste("Xidt.*",as.character(t),",.*,.*",sep = ""),data$X.from.D.to.I)
        delivery_at_T[[t]] = data$X.from.D.to.I[l][data$D.to.I.X[l]==1]
}
top_D_from_I = list()
for (i in 1:606)
{
        l = grepl(paste("Did.*,",as.character(i),"]",sep = ""),data$Distance.from.D.to.I)
        dist = data.frame(data$Distance.from.D.to.I,data$D.to.I.dist)
        dist = dist[l,]
        dist = dist[order(dist$data.D.to.I.dist),]
        dist = dist$data.Distance.from.D.to.I
        top_D_from_I[[i]] = dist
}
place_of_D = list()
for (d in 1:38)
{
        l = c()
        for (i in 1:606)
        {
                count = 1
                for (t in top_D_from_I[[i]])
                {
                        
                        if(grepl(paste("Did.",as.character(d),",.*",sep = ""),t))
                        {
                                l = c(l,count)
                                break
                        }
                        count = count +1
                }       
        }
        place_of_D[[d]] = l
}
sum(place_of_D[[38]]<10)
