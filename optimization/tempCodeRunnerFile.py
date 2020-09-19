or I in clinics for T in time),name = "clinic_cap")

#Production capacity constraints
production_cap = model.addConstrs((gp.quicksum(Qgmt[T,M,G] for G in gmsd)<= Bmt[T-1][M-1] for M in manufacturers for T in time)
                                ,name = "production_cap")

#Facility selection constraints
fac_sg = model.addConstrs((gp.quicksum(Xsgt[T,G,S] for G in gmsd)<=1 for S in svs for T in time),name = "fac_sg")
fac_rs = model.addConstrs((gp.quicksum