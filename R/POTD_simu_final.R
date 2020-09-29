library(Rdimtools)
library(class)
library(caret)
library(foreach)
library(doParallel)
source("D:/Dropbox/Density Estimation/code/POTD/Final_code/POTD_utility.R")
library(dr)


N=400
nloop = 10



result_mean = matrix(0,0,16)
result_sd = matrix(0,0,16)




######################################################################
cl <- makeCluster(5)
registerDoParallel(cl)


for(p in c(10,20,30)){ 
  for(fun_type in 1:4){
    if(fun_type==1){
      y_generate = function(X){fX = sin(X[,1]/(X[,2]^2)); y_refactor(sign( fX +0.2*rnorm(N)))}
      ndim = 2
      B_true = cbind(c(1,rep(0,p-1)),c(0,1,rep(0,p-2)))
    }
    if(fun_type==2){
      y_generate = function(X){fX = (X[,1]+0.5)*(X[,2]-0.5)^2; y_refactor(sign( fX +0.2*rnorm(N)))}
      ndim = 2
      B_true = cbind(c(1,rep(0,p-1)),c(0,1,rep(0,p-2)))
    }
    if(fun_type==3){
      y_generate = function(X){fX = log(X[,1]^2)*(X[,2]^2+X[,3]^2/2+X[,4]^2/4); y_refactor(sign( fX +0.2*rnorm(N)))}
      ndim = 4
      B_true = cbind(c(1,rep(0,p-1)), c(0,1,rep(0,p-2)), c(0,0,1,rep(0,p-3)), c(0,0,0,1,rep(0,p-4)))
    }
    if(fun_type==4){
      y_generate = function(X){fX = sin(X[,1])/(X[,2]*X[,3]*X[,4]); y_refactor(sign( fX +0.2*rnorm(N)))}
      ndim = 4
      B_true = cbind(c(1,rep(0,p-1)), c(0,1,rep(0,p-2)), c(0,0,1,rep(0,p-3)), c(0,0,0,1,rep(0,p-4)))
    }
  
    ptm <- proc.time()
    aaa<- foreach(i = 1:nloop, .packages=c("class", "caret", "Rdimtools", "transport", "Rfast", "dr"), .combine="rbind") %dopar% {
      
      set.seed(1234+13*i)
      if(p==30&&fun_type==4) set.seed(1234+12*i)
      X = matrix(runif(N*p, -2, 2), N)
      y = y_generate(X)
      
      res_ammc = space_dist(B_true, do.ammc(X, y, ndim)$projection)
      res_anmm = space_dist(B_true, do.anmm(X, y, ndim)$projection)
      res_dagdne = space_dist(B_true, do.dagdne(X, y, ndim)$projection)
      res_dne = space_dist(B_true, do.dne(X, y, ndim)$projection)
      res_elde = space_dist(B_true, do.elde(X, y, ndim)$projection)
      res_lde = space_dist(B_true, do.lde(X, y, ndim)$projection)
      res_ldp = space_dist(B_true, do.ldp(X, y, ndim)$projection)
      res_lpfda = space_dist(B_true, do.lpfda(X, y, ndim)$projection)
      res_lsda = space_dist(B_true, do.lsda(X, y, ndim)$projection)
      res_mmc = space_dist(B_true, do.mmc(X, y, ndim)$projection)
      res_modp = space_dist(B_true, do.modp(X, y, ndim)$projection)
      res_msd = space_dist(B_true, do.msd(X, y, ndim)$projection)
      res_odp = space_dist(B_true, do.odp(X, y, ndim)$projection)
      res_save = space_dist(B_true, dr(y~.,data=as.data.frame(X), method="save")$evectors[,1:ndim])
      res_phd = space_dist(B_true, dr(y~.,data=as.data.frame(X), method="phdy")$evectors[,1:ndim])
      #res_potd = space_dist(B_true, potd(X, y, ndim))
      res_potd2 = space_dist(B_true, potd(X, y, ndim, with_sigma = T))
      
      
      c(res_ammc, res_anmm, res_dagdne, res_dne, res_elde, res_lde, res_ldp, res_lpfda, res_lsda, 
        res_mmc, res_modp, res_msd, res_odp, res_save, res_phd, res_potd2)
    }
    
    
   result_mean = rbind(result_mean, round(apply(aaa,2,mean),2))
   result_sd = rbind(result_sd, round(apply(aaa,2,sd),2))
  

   cat("p=",p, "; fun_type=",fun_type, "; time=", (proc.time()-ptm)[3],"\n")

  }
}

stopCluster(cl)


########################################################################################
result_mean = as.data.frame(result_mean)
result_sd = as.data.frame(result_sd)
names(result_mean) = c("AMMC", "ANMM", "DAGDNE", "DNE", "ELDE", "LDE", "LDP", "LPFDA", "LSDA",
                       "MMC", "MODP", "MSD", "ODP", "SAVE", "PHD", "POTD")
names(result_sd) = c("AMMC", "ANMM", "DAGDNE", "DNE", "ELDE", "LDE", "LDP", "LPFDA", "LSDA",
                     "MMC", "MODP", "MSD", "ODP", "SAVE", "PHD", "POTD")

result_mean











