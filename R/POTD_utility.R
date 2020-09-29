library(Rdimtools)
library(class)
library(caret)
library(transport)
library(Rfast)

matpower = function(a,alpha){
  a = round((a + t(a))/2,7); tmp = eigen(a)
  return(tmp$vectors%*%diag((tmp$values)^alpha)%*%t(tmp$vectors))}


train_test_split = function(X, y, test_size, seed){
  set.seed(seed)
  n=nrow(X)
  test_id = sample(n, round(n*test_size))
  list_final = list("X_train" = X[-test_id,], "X_test" = X[test_id,], 
                    "y_train" = y[-test_id], "y_test" = y[test_id])
  return(list_final)
}


NN_score = function(X_test, y_train, y_test, k, fit){
  X_train_dr = fit$Y
  X_test_dr = sweep(X_test, 2, fit$trfinfo$mean)%*%fit$projection
  y_test_pred = class::knn(X_train_dr, X_test_dr, y_train, k = k, prob=F)
  cm = confusionMatrix(y_test_pred, as.factor(y_test))
  return(unname(cm$overall[1]))
}


NN_score2 = function(X_train, X_test, y_train, y_test, k, dirct, col_mean){
  X_train_dr = sweep(X_train, 2, col_mean)%*%dirct
  X_test_dr = sweep(X_test, 2, col_mean)%*%dirct
  y_test_pred = class::knn(X_train_dr, X_test_dr, y_train, k = k, prob=F)
  cm = confusionMatrix(y_test_pred, as.factor(y_test))
  return(unname(cm$overall[1]))
}


y_refactor = function(y_temp){
  slice_cate = names(table(y_temp))
  H = length(slice_cate)
  y = rep(0, length(y_temp))
  for(idx in 1:H){
    temp_index = which(y_temp == slice_cate[idx])
    y[temp_index] = idx
  }
  return(y)
}



potd = function(X_train, y_train, ndim, with_sigma=F){
  pp = ncol(X_train)
  slice_cate = names(table(y_train))
  H = length(slice_cate)
  direct_meta = matrix(0,0,pp)
  
  if(with_sigma){
    signrt=matpower(cova(X_train),-1/2) 
    X_train = X_train%*%signrt
  }
  
  
  for(i in 1:(H-1)){
    for(j in (i+1):H){
      data_source = X_train[y_train == slice_cate[i],]
      data_target = X_train[y_train == slice_cate[j],]
      
      NN = nrow(data_source); MM = nrow(data_target)
      a = rep(1, NN)/NN;  b = rep(1, MM)/MM
      
      ds_wt <- wpp(data_source, a)
      dt_wt <- wpp(data_target, b)
      res <- transport(ds_wt, dt_wt, method="shortsimplex")

      direct = matrix(0,nrow(res),pp)
      for(k in 1:nrow(res)){
        direct[k,] = (data_source[res$from[k],] -data_target[res$to[k],])*res$mass[k]
      }
      
      direct_meta = rbind(direct_meta, direct)
    }
  }
  ###
  direct_meta = rbind(direct_meta, -direct_meta)
  ###
  svd_res = svd(direct_meta, nu=0, nv = ndim)
  res = svd_res$v
  
  if(with_sigma){
    res = signrt%*%res
  }
  
  return(res)
}


space_dist=function(bt.e, bt.t, type=1){
  if(type==1){
    nn=nrow(bt.t)
    d1=diag(nn)-bt.e%*%t(bt.e)
    d2=d1%*%bt.t
    d=sum(d2^2)
    return(d)
  }
  if(type==2){
    pb1 = bt.e%*%(t(bt.e)%*%bt.e)%*%t(bt.e)
    pb2 = bt.t%*%(t(bt.t)%*%bt.t)%*%t(bt.t)
    d = sum((pb1-pb2)^2)
    return(d)
  }
}
