source("D:/Dropbox/Density Estimation/code/POTD/Final_code/POTD_utility.R")


set.seed(1000)
N=300
pp=10
z1 = rnorm(N,0,1)
z2 = rnorm(N,0,1)
theta = rnorm(N, pi, (0.25*pi)^2)
x1 = 20*cos(theta)+z1
x2 = 20*sin(theta)+z2
x_rest = matrix(rnorm(N*8,0,1),N)
X=cbind(x1,x2,x_rest)

z1 = rnorm(N,0,1)
z2 = rnorm(N,0,1)
theta = rnorm(N, 0, (0.25*pi)^2)
x1 = 20*cos(theta)+z1
x2 = 20*sin(theta)-20+z2
x_rest = matrix(rnorm(N*8,0,1),N)
Y=cbind(x1,x2,x_rest)


X=scale(X)
Y=scale(Y)
x=rbind(X,Y)
y=c(rep(-1,N),rep(1,N))
plot(x[,1], x[,2], type = "n", xlab = "X1", ylab  = "X2")
points(X[,1], X[,2], col = 4, pch = "+")
points(Y[,1], Y[,2], col = 2)






######################################################
## POTD
######################################################

res_potd = potd(x, y, 2)
xx = x%*%res_potd

plot(xx[,1], xx[,2], type = "n", xlab = "POTD_1", ylab  = "POTD_2")
points(xx[1:N, 1], xx[1:N, 2], col = 4, pch = "+")
points(xx[1:N+N, 1], xx[1:N+N,2], col = 2)








