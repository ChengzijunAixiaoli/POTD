import os
import numpy as np
import math
import matplotlib.pyplot as plt
os.chdir("D:/Dropbox/Density Estimation/code/POTD/Final_code")
from POTD_utility import potd, saveDir2
#from scipy.linalg import sqrtm
from sliced import SlicedInverseRegression
from sklearn.preprocessing import scale
#from sliced.save import SlicedAverageVarianceEstimation



# =============================================================================
# #Generate the synthetic data
# =============================================================================
N = 300
pp = 10


z1 = np.random.normal(0, 1, N)
z2 = np.random.normal(0, 1, N)
theta = np.random.normal(math.pi, (0.25*math.pi)**2, N)
x1 = 20*np.cos(theta)+z1
x2 = 20*np.sin(theta)+z2
x_rest=np.random.normal(0, 1, N*8).reshape(N, 8)
X=np.c_[x1,x2,x_rest]


z1 = np.random.normal(0, 1, N)
z2 = np.random.normal(0, 1, N)
theta = np.random.normal(0, (0.25*math.pi)**2, N)
x1 = 20*np.cos(theta)+z1
x2 = 20*np.sin(theta)-20+z2
x_rest=np.random.normal(0, 1, N*8).reshape(N, 8)
Y=np.c_[x1,x2,x_rest]


X = scale(X)
Y = scale(Y)


plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c='r', zorder=10, alpha=0.7)
plt.scatter(Y[:,0], Y[:,1], c='b', zorder=10, alpha=0.7)
plt.axis('equal')
plt.xlabel("X1",size=25)
plt.ylabel("X2",size=25)
plt.show() 


# =============================================================================
# 
# 
# SAVE 
# 
# 
# =============================================================================

X_train = np.concatenate((X, Y))
y_train = np.repeat(list(range(2)), N, axis=0)


eigen_meta, dir_meta = saveDir2(X_train, y_train, 2)

plt.figure(figsize=(6,6))
plt.scatter(X@dir_meta[:,0], X@dir_meta[:,1], c='r', zorder=10, alpha=0.7)
plt.scatter(Y@dir_meta[:,0], Y@dir_meta[:,1], c='b', zorder=10, alpha=0.7)
plt.axis('equal')
plt.xlabel("SAVE-1",size=25)
plt.ylabel("SAVE-2",size=25)
plt.show() 





# =============================================================================
# #SIR
# =============================================================================


sir = SlicedInverseRegression(n_directions=2-1, n_slices=2)
sir.fit(X_train, y_train)
sir.directions_

plt.figure(figsize=(6,6))
plt.scatter(X@np.transpose(sir.directions_), np.zeros([N]), c='r', zorder=10, alpha=0.7)
plt.scatter(Y@np.transpose(sir.directions_), np.zeros([N])+1, c='b', zorder=10, alpha=0.7)
plt.axis('equal')
plt.xlabel("SIR-1",size=25)
plt.ylabel("Class",size=25)
plt.show() 



# =============================================================================
# POTD with EMD
# =============================================================================

eigen_meta, dir_meta = potd(X_train, y_train, np.ones(2*N), 2)

plt.figure(figsize=(6,6))
plt.scatter(X@dir_meta[:,0], X@dir_meta[:,1], c='r', zorder=10, alpha=0.7)
plt.scatter(Y@dir_meta[:,0], Y@dir_meta[:,1], c='b', zorder=10, alpha=0.7)
plt.axis('equal')
plt.xlabel("POTD-1",size=25)
plt.ylabel("POTD-2",size=25)
plt.show() 



# =============================================================================
# POTD with Sinkhorn
# =============================================================================


eigen_meta, dir_meta = potd(X_train, y_train, np.ones(2*N), 2, use_sink=True, regg=0.01)
plt.figure(figsize=(6,6))
plt.scatter(X@dir_meta[:,0], X@dir_meta[:,1], c='r', zorder=10, alpha=0.7)
plt.scatter(Y@dir_meta[:,0], Y@dir_meta[:,1], c='b', zorder=10, alpha=0.7)
plt.axis('equal')
plt.xlabel("POTD-1",size=25)
plt.ylabel("POTD-2",size=25)
plt.show() 






