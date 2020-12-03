import time
import ot
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import skimage.measure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import KernelPCA
from sliced import SlicedInverseRegression
from sliced.save import SlicedAverageVarianceEstimation
from ot.dr import wda, fda
from wda_functions import wda_eig



#==============================================================================
############# Simple Image Pooling ##############
def imagePooling(data, ratio, width, height):
    
    data_pool = np.zeros([len(data), round(width*height/(ratio**2))])
    for i in range(len(data)):
        dpp = data[i,].reshape(width, height)
        dppp =  skimage.measure.block_reduce(dpp, (ratio, ratio), np.max)
        data_pool[i,] = dppp.reshape(1, round(width*height/(ratio**2)))
    return data_pool   
#==============================================================================


#===============================================================================
############# A Subset of MNIST Data ##############
def sub_data(data, test, pooling_ratio = 1, width = 28, height = 28, N = 100):
    #N: sample size for each class
    NN = N*10 #full sample size
    pp = width*height #feature size

    data_mat = np.zeros([NN, pp])
    for i in range(10):
        temp_ind = np.where(data[:,0]==i)[0][range(N)]
        temp_data = data[temp_ind,:][:,range(1,(pp+1))]
        data_mat[range(i*N,i*N+N),] = temp_data
    
    #Max pooling
    data_train = imagePooling(data_mat, pooling_ratio, width, height)/255
    data_test = imagePooling(test[:,range(1,(pp+1))], pooling_ratio, width, height)/255
    y_train = np.repeat(list(range(10)), N, axis=0)
    y_test = test[:,0]
    return data_train, data_test, y_train, y_test
#==============================================================================



#==============================================================================
############# Classification Accuracy by KNN ##############
def KNN_score(data_fit, data_test, y_train, y_test, nn = 10):

    neigh = KNeighborsClassifier(n_neighbors = nn)
    neigh.fit(data_fit, y_train)

    y_pred = neigh.predict(data_test)
    cm = confusion_matrix(y_test, y_pred)
    score = np.sum(np.diag(cm))/len(y_test)
    return score
#==============================================================================
  


#==============================================================================
############### POTD #################
def potd(data_bind, y_label, weight, k = 20, with_std = False, with_sigma = False, time_show = False, thres = 0.7, ovo = True, use_sink = False, regg = 0.01):    
    #When pp is large, "with_sigma = True" gives unstable results
    
    pp = data_bind.shape[1]
    slice_cate = pd.Categorical(y_label).categories   
    H = len(slice_cate)
    
    if with_std:
        std = np.std(data_bind, axis=0)
        for i in range(std.shape[0]):
            if std[i] != 0:
                std[i] = 1/std[i]
        data_bind = data_bind * std

    #weighted covariance matrix
    if with_sigma:
        svd = TruncatedSVD(n_components=pp-1, random_state=42)
        w_sum = 1-np.sum(weight**2)
        svd.fit((data_bind.T*np.sqrt(weight/w_sum)).T)               
        sv_cum = np.cumsum(svd.singular_values_)/sum(svd.singular_values_)
        nc = np.where(sv_cum>thres)[0][0]+1 
        v_mat = svd.components_[range(nc),:]        
        sigma_mat = np.diag(1/svd.singular_values_[range(nc)])
        signrt = np.transpose(v_mat)@sigma_mat@v_mat
        data_bind = data_bind@signrt

    time_start=time.time()
    direct_meta = np.zeros([0, pp])

    for i in range(H):
        if ovo:
            for j in range(i+1, H):
        
                X = data_bind[y_label==slice_cate[i],:]
                Y = data_bind[y_label==slice_cate[j],:]
                aa = weight[y_label==slice_cate[i]]
                bb = weight[y_label==slice_cate[j]]
                
                a, b = aa/sum(aa), bb/sum(bb)
                M = ot.dist(X, Y, metric = "euclidean")
                M = M/M.max()
        
                if use_sink:
                    G = ot.sinkhorn(a, b, M, regg)
                else:
                    G = ot.emd(a, b, M)
                
                direct = (X.T*a).T-G@Y
                direct_meta = np.concatenate([direct_meta, direct])   
                
                time_end=time.time()
                if time_show: 
                    print('i=', i, '; j=', j, '; Time Lapse: ', round(time_end-time_start,2),'s') 
        else:
        
            X = data_bind[y_label==slice_cate[i],:]
            Y = data_bind[y_label!=slice_cate[i],:]
            aa = weight[y_label==slice_cate[i]]
            bb = weight[y_label!=slice_cate[i]]
            
            a, b = aa/sum(aa), bb/sum(bb)
            M = ot.dist(X, Y, metric = "euclidean")
            M = M/M.max()
    
            if use_sink:
                G = ot.sinkhorn(a, b, M, regg)
            else:
                G = ot.emd(a, b, M)            
            
            direct_meta = (X.T*a).T-G@Y       
                
            time_end=time.time()
            if time_show: 
                print('i=', i, '; Time Lapse: ', round(time_end-time_start,2),'s')                 
                                         
    if ovo:      
        direct_meta = np.concatenate([direct_meta, -direct_meta])    
    
    pca = PCA(n_components = k)
    pca.fit(direct_meta)
    res = pca.components_

    if with_sigma:
        res = res@signrt
    
    return pca.singular_values_, np.transpose(res)



#==============================================================================
############# POTD_predict ##############
def potd_pre(data_train_center, data_test_center, y_label, weight, direction = 60, with_std = False, with_sigma = False, time_show = False, thres = 0.7, ovo = True, use_sink = False, regg = 0.01):

    potd_var, potd_dir = potd(data_bind = data_train_center, y_label = y_label, weight = weight,
                                   k = direction, with_std = with_std, with_sigma = with_sigma, 
                                   time_show = time_show, thres = thres, ovo = ovo, use_sink = use_sink, regg = regg)
    data_potd_fit = data_train_center@potd_dir
    data_potd_test = data_test_center@potd_dir
    
    return data_potd_fit, data_potd_test
#==============================================================================



#==============================================================================
############# SIR_predict ##############
def sir_pre(data_train_center, data_test_center, y_label):

    sir = SlicedInverseRegression(n_directions=9, n_slices=10)
    nonzero_id = np.where(np.std(data_train_center,0)!=0)[0]
    data_train_center_shrink = data_train_center[:,nonzero_id]
    sir.fit(data_train_center_shrink, y_label)
    data_sir_fit = data_train_center_shrink@np.transpose(sir.directions_)
    data_sir_test = data_test_center[:,nonzero_id]@np.transpose(sir.directions_)
  
    return data_sir_fit, data_sir_test
#==============================================================================



#==============================================================================
############# SAVEII_predict ##############
def save_pre(data_train_center, data_test_center, y_label, direction):

    save = SlicedAverageVarianceEstimation(n_directions=direction, n_slices=10)
    nonzero_id = np.where(np.std(data_train_center,0)!=0)[0]
    data_train_center_shrink = data_train_center[:,nonzero_id]
    save.fit(data_train_center_shrink, y_label)
    data_save_fit = data_train_center_shrink@np.transpose(save.directions_)
    data_save_test = data_test_center[:,nonzero_id]@np.transpose(save.directions_)
        
    return data_save_fit, data_save_test
#==============================================================================



#==============================================================================
############# WDA_predict ##############
def wda_pre(data_train_center, data_test_center, y_label, direction, reg = 100):

    P0 = np.random.randn(data_train_center.shape[1], direction)
    P0 /= np.sqrt(np.sum(P0**2, 0, keepdims=True))

    Pwda, projwda = wda(data_train_center, y_label, direction, reg, k=10, maxiter=30, P0=P0)
    data_wda_fit = projwda(data_train_center)
    data_wda_test = projwda(data_test_center)
  
    return data_wda_fit, data_wda_test
#==============================================================================




#==============================================================================
############# WDA-EIG_predict ##############
def wdae_pre(data_train_center, data_test_center, y_label, direction, reg = 100):

    P0 = np.random.randn(data_train_center.shape[1], direction)
    P0 /= np.sqrt(np.sum(P0**2, 0, keepdims=True))

    Pwdae, projwdae, obje = wda_eig(data_train_center, y_label, direction, reg, P0, Breg=2.0, k=10, maxiter=30)
    
    data_wdae_fit = projwdae(data_train_center)
    data_wdae_test = projwdae(data_test_center)
  
    return data_wdae_fit, data_wdae_test
#==============================================================================






