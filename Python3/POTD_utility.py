import time
import ot
import numpy as np
import pandas as pd
#from scipy import stats
from scipy.linalg import sqrtm
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import skimage.measure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


#==============================================================================
############# Simple image pooling ##############
def imagePooling(data, ratio):
    
    data_pool = np.zeros([len(data), round(784/(ratio**2))])
    for i in range(len(data)):
        dpp = data[i,].reshape(28, 28)
        dppp =  skimage.measure.block_reduce(dpp, (ratio,ratio), np.max)
        data_pool[i,] = dppp.reshape(1, round(784/(ratio**2)))
    return data_pool   
#==============================================================================



def NN_score(X_train_center, X_test, y_train, y_test, direct, col_mean, nn=5):
    data_fit = X_train_center@direct
    data_test = (X_test-col_mean)@direct

    neigh = KNeighborsClassifier(n_neighbors=nn)
    neigh.fit(data_fit, y_train)

    y_pred = neigh.predict(data_test)
    cm = confusion_matrix(y_test, y_pred)
    score = np.sum(np.diag(cm))/len(y_test)
    return score
    






#==============================================================================
############### SAVE direction #################
### x, y: 2-d array
def potd(data_bind, y_label, weight, k, with_sigma = False, time_show = False, thres=0.7, ova = True, use_sink=False, regg=1):    
    #When pp is large, "with_std = True" gives unstable results
    
    pp = data_bind.shape[1]
    slice_cate = pd.Categorical(y_label).categories   
    H = len(slice_cate)
    
    if with_sigma:
        svd = TruncatedSVD(n_components=pp-1, random_state=42)
        svd.fit(data_bind)               
        sv_cum = np.cumsum(svd.singular_values_)/sum(svd.singular_values_)
        nc = np.where(sv_cum>thres)[0][0]+1 
        v_mat = svd.components_[range(nc),:]        
        sigma_mat = np.diag(1/svd.singular_values_[range(nc)])
        signrt = np.transpose(v_mat)@sigma_mat@v_mat
        data_bind = data_bind@signrt
    

    time_start=time.time()
    direct_meta = np.zeros([0, pp])

    for i in range(H):
        if ova:
            for j in range(i+1, H):
        
                X = data_bind[y_label==slice_cate[i],:]
                Y = data_bind[y_label==slice_cate[j],:]
                aa = weight[y_label==slice_cate[i]]
                bb = weight[y_label==slice_cate[i]]
                
                #NN = len(X); MM = len(Y)
                #a, b = np.ones((NN,)) / NN, np.ones((MM,)) / MM
                a, b = aa/sum(aa), bb/sum(bb)
                M = ot.dist(X, Y, metric = "euclidean")
                M = M/M.max()
        
                if use_sink:
                    G = ot.sinkhorn(a, b, M, regg)
                else:
                    G = ot.emd(a, b, M)
                
                #direct = (1/NN)*X-G@Y
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
            
            #NN = len(X); MM = len(Y)
            #a, b = np.ones((NN,)) / NN, np.ones((MM,)) / MM
            a, b = aa/sum(aa), bb/sum(bb)
            M = ot.dist(X, Y, metric = "euclidean")
            M = M/M.max()
    
            G = ot.emd(a, b, M)      
            direct_meta = (X.T*a).T-G@Y       
                
            time_end=time.time()
            if time_show: 
                print('i=', i, '; Time Lapse: ', round(time_end-time_start,2),'s')                 
                                   
            
    direct_meta = np.concatenate([direct_meta, -direct_meta])
    pca = PCA(n_components = k)
    pca.fit(direct_meta)
    res = pca.components_


    if with_sigma:
        res = res@signrt
    
    return pca.singular_values_, np.transpose(res)

#==============================================================================











#==============================================================================
# distance between two spaces
def spDist(A, B):
    pp = A.shape[1]
    Q_a, R_a = np.linalg.qr(A.T)
    Q_b, R_b = np.linalg.qr(B.T)
    
    diag = np.diag(np.repeat(1, pp))
    dismat = (diag - Q_b@Q_b.T)@Q_a
    
    dist = np.matrix.trace(dismat.T@dismat)
    
    return dist

    

#==============================================================================
############### SAVE direction #################
### x, y: 2-d array
def saveDir2(data_bind, y_label, k, with_svd=False, thres=0.7): 

    NN = data_bind.shape[0]
    pp = data_bind.shape[1]

    
    if with_svd:
        svd = TruncatedSVD(n_components=pp-1, random_state=42)
        svd.fit(data_bind)               
        sv_cum = np.cumsum(svd.singular_values_)/sum(svd.singular_values_)
        nc = np.where(sv_cum>thres)[0][0]      
        v_mat = svd.components_[range(nc),:]        
        sigma_mat = np.diag(1/svd.singular_values_[range(nc)])
        signrt = np.transpose(v_mat)@sigma_mat@v_mat
        data_scale = data_bind@signrt
    else:
        data_cov = np.cov(data_bind.T)
        covinv = np.linalg.inv(data_cov)
        signrt = sqrtm(covinv)   
        cm = data_bind.mean(axis = 0)
        data_scale = (data_bind-cm)@signrt

    
    ### slice label
    slice_cate = pd.Categorical(y_label).categories   
    H = len(slice_cate)
    
    ### prob for each slice
    prob = []
    for i in range(H):
        prob.append(len(y_label[y_label==slice_cate[i]])/NN)
    
    
    ### save matrix
    diag = np.diag(np.repeat(1, pp))
    vxy = np.zeros([H,pp,pp])
    save_list = np.zeros([H,pp,pp])
    for i in range(H):
        datai = data_scale[y_label==slice_cate[i],:]
        vxy[i,:,:] = np.cov(datai.T)
        save_list[i,:,:] = prob[i]*((vxy[i,:,:]-diag)@(vxy[i,:,:]-diag))
    savemat = sum(save_list)


    eigenValues, eigenVectors = np.linalg.eig(savemat)
    idx = eigenValues.argsort()[::-1] 

    eigen_meta = []
    dir_meta = np.zeros((pp,k))
    for i in range(k):
        vector = eigenVectors[:, idx[i]]
        dir_temp = signrt@vector
        dir_meta[:,i] = dir_temp/np.sqrt(dir_temp@dir_temp)
        eigen_meta.append(eigenValues[idx[i]])

    return eigen_meta, dir_meta
#==============================================================================
    

