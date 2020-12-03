from scipy import linalg
import numpy as np

def pair_tensor(T,x1,x2):
    temp=x1[:,None] - x2 
    C=temp*T[:,:,None]
    ans = np.tensordot(C, temp, ([0,1],[0,1]))
    return ans

def dist(x1, x2):
    """ Compute squared euclidean distance between samples (autograd)
    """
    x1p2 = np.sum(np.square(x1), 1)
    x2p2 = np.sum(np.square(x2), 1)
    return x1p2.reshape((-1, 1)) + x2p2.reshape((1, -1)) - 2 * np.dot(x1, x2.T)


def sinkhorn(w1, w2, M, reg, k):
    """Sinkhorn algorithm with fixed number of iteration (autograd)
    """
    
    ## note: lambda=1/reg in paper
    K = np.exp(-M/reg )
    ui = np.ones((M.shape[0],))
    vi = np.ones((M.shape[1],))
    for i in range(k):
        vi = w2 / (np.dot(K.T, ui))
        ui = w1 / (np.dot(K, vi))
    G = ui.reshape((M.shape[0], 1)) * K * vi.reshape((1, M.shape[1]))
    return G


def split_classes(X, y):
    """split samples in X by classes in y
    """
    lstsclass = np.unique(y)
    return [X[y == i, :].astype(np.float32) for i in lstsclass]

def wda_eig(X, y, p, reg, P0, Breg=0, k=10, maxiter=100, verbose=0):
    """
    The function solves the ratio trace formulation of WDA.
    Parameters
    ----------
    X : ndarray, shape (n, d)
        Training samples.
    y : ndarray, shape (n,)
        Labels for training samples.
    p : int
        Size of dimensionality reduction.
    reg : float
        Wasserstein regularization term >0 (entropic regularization)
    P0 : ndarray, shape (d, p)
        Initial subspace for projection.
    Breg: float, optional, default set to 0
        Regularization for the B matrix in the denominator to make B positive definite
    k: int, optional, default set to 10
        Number of Sinkhorn iterations
    maxiter: int, optional, default set to 100
        Number of maximum number of iterations
    Returns
    -------
    P : ndarray, shape (d, p)
        Optimal transportation matrix for the given parameters
    proj : callable
        Projection function including mean centering
    obj: list
        List of angles s_k to measure the distance between consecutive subspaces
    
    """  
    mx = np.mean(X)
    X -= mx.reshape((1, -1))

    # data split between classes
    d = X.shape[1]
    xc = split_classes(X, y)
    # compute uniform weighs
    wc = [np.ones((x.shape[0]), dtype=np.float32) / x.shape[0] for x in xc]
    P=P0
    
    obj = []
    
    for it in range(maxiter):      
        loss_b = np.zeros((d,d))
        loss_w = np.zeros((d,d))
        for i, xi in enumerate(xc):
            pxi = np.dot(xi, P)
            for j, xj in enumerate(xc[i:]):
                pxj = np.dot(xj, P)
                M = dist(pxi, pxj)
                G = sinkhorn(wc[i], wc[j + i], M, reg, k)
                if j==0:
                    loss_w += pair_tensor(G, xi, xj)
                else:
                    loss_b += pair_tensor(G, xi, xj)
        if Breg==0:
            w, V = linalg.eig((loss_b+loss_b.T)/2, (loss_w+loss_w.T)/2)
        else:
            w, V = linalg.eig((loss_b+loss_b.T)/2, (loss_w+loss_w.T)/2+Breg*np.eye(d))
        w=np.real(w)
        V=np.real(V)
        #idx = np.argsort(w.real)
        #Pnew = np.real(V[:, 0:p])
        idx = np.argsort(-w)
        Pnew = V[:, idx[0:p]]
        

        Pinv = np.linalg.inv(P.T.dot(P))
        Pninv = np.linalg.inv(Pnew.T.dot(Pnew))
        
        ## s_k = ||sin\theta(P_{k-1}, P_k)||_2
        
        angle = np.linalg.norm((P.dot(Pinv.dot(P.T))-Pnew.dot(Pninv.dot(Pnew.T))),2)
        
        obj.append(angle)
        if (verbose==1):
            print("Iter: % 2d, angle: % 2.8f" %(it, angle))
            
        P=Pnew

        if (abs(angle)< 1e-3):
            break

    Popt = P

    def proj(X):
        return (X - mx.reshape((1, -1))).dot(Popt)

    return Popt, proj, obj