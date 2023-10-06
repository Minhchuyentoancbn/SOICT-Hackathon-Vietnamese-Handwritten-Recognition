import cv2 
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve, lsqr


def wlsFilter(IN, Lambda=1.0, Alpha=1.2):
    """
    IN        : Input image (2D grayscale image, type float)
    Lambda    : Balances between the data term and the smoothness term.
                Increasing lbda will produce smoother images.
                Default value is 1.0
    Alpha     : Gives a degree of control over the affinities by 
                non-lineary scaling the gradients. Increasing alpha 
                will result in sharper preserved edges. Default value: 1.2
    """
    
    L = np.log(IN+1e-22)        # Source image for the affinity matrix. log_e(IN)
    smallNum = 1e-6
    height, width = IN.shape
    k = height * width

    # Compute affinities between adjacent pixels based on gradients of L
    dy = np.diff(L, n=1, axis=0)   # axis=0 is vertical direction

    dy = -Lambda/(np.abs(dy)**Alpha + smallNum)
    dy = np.pad(dy, ((0,1),(0,0)), 'constant')    # add zeros row
    dy = dy.flatten(order='F')

    dx = np.diff(L, n=1, axis=1)

    dx = -Lambda/(np.abs(dx)**Alpha + smallNum)
    dx = np.pad(dx, ((0,0),(0,1)), 'constant')    # add zeros col 
    dx = dx.flatten(order='F')
    # Construct a five-point spatially inhomogeneous Laplacian matrix
    
    B = np.concatenate([[dx], [dy]], axis=0)
    d = np.array([-height,  -1])

    A = spdiags(B, d, k, k) 

    e = dx 
    w = np.pad(dx, (height, 0), 'constant'); w = w[0:-height]
    s = dy
    n = np.pad(dy, (1, 0), 'constant'); n = n[0:-1]

    D = 1.0 - (e + w + s + n)

    A = A + A.transpose() + spdiags(D, 0, k, k)

    # Solve
    OUT = spsolve(A, IN.flatten(order='F'))
    return np.reshape(OUT, (height, width), order='F') 


def gdft(img, r):
    eps = 0.04;
    
    I = np.double(img);
    # I = I/255;
    I2 = cv2.pow(I,2);
    mean_I = cv2.boxFilter(I,-1,((2*r)+1,(2*r)+1))
    mean_I2 = cv2.boxFilter(I2,-1,((2*r)+1,(2*r)+1))
    
    cov_I = mean_I2 - cv2.pow(mean_I,2);
    
    var_I = cov_I;
    
    a = cv2.divide(cov_I,var_I+eps)
    b = mean_I - (a*mean_I)
    
    mean_a = cv2.boxFilter(a,-1,((2*r)+1,(2*r)+1))
    mean_b = cv2.boxFilter(b,-1,((2*r)+1,(2*r)+1))
    
    q = (mean_a * I) + mean_b;
    
    return q


def SRS(r, i):
    mI = np.mean(i)
    r_eh = np.zeros_like(r)
    mask1 = r > mI
    mask2 = r <= mI

    r_changed = r * (i/mI)**0.5
    r_eh[mask1] = r_changed[mask1]
    r_eh[mask2] = r[mask2]

    return r_eh

def VIG(i, i_inv, v_s):
    i_inv /=np.max(i_inv)
    mI = np.mean(i)
    M = np.max(i)
    r = 1.0 - mI/M
    fv_s = [r*( 1/(1+np.exp(-1.0*(v - mI))) - 0.5 ) for v in v_s]

    I_k = [(1 + fv) * (i + fv * i_inv) for fv in fv_s]

    return I_k

def tone_production(R_eh, I_vts):
    L_s = [np.exp(R_eh) * I for I in I_vts]
    
    Ws = np.zeros_like(I_vts)
    for i, I in enumerate(I_vts):
        if i < 3:
            Ws[i,:,:] = I/np.max(I)
        else:
            Ws[i,:,:] = 1.0 - I/np.max(I)

    L_eh = np.zeros_like(R_eh)
    W_ = np.zeros_like(R_eh)
    for W, L in zip(Ws, L_s):
        L_eh += W * L
        W_ += W
    
    L_eh = L_eh/W_
    return L_eh


def HDR(image, flag=True):
    S = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0
    S = S + 1e-20
    image = 1.0*image/255

    if flag:
        I = gdft(S, 3)
    else:
        I = wlsFilter(S)
    mI = np.mean(I)
    R = np.log(S+1e-20) - np.log(I+1e-20)
    R_eh = SRS(R, I)

    v_s = [0.2, (mI+0.2)/2, mI, (mI+0.8)/2, 0.8]

    I_vts = VIG(I, 1.0-I, v_s)
    L_eh = tone_production(R_eh, I_vts)
    
    ratio = np.clip(L_eh/ S, 0, 3)
    b,g,r = cv2.split(image)

    b_eh = ratio * b
    g_eh = ratio * g
    r_eh = ratio * r

    out = cv2.merge((b_eh, g_eh, r_eh))
    out = np.clip(out, 0.0, 1.0)
    out = np.uint8(out * 255)
    return out