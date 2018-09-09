import numpy as np

# solve the stationary distributin given transition matrix
def solve_stationary(tm):
    n = len(tm)
    A = np.append( (tm.T-np.identity(n))[:-1,:], np.ones([1,n]), axis=0 )
    return np.linalg.solve(A, [0 for i in range(n-1)] + [1])


    
# forward algorithm
# A: [nx, nx] transition matrix a(i,j) = P(x(t+1)=j|x(t)=i)
# B: [nx, ny] emission matrix b(i,j) = P(y(t)=j|x(t)=i)
# y: [nt] observation sequence of index [0,1,...,ny-1]
# pxinit: [nx] initial probability distribution of hiddens states
def forward(A, B, y, pxinit=None):
    nx = A.shape[0]
    ny = B.shape[1]
    nt = len(y)
    if pxinit is None: pxinit = np.array([1.0/nx for i in range(nx)])
    
    alpha = np.zeros([nt, nx])
    
    alpha[0,:] = pxinit * B[:,y[0]]
    
    for t in range(1,nt):
        alpha[t,:] = np.dot(alpha[t-1,:], A) * B[:,y[t]]
    
    return np.sum(alpha[-1,:])


# Viterbi algorithm
# A: [nx, nx] transition matrix a(i,j) = P(x(t+1)=j|x(t)=i)
# B: [nx, ny] emission matrix b(i,j) = P(y(t)=j|x(t)=i)
# y: [nt] observation sequence of index [0,1,...,ny-1]
# pxinit: [nx] initial probability distribution of hiddens states
def viterbi(A, B, y, pxinit=None):
    nx = A.shape[0]
    ny = B.shape[1]
    nt = len(y)
    if pxinit is None: pxinit = np.array([1.0/nx for i in range(nx)])
    
    alpha = np.zeros([nt, nx])
    ind   = np.zeros([nt, nx]).astype(int)
    
    alpha[0,:] = pxinit * B[:,y[0]]
    ind[0,:] = -1
    x = [-1 for i in range(nt)]
    
    for t in range(1,nt):
        for i in range(nx):
            alpha[t,i] = np.max(alpha[t-1,:] * A[:,i]) * B[i,y[t]]
            ind[t,i] = np.argmax(alpha[t-1,:] * A[:,i])
    
    x[nt-1] = np.argmax(alpha[nt-1, :])
    for t in range(nt-1)[::-1]:
        x[t] = ind[t+1, x[t+1]]
    
    return x

# simulation with HMM
def HMM_simulation(A, B, nt=100, pxinit=None, xstr=None, ystr=None):
    nx = A.shape[0]
    ny = B.shape[1]
    if pxinit is None: pxinit = np.array([1.0/nx for i in range(nx)])

    x = [np.random.choice(nx, p = pxinit)]
    y = [np.random.choice(ny, p = B[x[0],:] )]
    for t in range(nt-1):
        x.append(np.random.choice(nx, p = A[x[-1], :]))
        y.append(np.random.choice(ny,p = B[x[-1], :]))

    return x, y

# Baum-Welch algorithm
# A: [nx, nx] transition matrix a(i,j) = P(x(t+1)=j|x(t)=i)
# B: [nx, ny] emission matrix b(i,j) = P(y(t)=j|x(t)=i)
# y: [nt] observation sequence of index [0,1,...,ny-1]
# pxinit: [nx] initial probability distribution of hiddens states
def baum_welch(y, nx=2, pxinit=None, maxiter=100):
    ny = len(set(y))
    nt = len(y)
    if pxinit is None: pxinit = np.array([1.0/nx for i in range(nx)])

    A = np.random.uniform(size=nx*nx).reshape([nx,nx])
    A /= np.sum(A, axis=1).reshape([nx,1])
    B = np.random.uniform(size=nx*ny).reshape([nx,ny])
    B /= np.sum(B, axis=1).reshape([nx,1])
    
    error = 999
    k = 0
    alpha = np.zeros([nt, nx])
    beta  = np.zeros([nt, nx])
    errors = []
    
    while (k<maxiter and error>1e-6):
        # forward
        alpha[0,:] = pxinit * B[:,y[0]]
        for t in range(1,nt):
            alpha[t,:] = np.dot(alpha[t-1,:], A) * B[:,y[t]]
        
        # backward
        beta[nt-1,:] = 1.0
        for t in range(nt-1)[::-1]:
            beta[t,:] = np.dot(A, beta[t+1,:] * B[:,y[t+1]])

        #print(alpha.T)
        #print(beta.T)
            
        # update
        px = alpha * beta / np.sum(alpha*beta, axis=1).reshape([nt, 1])
        p2d = np.zeros([nt, nx, nx])
        for t in range(nt-1):
            p2d[t,:,:] = alpha[t,:].reshape([nx,1]) * A * \
                beta[t+1,:].reshape([1,nx]) * B[:, y[t+1]].reshape([1,nx])
            p2d[t,:,:] /= np.sum(p2d[t,:,:])
        
        pxinit = px[0,:]
        Anew = np.sum(p2d[:nt-1,:,:], axis=0) / np.sum(px[:nt-1,:], axis=0).reshape([nx,1])
        Bnew = np.zeros([nx, ny])
        for t in range(nt-1):
            Bnew[:, y[t]] += px[t,:]
        Bnew /= np.sum(px, axis=0).reshape([nx, 1])
        
        error = np.mean((Anew-A)**2) + np.mean((Bnew-B)**2) 
        errors.append(error)
        A = Anew + 0
        B = Bnew + 0
        k += 1
        
        #print(k, error)
        
    return A, B, errors

