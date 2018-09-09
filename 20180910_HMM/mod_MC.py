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


