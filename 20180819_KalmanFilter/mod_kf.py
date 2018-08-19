import numpy as np

# generate multivariant Guassian noise, given mu and cov
def get_noise(mu = np.array([0, 0]), cov = np.array([[1, 0], [0, 1]]), size=100):
    L = np.linalg.cholesky(cov)
    tmp = np.random.normal(size=size*len(mu)).reshape([len(mu), size]) + mu.reshape(len(mu), 1)
    return np.dot(L, tmp).T  # [size, ndim]


# ======= 1D dynamic system =======
# x: states [position, velocity]
# dtime: time step (s)
# ac: external acceleration (m/s2)
# w: noises on states
# phy: based on physics only
# gt: "ground truth" = phy + noises

# input control function
def control_1d_constant(x = None, time = None, ac = -5):
    return np.array([0, ac])

def control_1d_sintime(x = None, time = None, period=5, amp=4.0):
    return np.array([0, amp*np.sin(time/period*2.0*np.pi)])

def dynsys_1d(x=np.array([0, 0]), time=0.0, dtime=1.0, control=control_1d_constant, w=np.array([0,0])):
    ndim = len(x)
    F = np.array([[1, dtime], [0, 1]])
    G = np.array([[0, 0.5*dtime*dtime], [0, dtime]])
    u = control(x = x, time = time)
    phy = np.dot(F, x.reshape(ndim, 1)) + np.dot(G, u.reshape(ndim, 1))
    gt  = phy + w.reshape(ndim, 1)
    return phy[:,0], gt[:,0]

def kf_1d(x=np.array([0, 0]), p=np.zeros([2,2]), z=np.array([0, 0]), 
        syscov=np.array([[4,2], [2,4]]), obscov=np.array([[4,2], [2,4]]), 
        time=0.0, dtime=1.0, control=control_1d_constant):
    ndim = len(x)
    F = np.array([[1, dtime], [0, 1]])
    G = np.array([[0, 0.5*dtime*dtime], [0, dtime]])
    H = np.array([[1,0], [0,1]])
    u = control(x = x, time = time)
    xpre = np.dot(F, x.reshape(ndim, 1)) + np.dot(G, u.reshape(ndim, 1))
    ppre = np.dot(F, np.dot(p, F.T)) + syscov
    K = np.dot( np.dot(ppre, H.T), np.linalg.inv(np.dot(H, np.dot(ppre, H.T)) + obscov) )
    xnew = xpre + np.dot(K, z.reshape(ndim,1)-np.dot(H, xpre))
    pnew = np.dot( np.dot((np.identity(2)-np.dot(K, H)), ppre), (np.identity(2)-np.dot(K, H)).T ) + \
        np.dot(K, np.dot(obscov, K.T))
    return K, xnew[:,0], pnew

def obs_1d(x=np.array([0,0]), v=np.array([0, 0])):
    H = np.array([[1,0], [0,1]])
    return np.dot(H, x.reshape(len(x),1))[:,0] + v

def simulation_1d(xinit = np.array([0, 0]), fcontrol = control_1d_sintime, endtime=20, dtime=0.25, 
        syscov = np.array([[4,2], [2,4]]), obscov = np.array([[5, 2], [2, 9]])):
    ntime = int(endtime/dtime)
    ws = get_noise(cov = syscov, size=ntime+1)
    vs = get_noise(cov = obscov, size=ntime+1)
    time = np.linspace(0, endtime, ntime+1)

    ideal = np.zeros([ntime+1, 2]); ideal[0, :] = xinit
    gt    = np.zeros([ntime+1, 2]); gt[0, :] = xinit + ws[0,:]
    obs   = np.zeros([ntime+1, 2]); obs[0, :] = obs_1d(xinit, vs[0,:])
    kf    = np.zeros([ntime+1, 2]); kf[0, :]  = obs[0, :]
    p = syscov + 0
    for i in range(ntime):
        ideal[i+1, :], y = dynsys_1d(x=ideal[i,:], dtime=dtime, control=fcontrol, time=i*dtime, w=ws[i+1,:])
        x, gt[i+1, :] = dynsys_1d(x=gt[i,:], dtime=dtime, control=fcontrol, time=i*dtime, w=ws[i+1,:]) 
        obs[i+1, :] = obs_1d(x=gt[i+1,:], v=vs[i+1,:])
        K, kf[i+1, :], p = kf_1d(x=kf[i,:], p=p, z=obs[i+1,:], 
          syscov=syscov, obscov=obscov, 
          time=i*dtime, dtime=dtime, control=fcontrol)
    return time, ideal, gt, obs, kf


# ======= 2D dynamic system =======
# x: states [x, y, vx, vy]
# dtime: time step (s)
# ac: external acceleration (m/s2)
# w: noises on states
# phy: based on physics only
# gt: "ground truth" = phy + noises

# input control function
def control_2d_coriolis(x = None, time = None, amp = 1.0):
    return np.array([0, 0, x[3]*amp, -x[2]*amp])

def control_2d_center(x = None, time = None, amp = 1.0):
    dis = np.sqrt(x[0]*x[0] + x[1]*x[1])
    return np.array([0, 0, -amp*x[0], -amp*x[1] ])

def control_2d_multiplecenters(x = None, time = None, amps = [1.0, 1.0], centers=[[-10, 0], [10, 0]]):
    ac = np.array([0,0,0,0])
    for i in range(len(centers)):
        dis = np.sqrt((x[0]-centers[i][0])*(x[0]-centers[i][0]) + (x[1]-centers[i][1])*(x[1]-centers[i][1]))
        ac[2] += -amps[i]*(x[0]-centers[i][0])*dis
        ac[3] += -amps[i]*(x[1]-centers[i][1])*dis    
    return ac

def dynsys_2d(x=np.array([0, 0, 0, 0]), time=0.0, dtime=1.0, control=control_2d_coriolis, w=np.array([0,0,0,0])):
    ndim = len(x)
    F = np.array([[1, 0, dtime, 0], 
                  [0, 1, 0, dtime], 
                  [0, 0, 1, 0], 
                  [0, 0, 0, 1] ])
    G = np.array([[0, 0, 0.5*dtime*dtime, 0], 
                  [0, 0, 0, 0.5*dtime*dtime], 
                  [0, 0, dtime, 0], 
                  [0, 0, 0, dtime] ])
    u = control(x = x, time = time)
    phy = np.dot(F, x.reshape(ndim, 1)) + np.dot(G, u.reshape(ndim, 1))
    gt  = phy + w.reshape(ndim, 1)
    return phy[:,0], gt[:,0]

def kf_2d(x, p, z, syscov, obscov, 
        time=0.0, dtime=1.0, control=control_2d_coriolis):
    ndim = len(x)
    F = np.array([[1, 0, dtime, 0], 
                  [0, 1, 0, dtime], 
                  [0, 0, 1, 0], 
                  [0, 0, 0, 1] ])
    G = np.array([[0, 0, 0.5*dtime*dtime, 0], 
                  [0, 0, 0, 0.5*dtime*dtime], 
                  [0, 0, dtime, 0], 
                  [0, 0, 0, dtime] ])
    H = np.identity(4)
    u = control(x = x, time = time)
    xpre = np.dot(F, x.reshape(ndim, 1)) + np.dot(G, u.reshape(ndim, 1))
    ppre = np.dot(F, np.dot(p, F.T)) + syscov
    K = np.dot( np.dot(ppre, H.T), np.linalg.inv(np.dot(H, np.dot(ppre, H.T)) + obscov) )
    xnew = xpre + np.dot(K, z.reshape(ndim,1)-np.dot(H, xpre))
    pnew = np.dot( np.dot((np.identity(4)-np.dot(K, H)), ppre), (np.identity(4)-np.dot(K, H)).T ) + \
        np.dot(K, np.dot(obscov, K.T))
    return K, xnew[:,0], pnew

def obs_2d(x, v):
    H = np.identity(4)
    return np.dot(H, x.reshape(len(x),1))[:,0] + v

def simulation_2d(xinit = np.array([0, 0, 0, 0]), fcontrol = control_2d_coriolis, endtime=20, dtime=0.25, 
                syscov=np.identity(4), obscov=np.identity(4)):
    ntime = int(endtime/dtime)
    ws = get_noise(mu = np.array([0,0,0,0]), cov = syscov, size=ntime+1)
    vs = get_noise(mu = np.array([0,0,0,0]), cov = obscov, size=ntime+1)
    time = np.linspace(0, endtime, ntime+1)

    ideal = np.zeros([ntime+1, 4]); ideal[0, :] = xinit
    gt    = np.zeros([ntime+1, 4]); gt[0, :] = xinit + ws[0,:]
    obs   = np.zeros([ntime+1, 4]); obs[0, :] = obs_2d(xinit, vs[0,:])
    kf    = np.zeros([ntime+1, 4]); kf[0, :]  = obs[0, :]
    p = syscov + 0
    for i in range(ntime):
        ideal[i+1, :], y = dynsys_2d(x=ideal[i,:], dtime=dtime, control=fcontrol, time=i*dtime, w=ws[i+1,:])
        x, gt[i+1, :] = dynsys_2d(x=gt[i,:], dtime=dtime, control=fcontrol, time=i*dtime, w=ws[i+1,:]) 
        obs[i+1, :] = obs_2d(x=gt[i+1,:], v=vs[i+1,:])
        K, kf[i+1, :], p = kf_2d(x=kf[i,:], p=p, z=obs[i+1,:], 
          syscov=syscov, obscov=obscov, 
          time=i*dtime, dtime=dtime, control=fcontrol)
    return time, ideal, gt, obs, kf