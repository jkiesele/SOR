from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LightSource

import matplotlib as mpl
import matplotlib.cm as cm




def cuboid_data(pos, size=(1,1,3)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(pos, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos=(0,0,0),size=(1,1,3),color=0,ax=None, alpha=1, m=None):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( pos, size )
        ax.plot_surface(X, Z, Y, color=m.to_rgba(color), rstride=1, cstride=1, alpha=alpha,
                        antialiased=True, shade=False)


#ax.set_aspect('equal')


if False:
    import uproot
    from TrainData_PF import TrainData_PF
    td = TrainData_PF()
    td.npart=1
    f,t,w = td.convertFromSourceFile("/afs/cern.ch/user/j/jkiesele/eos_miniCalo/PFTest4/929.root", {}, False)
    
    event=0
    
    calo = f[0][0:100]
    track = f[1][0:100]
    
    print(calo.shape)
    print(track.shape)
    
    
    calo.tofile('calo.dat')
    track.tofile('track.dat')

    exit()


if True:
    
    calo = np.fromfile('calo.dat',dtype='float32')
    calo = np.reshape(calo, (100, 16, 16, 6))
    track = np.fromfile('track.dat',dtype='float32')
    track = np.reshape(track, (100, 64, 64, 6))



    print(calo.shape)
    print(track.shape)

# e, x, y, z
def plotevent(event, arr, ax, iscalo=True):
    usearr = arr[event]
    tot_width = 352
    dxy_indiv = tot_width/float(usearr.shape[0]) #squared
    dz=1.
    if iscalo:
        dz=232
        
    scaled_emax = np.log(np.max(usearr[:,:,0])+1)
    norm = mpl.colors.Normalize(vmin=0, vmax=scaled_emax)
    cmap = cm.hot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    alpha = 0.5
    if not iscalo:
        alpha=0.01
    for i in range(usearr.shape[0]):
        for j in range(usearr.shape[1]):
            x = usearr[i,j,1]
            y = usearr[i,j,2]
            z = usearr[i,j,3]
            e = np.log(usearr[i,j,0]+1)
            alpha = (e+0.15)/(scaled_emax+0.15)
            plotCubeAt(pos=(x,y,dz/2.+z), size=(dxy_indiv,dxy_indiv,dz), color=e, ax=ax, m=m, alpha=alpha)
    
    

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel("x [mm]")
ax.set_zlabel("y [mm]")
ax.set_ylabel("z [mm]")
ax.grid(False)

ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

plotevent(0,calo,ax,True)
plotevent(0,track,ax,False)
plt.tight_layout()
plt.savefig("geometry.pdf")


