#inhomogeneous 2D Acoustic wave solution using PINN 
import pickle
#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit
from SALib.sample import sobol_sequence
import scipy.interpolate as interpolate
#from scipy import io
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.cluster import KMeans
#from scipy.stats import pearsonr
#import scipy.interpolate as sci_interp



#x = tf.placeholder(tf.float64, shape=(None,1))
#z = tf.placeholder(tf.float64, shape=(None,1))


ax=16#It doesn't matter if these are in km and in specfem we have dimesnions in meters. Because the mesh system is based on mesh number and not loation in terms of km or meters. 
az=10

nx=160
nz=120
N=nx*nz

dx=ax/nx
dz=az/nz

#a=2;b=1

def g(x,z,a,b,c,d):
  return ((x-c)**2/a**2+(z-d)**2/b**2)


def f(x,z):
  return 3-0.5*(1+np.tanh(100*(1-g(x,z,3,2,9,6))))
         


#3-0.25*(1+np.tanh(100*(y-3-0.5*(1+np.tanh(x-6)))))*(1+np.tanh(100*(x-4)))
#3-0.5*0.25*(1+np.tanh(100*(y-3.5)))*(1+np.tanh(100*(4-y)))*0.25*(1+np.tanh(100*(x-3)))*(1+np.tanh(100*(7-x)))\
#           -0.25*(1+np.tanh(100*(y-2.5)))*(1+np.tanh(100*(3.5-y)))*0.25*(1+np.tanh(100*(x-3)))*(1+np.tanh(100*(7-x)))
#

#3-np.exp(-(y-4)**2/b**2)
#3-np.exp(-(((x-4)**2)/a**2+((y-4)**2)/(b**2))) Ellipsoid
x=dx/2+np.linspace(0,ax,nx+1)#the location of the middle of each mesh is used for generating the model (P wave velocity)
y=dz/2+np.linspace(0,az,nz+1)

xx, yy = np.meshgrid(x[:-1],y[:-1])
mesh = np.concatenate((xx.reshape((-1,1)),yy.reshape((-1,1))),axis=1)

Vp=1000*f(mesh[:,0],mesh[:,1]).reshape((-1,1))





fig = plt.figure()
plt.contourf(xx, yy, Vp.reshape(xx.shape),100, cmap='jet')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('scaled')
plt.colorbar()
plt.savefig('velocity_model.png',dpi=400)
plt.show()
plt.close(fig)
            
rho=1000

k=0

file_model = open("model.txt","w") 
for k in range(N):
      L = [  str(k+1)+ " " "1" " "+str(rho)+" "+str(Vp[k][0]) +    " 1.0 0 0 9999 9999 0 0 0 0 0 0" "\n"]  
      # \n is placed to indicate EOL (End of Line) 
      file_model.writelines(L) 
file_model.close() #to change file access modes 




m=0
file_mesh = open("mesh.txt","w") 
for j in range(nz):
    for i in range(nx):
        m=m+1
        mode_m = [str(i+1)+" "+str(i+1)+" "+str(j+1)+" "+str(j+1)+" "+str(m)+ "\n"]  
      # \n is placed to indicate EOL (End of Line) 
        file_mesh.writelines(mode_m) 
file_mesh.close() #to change file access modes 
















