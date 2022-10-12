#this is for plotting the 2D normalized acoustic runs
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit
from SALib.sample import sobol_sequence
import scipy.interpolate as interpolate
import random
#from scipy import io
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.cluster import KMeans
#from scipy.stats import pearsonr
#import scipy.interpolate as sci_interp

tf.reset_default_graph()    
picklepickle = open("recorded_weights.pickle","rb")
pickle_in=pickle.load(picklepickle)
weights_opt = pickle_in[1]
biases_opt = pickle_in[2]



x = tf.placeholder(tf.float64, shape=(None,1))
z = tf.placeholder(tf.float64, shape=(None,1))
t = tf.placeholder(tf.float64, shape=(None,1))

rho=1.0
ax=8.0#dimension of the domain in the x direction
az=6.0#dimension of the domain in the z direction
t_m=5.0#total time

Lx=3;#this is for normalizing the wavespeed in the PDE via saling x coordinate
Lz=3;#this is for normalizing the wavespeed in the PDE via scaling z coordinate
####  Acoustic wave's speed
alpha=2#6-3*tf.exp(-((x*Lx-6)**2+(z*Lz-6)**2)/(2**2))#4+tf.tanh(100*(x-4))


ub=np.array([ax/Lx,az/Lz,t_m]).reshape(-1,1).T#Maziyar's normalization of the input to the NN, but since we are scaling the spatial coordinates so these need to be scaled accordingly 



def xavier_initw(l):
    return tf.Variable(weights_opt[l], dtype=tf.float64)

def xavier_initb(l):
    return tf.Variable(biases_opt[l], dtype=tf.float64)

def neural_net(X, weights, biases):
    num_layers = len(weights) + 1    
    H=2*(X/ub)-1#normalization map to [-1 1]
    for l in range(0,num_layers-2): #the last layer will be treated outside this loop
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
#        H = tf.nn.dropout(H, keep_prob = 0.5 )
        #H = tf.sin(tf.add(tf.matmul(H, W), b))

    W = weights[-1]     #[-1] takes the last object of the weights tensor. The shape of this last object is shape=(20,2) 
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b) #seems the last layer of nn is simply a linear unit and we don't apply the tanh 
    return Y


layers=[3]+[200]*3+[1] # layers for the NN approximating the pressure field; scalar pressure output 
                        #and three inputs x,z,t
print('NN layers',':',layers)
L = len(layers)
#weights = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]
#biases = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]

# weights0 = [xavier_init([layers0[l], layers0[l+1]]) for l in range(0, L0-1)]
weights = [xavier_initw(l) for l in range(0, L-1)]
biases = [xavier_initb(l) for l in range(0, L-1)]




#### Scalar acoustic wave potential
phi = neural_net(tf.concat((x,z,t),axis=1), weights, biases)


#Note that stress field is isotropic for this case and there is no shear stress 
#so we only have one stress function that describes both xx and zz cmps which is the pressure
P = (1/Lx)**2*tf.gradients(tf.gradients(phi,x)[0],x)[0] + (1/Lz)**2*tf.gradients(tf.gradients(phi,z)[0],z)[0]

eq = tf.gradients(tf.gradients(phi,t)[0],t)[0] - alpha**2*P #Scalar Wave equation

ux= tf.gradients(phi,x)[0] #u=grad(phi)
uz= tf.gradients(phi,z)[0]
Vel_x=tf.gradients(ux,t)[0]#velocity field
Vel_z=tf.gradients(uz,t)[0]

### PDE residuals
n_pde=10000
print('n_pde',':',n_pde)
X_pde = sobol_sequence.sample(n_pde+1, 3)[1:,:]#first row is (0,0,0) which we don't need
X_pde[:,0] = X_pde[:,0] * ax/Lx
X_pde[:,1] = X_pde[:,1] * az/Lz
X_pde[:,2] = X_pde[:,2] * t_m#this is time


    

import os
sms = os.listdir('seismograms/.')

seismo_list = [np.loadtxt('seismograms/'+f) for f in sms]
lsmx=np.arange(0, len(seismo_list), 2)
lsmz=np.arange(1, len(seismo_list), 2)

U_spec_seisX=np.empty([np.size(seismo_list[0][:,0]), int(len(seismo_list)/2)])
U_spec_seisZ=np.empty([np.size(seismo_list[0][:,0]), int(len(seismo_list)/2)])

spec_t=1.2+seismo_list[0][:,0]#specfem's time starts from-1.2
for i in range(int(len(seismo_list)/2)):
    U_spec_seisX[:,i]=seismo_list[lsmx[i]][:,1]
    U_spec_seisZ[:,i]=seismo_list[lsmz[i]][:,1]
    
xrand=sobol_sequence.sample(np.shape(U_spec_seisZ)[0]+1, np.shape(U_spec_seisZ)[1])[1:,:]

fig = plt.figure()
plt.plot(spec_t, U_spec_seisZ[:,[4,12]])
plt.savefig('Seism_spec.png',dpi=400)
plt.show()
plt.close(fig)
 
#theta=90
#u0=1
#x0=4.0
#b=1
#
#rad=np.pi/180
#sin=np.sin(rad*theta)
#cos=np.cos(rad*theta)
#def U_initial(x):
#   return (u0*np.exp(-((x[:,0]-x0)*sin+x[:,1]*cos)**2/(b**2)))
#

n_ini=400
xx, zz = np.meshgrid(np.linspace(0,ax/Lx,n_ini),np.linspace(0,az/Lz,n_ini))
xxzz = np.concatenate((xx.reshape((-1,1)), zz.reshape((-1,1))),axis=1)
X_init = np.concatenate((xx.reshape((-1,1)),zz.reshape((-1,1)),0.0*np.ones((n_ini**2,1),dtype=np.float64)),axis=1)




[xxx, zzz] = np.meshgrid(np.linspace(0,ax/Lx,n_ini), np.linspace(0,az/Lz,n_ini))
xxx0, zzz0 = xxx.reshape((-1,1)), zzz.reshape((-1,1))

t_seis=np.linspace(0,t_m,200)
x_pre1=2/Lx*np.ones((t_seis.shape[0],1))
z_pre1 =6.0/Lz*np.ones((t_seis.shape[0],1))
t_seis=t_seis.reshape((-1,1))

x_pre2=6.0/Lx*np.ones((t_seis.shape[0],1))
z_pre2 =6.0/Lz*np.ones((t_seis.shape[0],1))



X_eval0=np.concatenate((xxx0,zzz0,0.0*np.ones((xxx0.shape[0],1))),axis=1)
X_evalt=np.concatenate((xxx0,zzz0,3*np.ones((xxx0.shape[0],1))),axis=1)


feed_dict0 = { x: X_eval0[:,0:1], z: X_eval0[:,1:2], t: X_eval0[:,2:3]}#this dictionary is for evaluating the initial condition on new test points
feed_dict2 = { x: X_evalt[:,0:1], z: X_evalt[:,1:2], t: X_evalt[:,2:3]}#this dictionary is for evaluating the wavefield on new test points
feed_dict_seis1 = { x: x_pre1, z: z_pre1, t: t_seis}#this dictionary is for evaluating seismograms
feed_dict_seis2 = { x: x_pre2, z: z_pre2, t: t_seis}#this dictionary is for evaluating seismograms



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ux0=sess.run([ux], feed_dict =feed_dict0 )
    uxt=sess.run([ux], feed_dict =feed_dict2 )
    uzt=sess.run([uz], feed_dict =feed_dict2 )
    disp_pred_z1=sess.run([uz], feed_dict =feed_dict_seis1 )
    disp_pred_z2=sess.run([uz], feed_dict =feed_dict_seis2 )

u_tot=((uxt[0].reshape(xxx.shape))**2+(uzt[0].reshape(xxx.shape))**2)**0.5
  

fig = plt.figure()
plt.plot(t_seis, disp_pred_z1[0],'b',label='PINNs 2km')
plt.plot(spec_t, U_spec_seisZ[:,4],'r',label='specfem 2km')
plt.savefig('Seismograms,x=2km,z=top.png',dpi=400)
plt.legend()
plt.show()
plt.close(fig)



fig = plt.figure()
plt.plot(t_seis, disp_pred_z2[0],'b',label='PINNs 6km')
plt.plot(spec_t, U_spec_seisZ[:,12],'r',label='specfem 6km')
plt.savefig('Seismograms,x=6km,z=top.png',dpi=400)
plt.legend()
plt.show()
plt.close(fig)

#fig = plt.figure()
#plt.contourf(xxx, zzz, ux0[0].reshape(xxx.shape),100, cmap='jet')
#plt.xlabel('x')
#plt.ylabel('z')
#plt.title(r'$u_x(x,z,t=0)$')
#plt.colorbar()
#plt.savefig('Predicted_wavefiled_t0.png',dpi=400)
#plt.show()
#plt.close(fig)
fig = plt.figure()
plt.contourf(xxx, zzz, u_tot.reshape(xxx.shape),100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title(r'$u_x(x,z,t=1)$')
plt.colorbar()
plt.savefig('Predicted_wavefiled_t6sec.png',dpi=400)
plt.show()
plt.close(fig)



