#In this code we are solving the wave equations 
#with known, constant seismic velocity using B.C, I.C and PDE
import pickle

import tensorflow as tf
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

tf.reset_default_graph()    

x = tf.placeholder(tf.float64, shape=(None,1))
z = tf.placeholder(tf.float64, shape=(None,1))
t = tf.placeholder(tf.float64, shape=(None,1))

rho=1.0
ax=16.0#dimension of the domain in the x direction
az=10.0#dimension of the domain in the z direction
t_m=5.0#total time for PDE training
t_s=5.0#total time series used from the seismograms

s_spec=5.1e-4#specfem time stepsize
t01=5*s_spec#initial disp. input at this time from spec
t02=100*s_spec#sec "initial" disp. input at this time from spec instead of enforcing initial velocity
t_la=4000*s_spec# test data for comparing specfem and trained PINNs



n_event=2
n_seis=17#number of input seismograms from SPECFEM; if events have different 
#numbers of seismograms, you have to change the lines containing n_seis accordingly
x0_s=0.0# x location of the first seismogram from SPECFEM.Here it must
# be in km while in SPECFEM it's in meters. Note we assume seismograms are
# on the surface and start from the same location for all events;
# otherwise X_S needs to change accordingly in the code
xl_s=ax# x location of the last seismogram on the surface. this doesn't have 
#to be the same as ax and it can be smaller, change
#this accordingly based on what you used from specfem 



Lx=3;#this is for normalizing the wavespeed in the PDE via saling x coordinate
Lz=3;#this is for normalizing the wavespeed in the PDE via scaling z coordinate
####  Acoustic wave's speed
#alpha=3-1*tf.exp(-((x*Lx-6)**2+(z*Lz-4)**2)/(2**2))#4+tf.tanh(100*(x-4))



def g(x,z,a,b,c,d):
  return ((x-c)**2/a**2+(z-d)**2/b**2)


alpha_true=3-0.5*(1+tf.tanh(100*(1-g(x*Lx,z*Lz,3,2,9,6))))

#alpha_true=7-(1+tf.tanh(100*(1-g(x*Lx,z*Lz,2,1,9,6))))-0.5*(1+tf.tanh(100*(1-g(x*Lx,z*Lz,3,1,15,4))))
         

#alpha_true=3-0.5*(1+tf.tanh(100*(z*Lz-3-3*g)))



ub=np.array([ax/Lx,az/Lz,t_m]).reshape(-1,1).T#Maziyar's normalization of the input to the NN, but since we are scaling the spatial coordinates so these need to be scaled accordingly 
ub0=np.array([ax/Lx,az/Lz]).reshape(-1,1).T#for the NN estimating the wave_speed 



def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]        
    xavier_stddev = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev,dtype=tf.float64), dtype=tf.float64)

    
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

def neural_net0(X, weights, biases):
    num_layers = len(weights) + 1    
    H=2*(X/ub0)-1#normalization map to [-1 1]
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

layers=[3]+[100]*8+[1] # layers for the NN approximating the pressure field; scalar pressure output 
                        #and three inputs x,z,t
print('NN layers',':',layers)
L = len(layers)
weights = [xavier_init([layers[l], layers[l+1]]) for l in range(0, L-1)]
biases = [tf.Variable( tf.zeros((1, layers[l+1]),dtype=tf.float64)) for l in range(0, L-1)]
num_epoch = 10000001

layers0=[2]+[20]*5+[1] # layers for the second NN to approximate Vp; The input is x,z

L0 = len(layers0)
weights0 = [xavier_init([layers0[l], layers0[l+1]]) for l in range(0, L0-1)]
biases0 = [tf.Variable( tf.zeros((1, layers0[l+1]),dtype=tf.float64)) for l in range(0, L0-1)]


learning_rate = 1.e-4

alpha_star=tf.tanh(neural_net0(tf.concat((x,z),axis=1), weights0, biases0))

alpha_bound=0.5*(1+tf.tanh(100*(z-3/Lz)))*0.5*(1+tf.tanh(100*(-z+9/Lz)))*0.5*(1+tf.tanh(100*(x-5/Lx)))*0.5*(1+tf.tanh(100*(-x+14/Lx)))#confining the inversion to a box and not the whole region

alpha=3+alpha_star*alpha_bound

#### Scalar acoustic wave potential
phi = neural_net(tf.concat((x,z,t),axis=1), weights, biases)

#ux, uz, = U[:,0:1], U[:,1:2]

#Note that stress field is isotropic for this case and there is no shear stress 
#so we only have one stress function that describes both xx and zz cmps which is the pressure
#so we only have one stress function that describes both xx and zz cmps which is the pressure
#Note it's better to scale the pressure by the maximum of the wavespeed in the domain so that both terms in PDE have the same order of magnitude which I saw frequently is the only case that PINNs converges well.
P = (1/Lx)**2*tf.gradients(tf.gradients(phi,x)[0],x)[0] + (1/Lz)**2*tf.gradients(tf.gradients(phi,z)[0],z)[0]

eq = tf.gradients(tf.gradients(phi,t)[0],t)[0] - alpha**2*P #Scalar Wave equation

ux= tf.gradients(phi,x)[0] #u=grad(phi)
uz= tf.gradients(phi,z)[0]
Vel_x=tf.gradients(ux,t)[0]#velocity field
Vel_z=tf.gradients(uz,t)[0]

### PDE residuals
batch_size=40000
n_pde=batch_size*2000
print('batch_size',':',batch_size)
X_pde = sobol_sequence.sample(n_pde+1, 3)[1:,:]#first row is (0,0,0) which we don't need
X_pde[:,0] = X_pde[:,0] * ax/Lx
X_pde[:,1] = X_pde[:,1] * az/Lz
X_pde[:,2] = X_pde[:,2] * t_m#this is time



###initial conditions for all events
X0=np.loadtxt('event1/wavefields/wavefield_grid_for_dumps_000.txt')# coordinates on which the wavefield output is recorded on specfem. It's the same for all the runs with the same meshing system in specfem
#if you use different meshing system for different events from specfem, then
# you need to use different X0 for each event. This won't add any computational
#burden since X0 is only used for interpolation and the interpolated data for 
# all events can be mapped onto a single grid (xx,zz given below) for all events
X0=X0/1000#specfem works with meters unit so we need to convert them to Km
X0[:,0:1]=X0[:,0:1]/Lx#scaling the spatial domain
X0[:,1:2]=X0[:,1:2]/Lz#scaling the spatial domain
xz=np.concatenate((X0[:,0:1],X0[:,1:2]),axis=1)


n_ini=40#Note specfem first snapshot is after 5 steps and not at zero step
xx, zz = np.meshgrid(np.linspace(0,ax/Lx,n_ini),np.linspace(0,az/Lz,n_ini))
xxzz = np.concatenate((xx.reshape((-1,1)), zz.reshape((-1,1))),axis=1)
X_init1 = np.concatenate((xx.reshape((-1,1)),zz.reshape((-1,1)),t01*np.ones((n_ini**2,1),dtype=np.float64)),axis=1)#for enforcing the disp I.C
X_init2 = np.concatenate((xx.reshape((-1,1)),zz.reshape((-1,1)),t02*np.ones((n_ini**2,1),dtype=np.float64)),axis=1)#for enforcing the sec I.C, another snapshot of specfem

u_scl=4 #scaling the input data to smaller than 1 


import os

wfs = sorted(os.listdir('event1/wavefields/.'))
U0 = [np.loadtxt('event1/wavefields/'+f) for f in wfs]

U_ini1 = interpolate.griddata(xz, U0[0], xxzz, fill_value=0.0)
U_ini1x=U_ini1[:,0:1]/u_scl
U_ini1z=U_ini1[:,1:2]/u_scl


U_ini2 = interpolate.griddata(xz, U0[1], xxzz, fill_value=0.0)
U_ini2x=U_ini2[:,0:1]/u_scl
U_ini2z=U_ini2[:,1:2]/u_scl

U_spec = interpolate.griddata(xz, U0[2], xxzz, fill_value=0.0)#Test data. We don't scale this only for illustration purposes
U_specx=U_spec[:,0:1]
U_specz=U_spec[:,1:2]



#the first event's data has been uploaded above and below I add
#the rest of the n-1 events
for ii in range(n_event-1):
    wfs = sorted(os.listdir('event'+str(ii+2)+'/wavefields/.'))
    U0 = [np.loadtxt('event'+str(ii+2)+'/wavefields/'+f) for f in wfs]

    U_ini1 = interpolate.griddata(xz, U0[0], xxzz, fill_value=0.0)
    U_ini1x +=U_ini1[:,0:1]/u_scl
    U_ini1z +=U_ini1[:,1:2]/u_scl


    U_ini2 = interpolate.griddata(xz, U0[1], xxzz, fill_value=0.0)
    U_ini2x +=U_ini2[:,0:1]/u_scl
    U_ini2z +=U_ini2[:,1:2]/u_scl

    U_spec = interpolate.griddata(xz, U0[2], xxzz, fill_value=0.0)
    U_specx +=U_spec[:,0:1]
    U_specz +=U_spec[:,1:2]
#U_ini=U_ini.reshape(-1,1)



################### plots of inputs for sum of the events
fig = plt.figure()
plt.contourf(xx, zz, np.sqrt(U_ini1x**2+U_ini1z**2).reshape(xx.shape),100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Scaled I.C total disp. input specfem t='+str(t01))
plt.colorbar()
plt.savefig('Ini_total_disp_spec_sumEvents.png', dpi=400)
plt.show()
plt.close(fig)



fig = plt.figure()
plt.contourf(xx, zz, np.sqrt(U_ini2x**2+U_ini2z**2).reshape(xx.shape),100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Scaled sec I.C total disp. input specfem t='+str(round(t02, 4)))
plt.colorbar()
plt.savefig('sec_wavefield_input_spec_sumEvents.png', dpi=400)
plt.show()
plt.close(fig)



fig = plt.figure()
plt.contourf(xx, zz, np.sqrt(U_specx**2+U_specz**2).reshape(xx.shape),100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Test data: Total displacement specfem t='+str(round(t_la, 4)))
plt.colorbar()
plt.savefig('total_disp_spec_testData_sumEvents.png', dpi=400)
plt.show()
plt.close(fig)
###############################################################




#################input seismograms for the first event


import os
sms = sorted(os.listdir('event1/seismograms/.'))
seismo_list = [np.loadtxt('event1/seismograms/'+f) for f in sms if f[-6]=='Z']#uploading only Z components for the acoustic case

t_spec=1.2+seismo_list[0][:,0]#specfem's time starts from-1.2
cut=t_spec>t_s#here we include only part of the seismograms from specfem that are within the PINNs training time domain which is [0 t_m]
l_s=len(cut)-sum(cut)#this is the index of the time axis in specfem after which t>t_m

t_spec=t_spec[:l_s,]


l_f=100#subsampling seismograms from specfem
index = np.arange(0,l_s,l_f) #subsampling every l_s time steps from specfem
l_sub=len(index)
t_spec_sub=t_spec[index].reshape((-1,1))#subsampled time axis of specfem for the seismograms


for ii in range(len(seismo_list)):
    seismo_list[ii]=seismo_list[ii][index]


fig = plt.figure()
for ii in range(len(seismo_list)):
    plt.plot(seismo_list[ii][:,0],seismo_list[ii][:,1])
plt.savefig('input_Seismograms_event1.png',dpi=400)


Sz=seismo_list[0][:,1].reshape(-1,1)
for ii in range(len(seismo_list)-1):
    Sz=np.concatenate((Sz,seismo_list[ii+1][:,1].reshape(-1,1)),axis=0)

#################################################################
#######input seismograms for the rest of the events added to the first event
    
for ii in range(n_event-1):
    sms = sorted(os.listdir('event'+str(ii+2)+'/seismograms/.'))
    seismo_list = [np.loadtxt('event'+str(ii+2)+'/seismograms/'+f) for f in sms if f[-6]=='Z']
    
    for jj in range(len(seismo_list)):
        seismo_list[jj]=seismo_list[jj][index]


    fig = plt.figure()
    for jj in range(len(seismo_list)):
        plt.plot(seismo_list[jj][:,0],seismo_list[jj][:,1])
    plt.savefig('input_Seismograms_event'+str(ii+2)+'.png',dpi=400)


    Sze=seismo_list[0][:,1].reshape(-1,1)
    for jj in range(len(seismo_list)-1):
       Sze=np.concatenate((Sze,seismo_list[jj+1][:,1].reshape(-1,1)),axis=0)
       
    Sz +=Sze
###########################################################


Sz=Sz/u_scl #scaling the sum of all seismogram inputs


#X_S is the training collection of input coordinates in space-time for all seismograms
X_S=np.empty([int(np.size(Sz)), 3])

d_s=(xl_s-x0_s)/(n_seis-1)#the distance between seismograms

for i in range(len(seismo_list)):
  X_S[i*l_sub:(i+1)*l_sub,]=np.concatenate(((x0_s+i*d_s)/Lx*np.ones((l_sub,1),dtype=np.float64), \
                               az/Lz*np.ones((l_sub,1),dtype=np.float64),t_spec_sub),axis=1)


#fig = plt.figure()
#plt.plot(X_S[:,2],Sz)
#plt.savefig('input_Seismograms.png',dpi=400)

fig = plt.figure()
plt.plot(Lx*X_S[:,0],Lz*X_S[:,1],'r*')
plt.ylim(top=Lz*X_S[0,1])
plt.grid(color='k', linestyle='-', linewidth=1)
plt.title('seismograms locations')
plt.savefig('Seismo_loc.png',dpi=400)


fig = plt.figure()
for i in range(len(seismo_list)):
    plt.plot(X_S[i*l_sub:(i+1)*l_sub,2],Sz[i*l_sub:(i+1)*l_sub])
plt.title('Scaled sum over all events')
plt.savefig('input_Seismograms_sum_all_events.png',dpi=400)

#####  BCs: Free stress on top and no BC for other sides (absorbing)
bcxn=100
bctn=50
x_vec = np.random.rand(bcxn,1)*ax/Lx
t_vec = np.random.rand(bctn,1)*t_m
xxb, ttb = np.meshgrid(x_vec, t_vec)
X_BC_t = np.concatenate((xxb.reshape((-1,1)),az/Lz*np.ones((xxb.reshape((-1,1)).shape[0],1)),ttb.reshape((-1,1))),axis=1)
#X_BC_r = np.concatenate((a*np.ones((xxb.reshape((-1,1)).shape[0],1)),xxb.reshape((-1,1)),ttb.reshape((-1,1))),axis=1)
#X_BC_b = np.concatenate((xxb.reshape((-1,1)),0.0*np.ones((xxb.reshape((-1,1)).shape[0],1)),ttb.reshape((-1,1))),axis=1)
#X_BC_l = np.concatenate((0.0*np.ones((xxb.reshape((-1,1)).shape[0],1)),xxb.reshape((-1,1)),ttb.reshape((-1,1))),axis=1)
#Note above four lines go clock-wise -->t-->R-->b-->l. Not that it matters, it only makes the code more intuitive


#input locations projected on time=0 plane
#plt.plot(X_pde[:,0], X_pde[:,1],'r.', X_init[:,0], X_init[:,1], 'b.',
 #        X_BC_t[:,0],X_BC_t[:,1],'r.',X_BC_b[:,0],X_BC_b[:,1],'r.',
  #       X_BC_l[:,0],X_BC_l[:,1],'r.',X_BC_r[:,0],X_BC_r[:,1],'r.') #Note this is missleading cause the ploting only x and z of the X_pde or X_stress doesn't show you the spatial density at every time stpe!  These points are scattered at different times 
#plt.savefig('input_locations.png', dpi=400)
#plt.show()
#plt.close(fig)




N1 = batch_size
N2 = X_init1.shape[0]
N3 = X_init2.shape[0]
N4 = X_S.shape[0]

XX = np.concatenate((X_pde[0:batch_size], X_init1,X_init2, X_S, X_BC_t),axis=0)
#XX = np.concatenate((X_pde, X_init, X_BC_t,X_BC_r,X_BC_b,X_BC_l),axis=0)

feed_dict1 = { x: XX[:,0:1], z: XX[:,1:2], t: XX[:,2:3]} # This dictionary is for training



loss_pde = tf.reduce_mean(tf.square(eq[:N1,0:1]))

loss_init_disp1 = tf.constant(0.0, dtype=tf.float64)
loss_init_disp2 = tf.constant(0.0, dtype=tf.float64)


loss_init_disp1 = tf.reduce_mean(tf.square(ux[N1:(N1+N2),0:1]-U_ini1x)) \
          + tf.reduce_mean(tf.square(uz[N1:(N1+N2),0:1]-U_ini1z))

loss_init_disp2 = tf.reduce_mean(tf.square(ux[(N1+N2):(N1+N2+N3),0:1]-U_ini2x)) \
          + tf.reduce_mean(tf.square(uz[(N1+N2):(N1+N2+N3),0:1]-U_ini2z))

loss_seism = tf.reduce_mean(tf.square(ux[(N1+N2+N3):(N1+N2+N3+N4),0:1]-0*Sz)) \
          + tf.reduce_mean(tf.square(uz[(N1+N2+N3):(N1+N2+N3+N4),0:1]-Sz))
#Note for the acoustic case U_spec_seisX=0

          
#free stress BC on top surface
#All other sides, no BC
loss_BC = tf.reduce_mean(tf.square(P[(N1+N2+N3+N4):,0:1]))


#tf.reduce_mean(tf.square(P[(N1+N2):(N1+N2+N3),0:1]-P[(N1+N2+2*N3):(N1+N2+3*N3),0:1]))\
            #+ tf.reduce_mean(tf.square(P[(N1+N2+N3):(N1+N2+2*N3),0:1]))\
            #+ tf.reduce_mean(tf.square(P[(N1+N2+3*N3):,0:1]))


loss = 1e-2*loss_pde + loss_init_disp1 +loss_init_disp2+loss_seism+loss_BC

#loss = loss_init_disp



#plt.plot(x_bound[:,2:3], Ux_bound)
#plt.plot(x_bound[:,2:3], Uz_bound,'r.')
#plt.show()

optimizer_Adam = tf.train.AdamOptimizer(learning_rate)
train_op_Adam = optimizer_Adam.minimize(loss)   
#
##optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, 
##                                                   method = 'L-BFGS-B', 
##                                                   options = {'maxiter': 50000,
##                                                   'maxfun': 50000,
##                                                   'maxcor': 50,
##                                                   'maxls': 50,
##                                                   'ftol' : 1.0 * np.finfo(float).eps})      
#
#batch_size = 10000
#num_batch = int(Y_train2.shape[0]/batch_size)



#XX = np.concatenate((X_obs,X_pde),axis=0)
#XX =X_obs


##path = './t1_results/'
##


xx0, zz0 = xx.reshape((-1,1)), zz.reshape((-1,1))
#x_pre1, z_pre = [3.0/Lx, az/Lz] #predicted displacement at a seismogram at the top surface midpoint
#x_pre2, z_pre = [6.0/Lx, az/Lz] #predicted displacement at a seismogram at the top surface midpoint

X_eval01=np.concatenate((xx0,zz0,t01*np.ones((xx0.shape[0],1))),axis=1)#evaluating PINNs at time=0
X_eval02=np.concatenate((xx0,zz0,t02*np.ones((xx0.shape[0],1))),axis=1)#evaluating PINNs at time when the second input from specfem is provided
X_evalt=np.concatenate((xx0,zz0,t_la*np.ones((xx0.shape[0],1))),axis=1)#evaluating PINNs at a later time>0

feed_dict01 = { x: X_eval01[:,0:1], z: X_eval01[:,1:2], t: X_eval01[:,2:3]}#this dictionary is for evaluating the initial condition recovered from PINNs on new test points other than the ones used for training
feed_dict02 = { x: X_eval02[:,0:1], z: X_eval02[:,1:2], t: X_eval02[:,2:3]}#this dictionary is for evaluating the initial condition recovered from PINNs on new test points other than the ones used for training
feed_dict2 = { x: X_evalt[:,0:1], z: X_evalt[:,1:2], t: X_evalt[:,2:3]}#this dictionary is for evaluating PINNs at a later time>0
feed_dict_seism={ x: X_S[:,0:1], z: X_S[:,1:2], t: X_S[:,2:3]}
i=int(-1)
loss_eval=np.zeros((1,7))
loss_rec=np.empty((0,7))


with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      alpha_true0=sess.run([alpha_true], feed_dict =feed_dict01 )#note alpha takes two variables but feed_dict01 has three input. but it's ok and won't cause any issues 

alpha_true0 = alpha_true0[0].reshape((xx.shape))

fig = plt.figure()
plt.contourf(Lx*xx, Lz*zz, alpha_true0.reshape((xx.shape)), 100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title(r'True acoustic wavespeed ($\alpha$)')
plt.colorbar()
plt.plot(Lx*X_S[:,0],Lz*X_S[:,1]+0.1,'r*',markersize=5)
plt.savefig('True_wavespeed.png', dpi=400)
plt.show()
plt.close(fig)

with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      alpha_plot=sess.run([alpha], feed_dict =feed_dict01 )#note alpha takes two variables but feed_dict01 has three input. This is ok and won't cause any issues 

alpha_plot = alpha_plot[0].reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, zz, alpha_plot.reshape((xx.shape)), 100, cmap='jet')
plt.xlabel('x')
plt.ylabel('z')
plt.title(r'Initial guess ($\alpha$)')
plt.colorbar()
plt.savefig('Ini_guess_wavespeed.png', dpi=400)
plt.show()
plt.close(fig)

bbn=0
with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())  
#      optimizer.minimize(sess, fetches = [loss])   # L-BFGS-B algorithm 
#      uu_high=net_high(X_test2,weights_L1, weights_H1,biases_L1, biases_H1)
#      uu_low=net_low(X_test1,weights_L1,biases_L1)
#      
      #phi0 = neural_net(X_eval0,weights, biases) #scalar wave potential at time zero (initial condition))
      #phi3 = neural_net(X_evalt,weights, biases)#calar wave potential at time 0.5
      
      #u=grad(phi)
      #disp = neural_net(np.concatenate((xxx0,zzz0,2*np.ones((xxx0.shape[0],1))),axis=1),\
       #                 weights, biases)#displacement field at time=2sec
      #seismo_pre1 = neural_net(np.concatenate((x_pre1*np.ones((t_seism.shape[0],1)),z_pre*np.ones((t_seism.shape[0],1)),t_seism),axis=1),\
       #                 weights, biases) # This produces the predicted seismogram at the location of first observed seismogram 
      #seismo_pre2 = neural_net(np.concatenate((x_pre2*np.ones((t_seism.shape[0],1)),z_pre*np.ones((t_seism.shape[0],1)),t_seism),axis=1),\
       #                 weights, biases)
      start = timeit.default_timer()
      for epoch in range(num_epoch):

          sess.run(train_op_Adam, feed_dict = feed_dict1)  
   
          if epoch % 500 == 0:
              stop = timeit.default_timer()
              print('Time: ', stop - start)
              loss_val, loss_pde_val, loss_init_disp1_val,loss_init_disp2_val,loss_seism_val, loss_BC_val \
              = sess.run([loss, loss_pde, loss_init_disp1,loss_init_disp2,loss_seism, loss_BC], feed_dict = feed_dict1)

              print ('Epoch: ', epoch, ', Loss: ', loss_val, ', Loss_pde: ', loss_pde_val, ', Loss_init_disp1: ', loss_init_disp1_val)
              print (', Loss_init_disp2: ', loss_init_disp2_val,'Loss_seism: ', loss_seism_val,'Loss_stress: ', loss_BC_val)

              ux01=sess.run([ux], feed_dict =feed_dict01 )
              uz01=sess.run([uz], feed_dict =feed_dict01 )
              ux02=sess.run([ux], feed_dict =feed_dict02 )
              uz02=sess.run([uz], feed_dict =feed_dict02 )
              uxt=sess.run([ux], feed_dict =feed_dict2 )
              uzt=sess.run([uz], feed_dict =feed_dict2 )
              uz_seism_pred=sess.run([uz], feed_dict =feed_dict_seism )
              alpha0=sess.run([alpha], feed_dict =feed_dict01 )
              i=i+1
              loss_eval[0,0],loss_eval[0,1],loss_eval[0,2],loss_eval[0,3],loss_eval[0,4],loss_eval[0,5],loss_eval[0,6]\
              =epoch,loss_val, loss_pde_val, loss_init_disp1_val,loss_init_disp2_val,loss_seism_val, loss_BC_val

              loss_rec= np.concatenate((loss_rec,loss_eval),axis=0)

              #####Defining a new training batch for both PDE and B.C input data
              x_vec = np.random.rand(bcxn,1)*ax/Lx
              t_vec = np.random.rand(bctn,1)*t_m 
              xxb, ttb = np.meshgrid(x_vec, t_vec)
              X_BC_t = np.concatenate((xxb.reshape((-1,1)),az/Lz*np.ones((xxb.reshape((-1,1)).shape[0],1)),ttb.reshape((-1,1))),axis=1)

              bbn=bbn+1
              XX = np.concatenate((X_pde[bbn*batch_size:(bbn+1)*batch_size], X_init1,X_init2,X_S, X_BC_t),axis=0)
              feed_dict1 = { x: XX[:,0:1], z: XX[:,1:2], t: XX[:,2:3]} # This dictionary is for training

              #seismo_pre_val1,seismo_pre_val2,disp0x_val, dispx_val = sess.run([seismo_pre1,seismo_pre2,disp0[:,0:1], disp[:,0:1]])
              #phi0_val = sess.run([phi0])
              #phi3_val = sess.run([phi3])
              

              #fig = plt.figure()
              #plt.plot(t_seism, seismo_pre_val1[:,0:1], 'r', t_seism, Ux_obs_list[n-1], 'b')
#              plt.plot(t_seism, seismo_pre_val1[:,0:1], 'r')
#              plt.plot(t_seism, seismo_pre_val2[:,0:1], 'b')
#              plt.savefig('output/Seismograms.png',dpi=400)
#              plt.show()
#              plt.close(fig)
              U_PINN01=((ux01[0].reshape(xx.shape))**2+(uz01[0].reshape(xx.shape))**2)**0.5
              U_PINN02=((ux02[0].reshape(xx.shape))**2+(uz02[0].reshape(xx.shape))**2)**0.5
              U_PINNt=u_scl*((uxt[0].reshape(xx.shape))**2+(uzt[0].reshape(xx.shape))**2)**0.5
              U_diff=np.sqrt(U_specx**2+U_specz**2).reshape(xx.shape)-U_PINNt
              fig = plt.figure()
              plt.contourf(xx, zz, U_PINN01,100, cmap='jet')
              plt.xlabel('x')
              plt.ylabel('z')
              plt.title(r'PINNs $U(x,z,t=$'+str(round(t01, 4))+r'$)$')
              plt.colorbar()
              plt.savefig('Total_Predicted_dispfield_t='+str(round(t01, 4))+'.png',dpi=400)
              plt.show()
              plt.close(fig)
              fig = plt.figure()
              plt.contourf(xx, zz, U_PINN02,100, cmap='jet')
              plt.xlabel('x')
              plt.ylabel('z')
              plt.title(r'PINNs $U(x,z,t=$'+str(round(t02, 4))+r'$)$')
              plt.colorbar()
              plt.savefig('Total_Predicted_dispfield_t='+str(round(t02, 4))+'.png',dpi=400)
              plt.show()
              plt.close(fig)
              fig = plt.figure()
              plt.contourf(xx, zz, U_PINNt,100, cmap='jet')
              plt.xlabel('x')
              plt.ylabel('z')
              plt.title(r'PINNs $U(x,z,t=$'+str(round(t_la, 4))+r'$)$')
              plt.colorbar()
              plt.savefig('Total_Predicted_dispfield_t='+str(round(t_la, 4))+'.png',dpi=400)
              plt.show()
              plt.close(fig)
              
              fig = plt.figure()
              plt.contourf(xx, zz, U_diff,100, cmap='jet')
              plt.xlabel('x')
              plt.ylabel('z')
              plt.title(r'Total disp. Specfem-PINNs ($t=$'+str(round(t_la, 4))+r'$)$')
              plt.colorbar()
              plt.savefig('pointwise_Error_spec_minus_PINNs_t='+str(round(t_la, 4))+'.png',dpi=400)
              plt.show()
              plt.close(fig)
              
              fig = plt.figure()
              plt.contourf(xx, zz, alpha0[0].reshape(xx.shape),100, cmap='jet')
              plt.xlabel('x')
              plt.ylabel('z')
              plt.title(r'Inverted $\alpha$')
              plt.colorbar()
              plt.savefig('inverted_alpha.png',dpi=400)
              plt.show()
              plt.close(fig)
              
              fig = plt.figure()
              plt.contourf(xx, zz, alpha_true0-(alpha0[0].reshape(xx.shape)),100, cmap='jet')
              plt.xlabel('x')
              plt.ylabel('z')
              plt.title(r' $\alpha$ misfit (true-inverted)')
              plt.colorbar()
              plt.savefig('alpha_misfit.png',dpi=400)
              plt.show()
              plt.close(fig)

              fig = plt.figure()
              plt.plot(loss_rec[0:,0], loss_rec[0:,4],'g',label='ini_disp2')
              plt.plot(loss_rec[0:,0], loss_rec[0:,6],'black',label='B.C')
              plt.plot(loss_rec[0:,0], loss_rec[0:,1],'--y',label='Total')
              plt.plot(loss_rec[0:,0], loss_rec[0:,2],'r',label='PDE')
              plt.plot(loss_rec[0:,0], loss_rec[0:,3],'b',label='ini_disp1')
              plt.plot(loss_rec[0:,0], loss_rec[0:,5],'c',label='Seism')
              plt.yscale("log")
              plt.xlabel('epoch')
              plt.ylabel('misfit')
              plt.legend()
              plt.savefig('misfit.png',dpi=400)
              plt.show()
              plt.close(fig)
              
              fig = plt.figure()
              plt.plot(X_S[:,2],Sz,'b',label='Input_seism')
              plt.plot(X_S[:,2],uz_seism_pred[0],'r',label='Pred_seism')
              plt.legend()
              plt.savefig('Seismograms_compare1.png',dpi=400)
              plt.show()
              plt.close(fig)
          

              w_f=sess.run(weights)#saving weights at each 100 iterations
              b_f=sess.run(biases)#saving biases at each 100 iterations
              w_alph=sess.run(weights0)#saving weights at each 100 iterations
              b_alph=sess.run(biases0)
              with open('recorded_weights.pickle', 'wb') as f:
                   pickle.dump(['The first tensor contains weights, the second biases and the third losses',w_f,b_f,w_alph,b_alph,loss_rec], f)
              

#
#fig = plt.figure()
#plt.plot(t_seism,Ux_obs[:,30])
#






