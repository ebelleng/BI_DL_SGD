# My Utility : auxiliary functions

import pandas as pd
import numpy  as np

# Initialize weights
def iniWs(prev,next):
    w1 = iniW(next,prev)
    w2 = iniW(prev,next)
    W = (w1,w2)
    v1 = np.zeros(w1.shape)
    v2 = np.zeros(w2.shape)
    V = (v1,v2)
    return(W,V)
# Initialize Matrix's weight    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

    
# STEP 1: Feed-forward of AE
def forward_ae(x,w):
    w1,w2 = w
    z1 = np.dot(w1,x)
    a1 = act_sigmoid(z1)
    
    z2 = np.dot(w2, a1)
    a2 = act_sigmoid(z2)
    return (x,a1,a2)
#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# STEP 2: Feed-Backward for AE
def gradW_ae(a,W):
    x, a1,a2 = a
    w1,w2 = W
    z1 = deriva_sigmoid(a1)
    z2 = deriva_sigmoid(a2)
    e = (a2 - x)**2
    e = sum(e)

    delta2 = e**2 * deriva_sigmoid(z2)
    gW2 = np.dot(delta2, a1.T)

    delta1 = np.multiply( np.dot(w2.T, delta2), deriva_sigmoid(z1) )
    gW1 = np.dot( delta1, x.T)

    gW = (gW1,gW2)

    return(gW,e)    
# Update AE's weights via SGD Momentum
def updW_ae_sgd(w,v,gw,mu):
    beta= 0.9
    w1,w2 = w
    v1,v2 = v
    gw1,gw2 = gw
    # Ajuste pesos decoder
    v2 = beta*v2 + mu*gw2
    w2 = w2 - v2
    #Ajuste pesos encoder
    v1 = beta*v1 + mu*gw1
    w1 = w1 - v1

    w = (w1,w2)
    v = (v1,v2)
    return(w,v)    

# Softmax's gradient
def gradW_softmax(x,y,w):    
    z = np.dot(w,x)
    a = softmax(z)
    ya = y*np.log(a)
    cost = (-1/x.shape[1])*np.sum(np.sum(ya))
    gW = ((-1/x.shape[1])*np.dot((y-a),x.T))
    return(gW,cost)

# Update Softmax's weights via SGD Momentum
def updW_sft_sgd(w,v,gw,mu):
    beta= 0.9
    v = beta*v + mu*gw
    w = w - v
    return(w,v)         

# Calculate Softmax
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return(exp_z/exp_z.sum(axis=0,keepdims=True))

# Métrica
def metricas(x,y):
    confussion_matrix = np.zeros((y.shape[0], x.shape[0]))

    for real, predicted in zip(y.T, x.T):
        confussion_matrix[np.argmax(real)][np.argmax(predicted)] += 1    
    
    f_score = []
    for index, caracteristica in enumerate(confussion_matrix):
        TP = caracteristica[index]
        FP = confussion_matrix.sum(axis=0)[index] - TP
        FN = confussion_matrix.sum(axis=1)[index] - TP
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f_score.append(2 * (precision * recall) / (precision + recall))

    metrics = pd.DataFrame(f_score)
    metrics.to_csv("metrica_dl.csv", index=False, header=False)
    return (f_score)

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the SNN
def load_config():      
    par = np.genfromtxt('cnf_sae.csv',delimiter=',')    
    par_sae=[]
    par_sae.append(np.int16(par[0])) # Batch size
    par_sae.append(np.int16(par[1])) # MaxIter
    par_sae.append(np.float(par[2])) # Learn rate    
    for i in range(3,len(par)):
        par_sae.append(np.int16(par[i]))
    par    = np.genfromtxt('cnf_softmax.csv',delimiter=',')
    par_sft= []
    par_sft.append(np.int16(par[0]))   #MaxIters
    par_sft.append(np.float(par[1]))   #Learning 
    return(par_sae,par_sft)
# Load data 
def load_data_csv(fname):
    x   = pd.read_csv(fname, header = None)
    x   = np.array(x)   
    return(x)

# save weights SAE and costo of Softmax
def save_w_dl(W,Ws,cost):    
    np.savetxt('costo_softmax.csv', cost, delimiter=",")
    W.append(Ws)
    np.savez('w_dl.npz', W=W)
    return()
    
#load weight of the DL in numpy format
def load_w_dl():
    w = np.load('w_dl.npz', allow_pickle=True)
    return w['W']
    
# save weights in numpy format
def save_w_npy(w1,w2,mse):  
    return
    

