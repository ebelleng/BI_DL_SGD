# My Utility : auxiliary functions

import pandas as pd
import numpy  as np

# Initialize weights
def iniWs(prev,next):
    #comple code
    return(W,V)
# Initialize Matrix's weight    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

    
# STEP 1: Feed-forward of AE
def forward_ae(x,w):
    #complete code
    return(...)    
#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))

# STEP 2: Feed-Backward for AE
def gradW_ae(a,W):    
    #comple code
    return(gW,Cost)    
# Update AE's weights via SGD Momentum
def updW_ae_sgd(w,v,gw,mu):
    beta= 0.9
    #comple code    
    return(w,v)    

# Softmax's gradient
def gradW_softmax(x,y,w):    
    #complte code    
    return(gW,Cost)

# Update Softmax's weights via SGD Momentum
def updW_sft_sgd(w,v,gw,mu):
    beta= 0.9
    #complete code
    return(w,v)         

# Calculate Softmax
def softmax(z):
    #complete code    

# Métrica
def metricas(x,y):
    #complete code
    return(...)
    
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
    par_sft.append(np.float(par[2]))   #Lambda
    return(par_sae,par_sft)
# Load data 
def load_data_csv(fname):
    x   = pd.read_csv(fname, header = None)
    x   = np.array(x)   
    return(x)

# save weights SAE and costo of Softmax
def save_w_dl(W,Ws,cost):    
    #complete code
    
#load weight of the DL in numpy format
def load_w_dl():
    #complete code
    
# save weights in numpy format
def save_w_npy(w1,w2,mse):  
    #complete code
    

