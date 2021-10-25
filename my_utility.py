'''
INTEGRANTES
ETIENNE BELLENGER HERRERA   17619315-8
JUAN IGNACIO AVILA OJEDA    19013610-8
'''
# My Utility : auxiliary functions

import pandas as pd
import numpy  as np

# Initialize weights
def iniWs(prev,next):
    print(f'N: {next} \nP: {prev}')
    r = np.sqrt(6/(next + prev))
    W = np.random.rand(next,prev)
    W = W*2*r-r    

    V = np.zeros(W.shape)

    return(W,V)
    
# Initialize Matrix's weight    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

    
# STEP 1: Feed-forward of AE
def forward_ae(x,w):
    # Salida del encoder
    z1 = np.dot(w, x)
    a1 = act_sigmoid(z1)
    
    return (a1)
    
#Activation function
def act_sigmoid(z):
    return(1/(1+np.exp(-z)))   
# Derivate of the activation funciton
def deriva_sigmoid(a):
    return(a*(1-a))
def gradW_ae(a,x,w2,e):    
    z2 = deriva_sigmoid(a2) # 1, 375
    z1 = deriva_sigmoid(a1) # 20, 375
    
    # Calcular gradiente decoder
    delta2 = np.multiply(e, deriva_sigmoid(z2)) # Probar con a2
    dCdW2 = np.dot(delta2, a.T)
    # Calcular gradiente 
    delta1 = np.multiply( np.dot(w2.T, delta2), deriva_sigmoid(z1) )
    dCdW1 = np.dot( delta1, x.T)

    return dCdW1, dCdW2
# STEP 2: Feed-Backward for AE
def gradW_ae(a,W, e):    
    z = deriva_sigmoid(a) 

    # Calcular gradiente decoder
    delta2 = np.multiply(e, deriva_sigmoid(z)) # Probar con a2
    dCdW2 = np.dot(delta2, a.T)

    gW = (dCdW2)

    Cost = (1/2) * np.sum( np.sum( y * np.log10(a) ) ) 

    return(gW,Cost)  

# Update AE's weights via SGD Momentum
def updW_ae_sgd(w,v,gw,mu):
    beta= 0.9
    v = np.dot(beta, v)+mu*gw
    w -= v

    return(w,v)    

# Softmax's gradient
def gradW_softmax(x,y,w):    
    z = np.exp( np.dot(w, x) )
    a = softmax(z)
    _, N = y.shape

    Cost = (-1/N) * np.sum( np.sum( y * np.log10(a) ) ) 

    gW = ((-1/N) * (np.dot((y-a),np.transpose(x)))) 

    return(gW,Cost)

# Update Softmax's weights via SGD Momentum
def updW_sft_sgd(w,v,gw,mu):
    beta= 0.9
    #complete code
    return(w,v)         

# Calculate Softmax
def softmax(z):
    #complete code    
    return

# Métrica
def metricas(x,y):
    #complete code
    return 
    
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
    keys = [f'w{i}' for i in range(1,len(W)+1) ]
    w = dict(zip(keys, W))     
    # Guardar pesos
    np.savez_compressed('w_dl.npz', **w, ws=Ws ) 

    archivo = open('costo_softmax.csv', 'w')
    [ archivo.write(f'{c}\n') for c in cost ]
    archivo.close()
    
#load weight of the DL in numpy format
def load_w_dl():
    W = []
    [ W.append(np.load('w_dl.npz')[w]) for w in np.load('w_dl.npz').files ]
    
    return (W)        
# save weights in numpy format
def save_w_npy(w1,w2,mse):  
    #complete code
    return