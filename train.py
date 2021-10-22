#Training SAE via SGD with Momentum

import pandas     as pd
import numpy      as np
import my_utility as ut
	
# Softmax's training
def train_softmax(x,y,param):
    W     = ut.iniW(y.shape[0],x.shape[0])
    V     = np.zeros(W.shape)    
    costo = []
    for iter in range(param[0]):        
        #complete code SGD with momentum              
    return(W,costo)

# Training AE miniBatch SGD
def train_batch(x,W,V,param):
    numBatch = np.int16(np.floor(x.shape[1]/param[0]))    
    for i in range(numBatch):                
        #complete code  SGD with momentum  
    return(W,V)
#Training AE by use SGD
def train_ae(x,hn,param):    
    W,V    = ut.iniWs(x.shape[0],param[hn])            
    for Iter in range(1,param[1]):        
        xe  = x[:,np.random.permutation(x.shape[1])]        
        W,V = train_batch(xe,W,V,param)                
    return(...) 
#SAE's Training 
def train_sae(x,param):    
    for hn in range(3,len(param)):
        #complete code  train AE      
    return(W,x) 

# Beginning ...
def main():
    p_sae,p_sft = ut.load_config()    
    xe          = ut.load_data_csv('train_x.csv')
    ye          = ut.load_data_csv('train_y.csv')
    W,Xr        = train_sae(xe,p_sae)     
    Ws, cost    = train_softmax(Xr,ye,p_sft)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

