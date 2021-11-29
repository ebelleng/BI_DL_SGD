#Training SAE via SGD with Momentum

import pandas     as pd
import numpy      as np
import my_utility as ut
	
# Softmax's training
def train_softmax(x,y,param):
    w = ut.iniW(y.shape[0],x.shape[0])
    v = np.zeros(np.shape(w))
    costo = []
    for iter in range(param[0]):
        gW,cost = ut.gradW_softmax(x,y,w) 
        costo.append(cost)
        w,v     = ut.updW_sft_sgd(w,v,gW,mu=param[1]) 
    return(w,costo)

# Training AE miniBatch SGD
def train_batch(x,W,V,param):
    numBatch = np.int16(np.floor(x.shape[1]/param[0]))
    mu = param[2]
    for i in range(numBatch):
        #complete code  SGD with momentum  
        a = ut.forward_ae(x,W) 
        gW, c = ut.gradW_ae(a,W)
        W,V = ut.updW_ae_sgd(W,V,gW,mu)
    
    return (W,V)

#Training AE by use SGD
def train_ae(x,hn,param):   
    W,V    = ut.iniWs(x.shape[0],hn)
    for Iter in range(1,param[1]):        
        xe  = x[:,np.random.permutation(x.shape[1])]        
        W,V = train_batch(xe,W,V,param)                
    return W[0]

#SAE's Training 
def train_sae(x,param):    
    W = []
    data = x
    for hn in range(3,len(param)):
        print('AE={} Hnode={}'.format(hn-2,param[hn]))
        # Se entrena el encoder
        w = train_ae(data, param[hn], param)
        # Se guarda el peso del encoder
        W.append(w)
        # Se calcula la nueva data
        data = ut.act_sigmoid( np.dot(w, data))

    return(W,data)

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

