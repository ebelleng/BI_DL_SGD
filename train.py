#Training SAE via SGD with Momentum

import pandas     as pd
import numpy      as np
import my_utility as ut
	
# Softmax's training
def train_softmax(x,y,param):
    W     = ut.iniW(y.shape[0],x.shape[0])
    V     = np.zeros(W.shape)
    beta  = 0.9 
    mu = param[1]
    costo = []
    for iter in range(param[0]):        
        gW, c = ut.grad_softmax(x,y,W,lambW=param[2])
        
        costo.append(c)

        V = beta * V + mu * gW
        W = W - V
    return(W,costo)

# Training AE miniBatch SGD
def train_batch(x,W,V,param):
    numBatch = np.int16(np.floor(x.shape[1]/param[0]))
    mu = param[2]
    print(numBatch)
    for i in range(numBatch):
        #complete code  SGD with momentum  
        a = ut.forward_ae(x,W) 

        # Calcular el error
        error = np.sum( (a - x) ** 2) 
        error = np.sum(error) * (1/2)
        
        # Calcular el gradiente oculto y salida    
        dCdW, _ = ut.gradW_ae(a, x, W, error)
        # Actualizar los pesos
        W, V = ut.updW_ae_sgd(W, V, dCdW, mu)
    
    return (W,V)

    return(W,V)

#Training AE by use SGD
def train_ae(x,hn,param):
    print(x.shape, hn)    
    W,V    = ut.iniWs(x.shape[0],hn)
    for Iter in range(1,param[1]):        
        xe  = x[:,np.random.permutation(x.shape[1])]        
        W,V = train_batch(xe,W,V,param)                
    return W

#SAE's Training 
def train_sae(x,param):    
    W = []
    data = x
    for hn in range(3,len(param)):
        print('AE={} Hnode={}'.format(hn-2,param[hn]))
        # Se entrena el encoder
        w, _ = train_ae(data, param[hn], param)
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

