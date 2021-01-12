import numpy as np
from numpy.random import RandomState
import os
import time
import torch
import uuid

def get_device(silent=False):
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        if not silent:            
            print('GPU device name:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        if not silent:
            print('No GPU available, using the CPU instead.')        
    return device

#torch model
class MyLinearModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim, loss_fn, optimizer=None, 
                 default_lr=None, init_seed=None, n_epochs=4, 
                 batch_size=None, shuffle_seed=None, silent=False, 
                 shuffle=False, device=None):
        super().__init__()
        if not device: 
            self.device = get_device(silent=True)
        else:
            self.device = device
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.silent = silent
        self.loss_fn = loss_fn
        self.n_epochs = n_epochs
        self.model = torch.nn.Linear(in_dim, out_dim)
        if init_seed: 
            torch.manual_seed(init_seed)        
            #initialize random weights
            torch.nn.init.uniform_(self.model.weight, a=-1, b=1)
        if optimizer:
            self.optimizer = optimizer(self.model.parameters())
        else:
            if default_lr:
                self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                                  lr=default_lr)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters())

    def forward(self, X):
        return self.model(X)

    def fit(self, X_train, Y_train, X_val, Y_val):              
        #get tensors
        X_train = torch.from_numpy(X_train.astype(np.float32))
        Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
        X_val = torch.from_numpy(X_val.astype(np.float32))
        Y_val = torch.tensor(Y_val, dtype=torch.float32).reshape(-1, 1)
        
        train_len = X_train.shape[0]        
        rng = RandomState(self.shuffle_seed)        
        if not self.batch_size:        
            self.batch_size = train_len
            n_batches = 1
        else:
            n_batches = int(train_len/self.batch_size)+1            
        #send model and validation data to device        
        self.model = self.model.to(self.device) 
        X_val_ = X_val.to(self.device) 
        Y_val_ = Y_val.to(self.device)
        
        train_losses = []
        val_losses = []
        val_loss_value=float('inf') 
        best_val_loss=float('inf')     
        n_val_drops=0   
        MAX_VAL_DROPS=20
        loss_margin = 1e-3      
        tmp_model_fname = str(uuid.uuid4())+".pt"
        if not self.silent: print("[tmp: {}]".format(tmp_model_fname))
        #placeholders for shuffled data
        X_train_shuff = X_train
        Y_train_shuff = Y_train
        
        for it in range(self.n_epochs):    
            t0_epoch = time.time()
            if self.shuffle:                                     
                idx = torch.tensor(rng.permutation(train_len))
                X_train_shuff = X_train[idx]
                Y_train_shuff = Y_train[idx]                
            for j in range(n_batches):               
                #get batches 
                x_train = X_train_shuff[j*self.batch_size:(j+1)*self.batch_size, :]
                y_train = Y_train_shuff[j*self.batch_size:(j+1)*self.batch_size]                
                x_train_ = x_train.to(self.device)              
                y_train_ = y_train.to(self.device)
                #forwad prop
                y_hat_train = self.forward(x_train_)                
                #losses
                train_loss = self.loss_fn(y_hat_train, y_train_)                
                train_loss_value = train_loss.item()
                train_losses.append(train_loss_value)        
                val_losses.append(val_loss_value)   
                #backprop
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()                
            #validation            
            outputs_val = self.forward(X_val_)
            val_loss = self.loss_fn(outputs_val, Y_val_)      
            val_loss_value =  val_loss.item()     
            if val_loss_value < best_val_loss:    
                n_val_drops=0            
                best_val_loss = val_loss
                #save best model
                # print("[updating best model]")
                torch.save(self.model.state_dict(), tmp_model_fname)
            elif val_loss_value > (best_val_loss - loss_margin):                
                n_val_drops+=1
                if n_val_drops == MAX_VAL_DROPS:
                    print("[early stopping: {} epochs]".format(it))
                    break
            if (it + 1) % 10 == 0 and not self.silent:
                time_elapsed = time.time() - t0_epoch
                print(f'[Epoch {it+1}/{self.n_epochs} | Training loss: {train_loss_value:.4f} | Val loss: {val_loss_value:.4f} | ET: {time_elapsed:.2f}]')
        self.model.load_state_dict(torch.load(tmp_model_fname))
        os.remove(tmp_model_fname)
        # self.model = self.model.cpu()
        return train_losses, val_losses  

    def predict_proba(self, X):        
        X = torch.from_numpy(X.astype(np.float32))
        X_ = X.to(self.device)        
        self.model = self.model.to(self.device) 

        with torch.no_grad():
            y_hat_prob = torch.sigmoid(self.forward(X_))
            y_hat_prob =  y_hat_prob.cpu().numpy()
        return y_hat_prob

    def predict(self, X):        
        y_hat_prob = self.predict_proba(X)
        threshold = 0.5 
        y_hat = (y_hat_prob > threshold)
        return y_hat

class MultiBERTSeq(MyLinearModel):
    def __init__(self, in_dim, out_dim, loss_fn, C=2, optimizer=None, 
                 default_lr=None, init_seed=None, n_epochs=4, 
                 batch_size=None, shuffle_seed=None, silent=False, 
                 shuffle=False, device=None):

        super().__init__(in_dim, out_dim, loss_fn, optimizer, 
                 default_lr, init_seed, n_epochs, 
                 batch_size, shuffle_seed, silent, 
                 shuffle, device)
        self.C = C
    
    def forward(self, X, pad=None, n_seqs=None):        
        Z = X.shape[1]        
        X_ = X.to(self.device)
        #prediction
        Y_tilde = torch.sigmoid(super().forward(X_))
        Y_tilde = Y_tilde.squeeze()
        if pad is None:
            pad, n_seqs = self.get_padding(X)
        #matrix with the number of sequences of each instance
        n_seqs_ = n_seqs.to(self.device)
        #padding matrix to mask unused sequence slots
        pad_ = pad.to(self.device)        
        #pad unused sequence slots
        Y_tilde = Y_tilde*pad_
        #(y_max+y_mean*Z/C) / 1+Z/C
        scaling = Z/self.C
        Y_max = torch.max(Y_tilde, dim=1)[0]
        Y_mean = torch.sum(Y_tilde, dim=1)/n_seqs_        
        Y_hat = (Y_max+Y_mean*scaling)/(1+scaling)        
        Y_hat = Y_hat.reshape(-1,1)
        return Y_hat

    def predict_proba(self, X):        
        X = torch.from_numpy(X.astype(np.float32))
        X_ = X.to(self.device)        
        self.model = self.model.to(self.device) 

        with torch.no_grad():
            y_hat_prob = self.forward(X_)
            y_hat_prob =  y_hat_prob.cpu().numpy()
        return y_hat_prob
    
    def get_padding(self, X):
        """ compute a padding mask to zero out unused sequences """
        Z = X.shape[1]     
        N = X.shape[0]    
        n_seqs = []        
        mask = torch.ones([N,Z])
        for n in range(N):
            #torch.where(torch.all(X_feats_[n, ...] == 0, axis=1) gives a list of indices where the columns are all zeroes 
            z = torch.where(torch.all(X[n, ...] == 0, axis=1))[0]                        
            mask[n, z] = 0
            #storing the sequence len so that we can compute the mean
            #the size of the list gives us the number of zeroed columns 
            n_seqs.append(Z-len(z))            
        n_seqs = torch.tensor(n_seqs, dtype=torch.long)
        return mask, n_seqs

    def fit(self, X_train, Y_train, X_val, Y_val):              
        #get tensors
        X_train = torch.from_numpy(X_train.astype(np.float32))
        Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
        X_val = torch.from_numpy(X_val.astype(np.float32))
        Y_val = torch.tensor(Y_val, dtype=torch.float32).reshape(-1, 1)
        #get padding masks
        train_pad, train_n_seqs = self.get_padding(X_train)        
        val_pad, val_n_seqs = self.get_padding(X_val)        
        
        train_len = X_train.shape[0]        
        rng = RandomState(self.shuffle_seed)        
        if not self.batch_size:        
            self.batch_size = train_len
            n_batches = 1
        else:
            n_batches = int(train_len/self.batch_size)+1            
        #send model and validation data to device        
        self.model = self.model.to(self.device) 
        X_val_ = X_val.to(self.device) 
        Y_val_ = Y_val.to(self.device)

        train_losses = []
        val_losses = []
        val_loss_value=float('inf') 
        best_val_loss=float('inf')     
        n_val_drops=0   
        MAX_VAL_DROPS=20
        loss_margin = 1e-3      
        tmp_model_fname = str(uuid.uuid4())+".pt"
        if not self.silent: print("[tmp: {}]".format(tmp_model_fname))
        #placeholders for shuffled data
        X_train_shuff = X_train
        Y_train_shuff = Y_train
        train_pad_shuff = train_pad
        train_n_seqs_shuff = train_n_seqs
        for it in range(self.n_epochs):    
            t0_epoch = time.time()
            if self.shuffle:                                     
                idx = torch.tensor(rng.permutation(train_len))
                X_train_shuff = X_train[idx]
                Y_train_shuff = Y_train[idx]
                train_pad_shuff = train_pad[idx]
                train_n_seqs_shuff = train_n_seqs[idx]
            for j in range(n_batches):               
                #get batches 
                x_train = X_train_shuff[j*self.batch_size:(j+1)*self.batch_size, :]
                y_train = Y_train_shuff[j*self.batch_size:(j+1)*self.batch_size]
                pad = train_pad_shuff[j*self.batch_size:(j+1)*self.batch_size]
                n_seqs = train_n_seqs_shuff[j*self.batch_size:(j+1)*self.batch_size]
                x_train_ = x_train.to(self.device)              
                y_train_ = y_train.to(self.device)
                #forwad prop
                y_hat_train = self.forward(x_train_, pad, n_seqs)                
                #losses
                train_loss = self.loss_fn(y_hat_train, y_train_)                
                train_loss_value = train_loss.item()
                train_losses.append(train_loss_value)        
                val_losses.append(val_loss_value)   
                #backprop
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()                
            #validation            
            outputs_val = self.forward(X_val_, val_pad, val_n_seqs)
            val_loss = self.loss_fn(outputs_val, Y_val_)      
            val_loss_value =  val_loss.item()     
            if val_loss_value < best_val_loss:    
                n_val_drops=0            
                best_val_loss = val_loss
                #save best model
                # print("[updating best model]")
                torch.save(self.model.state_dict(), tmp_model_fname)
            elif val_loss_value > (best_val_loss - loss_margin):                
                n_val_drops+=1
                if n_val_drops == MAX_VAL_DROPS:
                    print("[early stopping: {} epochs]".format(it))
                    break
            if (it + 1) % 10 == 0 and not self.silent:
                time_elapsed = time.time() - t0_epoch
                print(f'[Epoch {it+1}/{self.n_epochs} | Training loss: {train_loss_value:.4f} | Val loss: {val_loss_value:.4f} | ET: {time_elapsed:.2f}]')
        self.model.load_state_dict(torch.load(tmp_model_fname))
        os.remove(tmp_model_fname)
        # self.model = self.model.cpu()
        return train_losses, val_losses  


class LinearDataMaps(MyLinearModel):
    def __init__(self, in_dim, out_dim, loss_fn, optimizer=None, 
                 default_lr=None, init_seed=None, n_epochs=4, 
                 batch_size=None, shuffle_seed=None, silent=False, 
                 shuffle=False, device=None):
        super().__init__(in_dim, out_dim, loss_fn, optimizer, 
                 default_lr, init_seed, n_epochs, 
                 batch_size, shuffle_seed, silent, 
                 shuffle, device)
    
    def fit(self, X_train, Y_train, X_val, Y_val):      
        X_train = torch.from_numpy(X_train.astype(np.float32))
        Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
        X_val = torch.from_numpy(X_val.astype(np.float32))
        Y_val = torch.tensor(Y_val, dtype=torch.float32).reshape(-1, 1)

        train_len = X_train.shape[0]        
        rng = RandomState(self.shuffle_seed)        
        if not self.batch_size:        
            self.batch_size = train_len
            n_batches = 1
        else:
            n_batches = int(train_len/self.batch_size)+1            
        #send validation data and model to device
        X_val_ = X_val.to(self.device) 
        Y_val_ = Y_val.to(self.device)
        X_train_ = X_train.to(self.device)
        Y_train_ = Y_train.to(self.device)
        self.model = self.model.to(self.device) 
        idx = torch.tensor(rng.permutation(train_len))
        idx_ = idx.to(self.device) 
        train_preds = torch.zeros(X_train.shape[0], self.n_epochs)
        train_preds = train_preds.to(self.device)
        train_losses = []
        val_losses = []
        val_loss_value=float('inf') 
        best_val_loss=float('inf')     
        n_val_drops=0   
        MAX_VAL_DROPS=20
        loss_margin = 1e-3      
        tmp_model_fname = str(uuid.uuid4())+".pt"
        if not self.silent: print("[tmp: {}]".format(tmp_model_fname))
        for it in range(self.n_epochs):    
            t0_epoch = time.time()
            if self.shuffle:                                     
                X_train_ = X_train[idx_].to(self.device)
                Y_train_ = Y_train[idx_].to(self.device)                        
                idx = torch.tensor(rng.permutation(train_len))
                idx_ = idx.to(self.device) 
            for j in range(n_batches):                
                x_train = X_train_[j*self.batch_size:(j+1)*self.batch_size, :]
                y_train = Y_train_[j*self.batch_size:(j+1)*self.batch_size]                
                y_hat_train = self.forward(x_train)
                # print(y_hat_train)
                train_loss = self.loss_fn(y_hat_train, y_train)                
                train_loss_value = train_loss.item()
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()                
                train_losses.append(train_loss_value)        
                val_losses.append(val_loss_value)
            train_preds[:,it] = self.forward(X_train_).squeeze()  
            outputs_val = self.forward(X_val_)
            val_loss = self.loss_fn(outputs_val, Y_val_)      
            val_loss_value =  val_loss.item()     
            if val_loss_value < best_val_loss:    
                n_val_drops=0            
                best_val_loss = val_loss
                #save best model
                # print("[updating best model]")
                torch.save(self.model.state_dict(), tmp_model_fname)
            elif val_loss_value > (best_val_loss - loss_margin):                
                n_val_drops+=1
                # if n_val_drops == MAX_VAL_DROPS:
                    # print("[early stopping: {} epochs]".format(it))
                    # break
            if (it + 1) % 10 == 0 and not self.silent:
                time_elapsed = time.time() - t0_epoch
                print(f'[Epoch {it+1}/{self.n_epochs} | Training loss: {train_loss_value:.4f} | Val loss: {val_loss_value:.4f} | ET: {time_elapsed:.2f}]')
        self.model.load_state_dict(torch.load(tmp_model_fname))
        os.remove(tmp_model_fname)
        train_preds = torch.sigmoid(train_preds)
        train_preds = train_preds.detach().numpy()
        # self.model = self.model.cpu()
        return train_losses, val_losses, train_preds 