#torch model
class MyLinearModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim, loss_fn, optimizer=None, 
                 default_lr=None, init_seed=None, n_epochs=4, 
                 batch_size=None, shuffle_seed=None, silent=False, 
                 shuffle=False, device=None):
        super().__init__()
        if not device: self.device = get_device(silent=True)
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

    def forward(self, in_dim, out_dim):
        return self.model(in_dim, out_dim)

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

        train_losses = []
        val_losses = []
        val_loss_value=float('inf') 
        best_val_loss=float('inf')     
        n_val_drops=0   
        MAX_VAL_DROPS=10
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
                y_hat_train = self.model(x_train)
                train_loss = self.loss_fn(y_hat_train, y_train)                
                train_loss_value = train_loss.item()
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()                
                train_losses.append(train_loss_value)        
                val_losses.append(val_loss_value)   
            outputs_val = self.model(X_val_)
            val_loss = self.loss_fn(outputs_val, Y_val_)      
            val_loss_value =  val_loss.item()     
            if val_loss_value < best_val_loss:    
                n_val_drops=0            
                best_val_loss = val_loss
                #save best model
                # print("[updating best model]")
                torch.save(self.model.state_dict(), tmp_model_fname)
            elif val_loss_value > best_val_loss - loss_margin:                
                n_val_drops+=1
                # if n_val_drops == MAX_VAL_DROPS:
                #     print("[early stopping: {} epochs]".format(it))
                    # break
            if (it + 1) % 50 == 0 and not self.silent:
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
            y_hat_prob = torch.nn.functional.sigmoid(self.model(X_))
            y_hat_prob =  y_hat_prob.cpu().numpy()
        return y_hat_prob

    def predict(self, X):        
        y_hat_prob = self.predict_proba(X)
        threshold = 0.5 
        y_hat = (y_hat_prob > threshold)
        return y_hat

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