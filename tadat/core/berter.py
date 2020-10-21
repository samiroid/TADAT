import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import regex as re
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import itertools

def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def max_doc_len(docs, tokenizer=None):
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    return max([len(tokenizer.encode(d, add_special_tokens=True, max_length=512, truncation=True)) for d in docs])

def vectorize(data, max_len=512, tokenizer=None):    
    if not tokenizer: 
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = []
    attention_masks = []
    for sent in data:        
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=max_len,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length            
            return_attention_mask=True,      # Return attention mask
            truncation=True)
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)    
    return input_ids, attention_masks

def encode(inputs, masks, batch_size, encoder=None, pooling=True):
    if not encoder:
        encoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=False)
    #load data
    data = TensorDataset(inputs, masks)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)    
    features = []
    device = __get_device()
    for batch in dataloader:        
        # Load batch to GPU
        input_ids, attn_mask = tuple(t.to(device) for t in batch)
        pool_features, cls_features = encoder(input_ids=input_ids,
                                            attention_mask=attn_mask)        
        
        if pooling:
            X = pool_features.sum(axis=1)
        else:
            X = cls_features
        X = X.cpu().numpy()
        features.append(X)
    features = np.vstack(features)
    return features


def __get_device(silent=True):
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        if not silent:            
            print('GPU device name:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        if not silent:
            print('No GPU available, using the CPU instead.')        
    return device

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4,lr=5e-5, eps=1e-8,num_warmup_steps=0, evaluation=False):
    """Train the BertClassifier model.
    """
    # Specify loss function
    loss_fn = nn.CrossEntropyLoss()
    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=lr,    
                      eps=eps   
                      )
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    device = __get_device()
    model.to(device)
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # Put the model into the training mode
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            # Zero out any previously calculated gradients
            model.zero_grad()
            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)
            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)            
            batch_loss += loss.item()
            total_loss += loss.item()
            # Perform a backward pass to calculate gradients
            loss.backward()
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        val_accuracy = 0
        if evaluation:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, loss_fn, val_dataloader)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")    
    print("Training complete!")
    return model, val_accuracy


def evaluate(model, loss_fn, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    # Tracking variables
    val_accuracy = []
    val_loss = []
    device = __get_device()
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())
        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy

def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    return probs

def bert_classify(model, test_dataloader):        
    probs = bert_predict(model, test_dataloader)
    Y = np.argmax(probs, axis=1)
    return Y

def get_random_sample_loader(inputs, masks, labels, batch_size):
    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

def get_sequential_sample_loader(inputs, masks, labels, batch_size):
    data = TensorDataset(inputs, masks, labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader

def hyperparameter_search(train_data, val_data, freeze_bert, pool, n_confs=-1, hyperparams=None):
  #n_confs can be used to sample a fixed number of (random) hyperparameter configurations
  #n_confs=-1 will try all configurations
  
    results = []
    if not hyperparams:
        hyperparams = { "lr":[5e-6,5e-5,1e-4],
                    "num_warmup_steps":[0,1,2]
            }  
    confs = list(itertools.product(hyperparams["lr"],hyperparams["num_warmup_steps"]))
    #shuffle confs
    random.shuffle(confs)
    print(confs)
    if n_confs > -1:
        confs = confs[:n_confs]
    for conf in confs:
        print("Testing conf: {}".format(repr(conf)))
        lr = conf[0]
        num_warmup_steps = conf[1]
        bert_classifier = BertClassifier(freeze_bert=freeze_bert, pool=pool)
        val_acc = bert_classifier.fit(train_data, val_data, epochs=2, validation=True)
        results.append([lr, num_warmup_steps, val_acc])
    return results

# Create the BertClassfier class
class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False, pool=True):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2
        if pool:            
            print("BERT > POOl features")
        else:
            print("BERT > CLS features")
        self.pool = pool
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=False)

        # Instantiate an one-layer feed-forward classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(D_in, H),
        #     nn.ReLU(),
        #     #nn.Dropout(0.5),
        #     nn.Linear(H, D_out)
        # )

        self.classifier = nn.Sequential(           
            nn.ReLU(),            
            nn.Linear(D_in, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        pool_features, cls_features = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Feed input to classifier to compute logits
        if self.pool:
            logits = self.classifier(pool_features.sum(axis=1))
        else:
            logits = self.classifier(cls_features)
        return logits
    
    def fit(self, train_dataloader, val_dataloader, epochs, validation=True, seed=1):
        set_seed(seed)
        _, val_acc = train(self, train_dataloader, val_dataloader, epochs=epochs, evaluation=validation)
        return round(val_acc, 3)

    def predict_proba(self, test_dataloader):
        return bert_predict(self, test_dataloader)

    def predict(self, test_dataloader):
        probs = bert_predict(self, test_dataloader)
        preds = probs[:, 1]    
        y_hat = np.where(preds >= 0.5, 1, 0)
        return y_hat
