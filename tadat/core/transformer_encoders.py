import numpy as np
import time
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import sys
import random 
import os
import uuid 

DEFAULT_BERT_MODEL = 'bert-base-uncased'
BERT_MAX_INPUT=512
DEFAULT_BATCH_SIZE = 64

####################################################################
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

def get_BERT(pretrained_model=DEFAULT_BERT_MODEL, hidden_states=False):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(pretrained_model, output_hidden_states=hidden_states)
    return tokenizer, model

def encode_sequences(docs, tokenizer=None, encoder=None, cls_features=False, 
                    batchsize=DEFAULT_BATCH_SIZE, device=None):  
    feature_vectors = []    
    n_batches = int(len(docs)/batchsize)+1
    for j in range(n_batches):
        batch = docs[batchsize*j:batchsize*(j+1)]
        if len(batch) > 0:
            sys.stdout.write("\nbatch:{}/{} (size: {})".format(j+1,n_batches, str(len(batch))))
            sys.stdout.flush()
            cls_vec, pool_vec = sequence_features(batch,tokenizer, encoder,device)            
            feats = cls_vec if cls_features else pool_vec                       
            feature_vectors.append(feats)                
    feature_vectors = np.vstack(feature_vectors)

    return feature_vectors

def sequence_features(docs, tokenizer=None, encoder=None, device=None):    
    #BERT
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained(DEFAULT_BERT_MODEL)
    if not encoder:
        encoder = BertModel.from_pretrained(DEFAULT_BERT_MODEL, output_hidden_states=False)
    if not device:
        device = __get_device()

    docs_tensor = []    
    tokenized_docs = []    
    bertify = "[CLS] {} [SEP]"  
    tokenized_docs = [tokenizer.tokenize(bertify.format(doc)) for doc in docs]     
    #count the document lengths  
    max_len = max([len(d) for d in tokenized_docs]) 
    #document cannot exceed BERT input matrix size 
    max_len = min(BERT_MAX_INPUT, max_len)
    print("[max len: {}]".format(max_len))
    for doc in tokenized_docs:   
        # Convert tokens to vocabulary indices
        idxs = tokenizer.convert_tokens_to_ids(doc)        
        #truncate sentences longer than what BERT supports
        if len(idxs) > max_len: idxs = idxs[:max_len]
        pad_size = max_len - len(idxs)
        #add padding to indexed tokens
        idxs+=[0] * pad_size        
        docs_tensor.append(torch.tensor([idxs]))        
    
    # Convert inputs to PyTorch tensors
    docs_tensor = torch.cat(docs_tensor)
    segments_tensor = torch.zeros_like(docs_tensor)            
    #set encoder to eval mode
    encoder.eval()
    #device    
    docs_tensor = docs_tensor.to(device)
    segments_tensor =  segments_tensor.to(device)
    encoder = encoder.to(device)
    with torch.no_grad():        
        pool_features, cls_features = encoder(docs_tensor, 
                                            token_type_ids=segments_tensor)    
        pool_features = pool_features.sum(axis=1)
    return cls_features.cpu().numpy(), pool_features.cpu().numpy()

def encode_multi_sequences(docs, max_sequences, tokenizer=None, encoder=None,
                            cls_features=False, batchsize=DEFAULT_BATCH_SIZE, 
                            tmp_path=None, device=None):  
    tmp_fname = None      
    if tmp_path:        
        dirname = os.path.dirname(tmp_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        tmp_fname = tmp_path+"/"+str(uuid.uuid4())
        print("[saving @ {}]".format(tmp_fname))       
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained(DEFAULT_BERT_MODEL)
    if not encoder:
        encoder = BertModel.from_pretrained(DEFAULT_BERT_MODEL, output_hidden_states=False)
    if not device:
        device = __get_device()
    vectors = []      
    n_batches = int(len(docs)/batchsize)+1
    for j in range(n_batches):
        batch = docs[batchsize*j:batchsize*(j+1)]
        if len(batch) > 0:
            sys.stdout.write("\nbatch:{}/{} (size: {})".format(j+1,n_batches, str(len(batch))))
            sys.stdout.flush()
            cls_vec, pool_vec = multi_sequence_features(batch, max_sequences,tokenizer, encoder,device)                        
            feats = cls_vec if cls_features else pool_vec                       
            if tmp_fname:                    
                #save partial features     
                np.save(tmp_fname+"_"+str(j), feats)
            else:
                vectors.append(feats)                
    #combine all partial features
    if tmp_fname:
        print("[reconstructing features]")
        for j in range(n_batches):
            with open(tmp_fname+"_{}.npy".format(j), 'rb') as f:                            
                x = np.load(f)
                vectors.append(x)                
            #remove temporary file
            # os.remove(tmp_fname+"_{}.npy".format(j))
        
    vectors = np.vstack(vectors)        
    return vectors

def multi_sequence_features(docs, max_sequences, tokenizer, encoder, device):    
    docs_tensor = []    
    #keep track of the number of sequences of each document
    doc_n_sequences = []        
    #tokenize documents
    tokenized_docs = [tokenizer.tokenize(doc) for doc in docs]     
    for doc in tokenized_docs:           
        sequence_tensor = torch.zeros([max_sequences, BERT_MAX_INPUT], dtype=torch.long)        
        # split doc into sequences of size BERT_MAX_INPUT
        n_seqs = int(len(doc)/BERT_MAX_INPUT)+1        
        n_seqs = min(n_seqs, max_sequences)
        doc_n_sequences.append(n_seqs)        
        for i in range(n_seqs):
            #using (BERT_MAX_INPUT-2) to leave space for [CLS]/[SEP] tokens
            seq = doc[i*(BERT_MAX_INPUT-2):(i+1)*(BERT_MAX_INPUT-2)]
            seq = ["[CLS]"] + seq + ["[SEP]"]            
            # Convert tokens to vocabulary indices
            idxs = tokenizer.convert_tokens_to_ids(seq)        
            #add padding to indexed tokens
            pad_size = BERT_MAX_INPUT - len(idxs)            
            idxs+=[0] * pad_size            
            sequence_tensor[i, :] = torch.tensor([idxs])            
        docs_tensor.append(sequence_tensor)         
    
    docs_tensor = torch.stack(docs_tensor)
    doc_n_sequences = torch.Tensor(doc_n_sequences)
    #we are only encoding a single sentence so segment ids are all 0
    segment_ids_tensor = torch.zeros_like(docs_tensor) 
    #send data to device
    docs_tensor = docs_tensor.to(device)
    doc_n_sequences = doc_n_sequences.to(device)
    segment_ids_tensor =  segment_ids_tensor.to(device)
    encoder = encoder.to(device)
    #set encoder to eval mode
    encoder.eval()            
    pool_features = []
    cls_features = []
    with torch.no_grad(): 
        for j in range(docs_tensor.shape[0]):
            doc = docs_tensor[j,:,:]    
            segment_ids = segment_ids_tensor[j,:,:]    
            pool_x, cls_x = encoder(doc, token_type_ids=segment_ids)    
            pool_x = pool_x.sum(axis=1)            
            n_seqs = torch.tensor(doc_n_sequences[j], dtype=torch.long, device=device)            
            #zero out (pad) unused sequences  
            pool_x[n_seqs:, :] = 0
            cls_x[n_seqs:, :] = 0
            pool_features.append(pool_x)
            cls_features.append(cls_x)
    pool_features = torch.stack(pool_features).cpu().numpy()
    cls_features = torch.stack(cls_features).cpu().numpy()
    return pool_features, cls_features
    # return cls_features.cpu().numpy(), pool_features.cpu().numpy()