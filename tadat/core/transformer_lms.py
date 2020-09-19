import numpy as np
import time
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import sys
import random 

DEFAULT_BERT_MODEL = 'bert-base-uncased'
BERT_MAX_INPUT=510
DEFAULT_BATCH_SIZE = 100

####################################################################

def get_BERT(pretrained_model=DEFAULT_BERT_MODEL, hidden_states=False):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(pretrained_model, output_hidden_states=hidden_states)
    return tokenizer, model

def transformer_encode_batches(docs, tokenizer=None, model=None, batchsize=DEFAULT_BATCH_SIZE, device='cpu'):    
    #BERT
    if not tokenizer:
        tokenizer = BertTokenizer.from_pretrained(DEFAULT_BERT_MODEL)
    if not model:
        model = BertModel.from_pretrained(DEFAULT_BERT_MODEL, output_hidden_states=False)
    pool_vectors = []
    cls_vectors = []    
    n_batches = int(len(docs)/batchsize)+1
    for j in range(n_batches):
        batch = docs[batchsize*j:batchsize*(j+1)]
        if len(batch) > 0:
            sys.stdout.write("\nbatch:{}/{} (size: {})".format(j+1,n_batches, str(len(batch))))
            sys.stdout.flush()
            cls_vec, pool_vec = transformer_encode(batch,tokenizer, model,device)            
            cls_vectors.append(cls_vec)
            pool_vectors.append(pool_vec)            
    cls_vectors = np.vstack(cls_vectors)
    pool_vectors = np.vstack(pool_vectors)
        
    return cls_vectors, pool_vectors

def transformer_encode(docs, tokenizer, encoder, device):
    tokens_tensors = []
    segments_tensors = []
    tokenized_texts = []    
    bertify = "[CLS] {} [SEP]"  
    tokenized_texts = [tokenizer.tokenize(bertify.format(doc)) for doc in docs]     
    #count the document lengths  
    max_len = max([len(d) for d in tokenized_texts]) 
    #document cannot exceed BERT input matrix size 
    max_len = min(BERT_MAX_INPUT, max_len)
    print("[max len: {}]".format(max_len))
    for tokens in tokenized_texts:   
        # Convert tokens to vocabulary indices
        idxs = tokenizer.convert_tokens_to_ids(tokens)        
        #truncate sentences longer than what BERT supports
        if len(idxs) > BERT_MAX_INPUT: idxs = idxs[:BERT_MAX_INPUT]
        pad_size = max_len - len(idxs)
        #add padding to indexed tokens
        idxs+=[0] * pad_size
        segments_ids = [0] * len(idxs) 
        tokens_tensors.append(torch.tensor([idxs]))
        segments_tensors.append(torch.tensor([segments_ids]))
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.cat(tokens_tensors)
    segments_tensor = torch.cat(segments_tensors)        
    #set encoder to eval mode
    encoder.eval()
    #device
    tokens_tensor = tokens_tensor.to(device)
    segments_tensor =  segments_tensor.to(device)
    encoder = encoder.to(device)
    with torch.no_grad():        
        pool_features, cls_features = encoder(tokens_tensor, token_type_ids=segments_tensor)    
        pool_features = pool_features.sum(axis=1)
    return cls_features.cpu().numpy(), pool_features.cpu().numpy()