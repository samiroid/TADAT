#import codecs
from tadat.core.vectorizer import idx2word
import numpy as np
from sklearn.decomposition import PCA
import os
import sys
import torch
from transformers import BertTokenizer, BertModel
# from tsne import bh_sne    


def old_embeddings_to_dict(path, max_words=None):
    """
        Read word embeddings into a dictionary
    """
    w2v = {}    
    with open(path,"r") as fid:
        #ignore first line
        fid.readline()                
        #avoid extra comparisons if we want load all the words
        if max_words is None:
            for line in fid:
                entry = line.split()
                if len(entry) > 2:
                    w2v[entry[0]] = np.array(entry[1:]).astype('float32')
        else:
            for i, line in enumerate(fid):
                entry = line.split()
                if len(entry) > 2:
                    w2v[entry[0]] = np.array(entry[1:]).astype('float32')
                if i >= max_words:break
    return w2v 

def old_read_embeddings(path, wrd2idx=None, max_words=None):

    w2v = old_embeddings_to_dict(path,max_words)        
    #if no word index is specified read all the embedding vocabulary
    if wrd2idx is None:
        wrd2idx = {w:i for i,w in enumerate(w2v.keys())}

    common_vocab = set(w2v).intersection(set(wrd2idx))    
    #build embedding matrix
    emb_size = list(w2v.values())[0].shape[0]    
    E = np.zeros((emb_size, len(wrd2idx)))    
    for w,i in wrd2idx.items():
        if w in w2v: E[:,i] = w2v[w]            
    # Number of out of embedding vocabulary embeddings

    ooevs = len(wrd2idx)-len(common_vocab)
    perc = ooevs *100./len(wrd2idx)
    print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" % (ooevs, len(wrd2idx), perc))     

    return E, wrd2idx

def embeddings_to_dict(path, vocab, encoding="utf-8"):
    """
        Read word embeddings into a dictionary
    """    
    w2v = {}    
    with open(path,"r",encoding=encoding) as fid:
        #ignore first line
        fid.readline()                
        #avoid extra comparisons if we want load all the words        
        for line in fid:
            entry = line.split()
            if len(entry) > 2 and entry[0] in vocab:
                try:
                    w2v[entry[0]] = np.array(entry[1:]).astype('float32')
                except ValueError:
                    print("could not parse line {}".format(line))
                    continue        
    return w2v 

def read_embeddings(path, vocab, encoding):    
    w2v = embeddings_to_dict(path,vocab,encoding)        
    common_vocab = set(w2v).intersection(set(vocab))    
    #build embedding matrix
    emb_size = list(w2v.values())[0].shape[0]    
    E = np.zeros((emb_size, len(vocab)))    
    for w,i in vocab.items():
        if w in w2v: E[:,i] = w2v[w]            
    # Number of out of embedding vocabulary embeddings
    ooevs = len(vocab)-len(common_vocab)
    perc = ooevs *100./len(vocab)
    print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" % (ooevs, len(vocab), perc))     

    return E, vocab


def get_OOEVs(E, wrd2idx):

    ooev_idx = np.where(~E.any(axis=0))[0]
    idx2wrd = idx2word(wrd2idx)
    OOEVs = [idx2wrd[idx] for idx in ooev_idx]

    return OOEVs

def extract_embeddings(path_in, path_out, vocab):

    """
        Filter embeddings file to contain only the relevant set
        of words (so that it can be loaded faster)
    """

    w2v = embeddings_to_dict(path_in, vocab)  
    common_vocab = set(w2v).intersection(set(vocab))
    directory = os.path.dirname(path_out)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path_out,"w") as fod:
        voc_size = len(common_vocab)
        emb_size = list(w2v.values())[0].shape[0]   
        fod.write(str(voc_size)+"\t"+str(emb_size)+"\n")
        for word in common_vocab:
            emb = w2v[word]                
            fod.write(u"%s %s\n" % (word, " ".join(map(str, emb))))
        ooevs = len(vocab)-len(common_vocab)
        perc = ooevs *100./len(vocab)
        print ("%d/%d (%2.2f %%) words in vocabulary found no embedding" % (ooevs, len(vocab), perc))     

def save_txt(path, E, wrd2idx):
    with open(path,"w") as fod:
        fod.write(u"%d %d\n" % (E.shape[1],E.shape[0]))  
        for word, idx in wrd2idx.items():      
            emb = E[:,idx]
            fod.write(u"%s %s\n" % (word, " ".join(map(str, emb))))


# def project_vectors(X_in, model='tsne', perp=10, n_components=2):    
#     if model == 'tsne':        
#         X_in = X_in.reshape((X_in.shape[0], -1)).astype('float64')
#         if perp is not None:
#             X_out = bh_sne(X_in, perplexity=perp)    
#         else:
#             X_out = bh_sne(X_in)    
#     elif model == 'pca':
#         pca = PCA(n_components=n_components, whiten=True)
#         pca.fit(X_in)
#         X_out = pca.transform(X_in)        
#     else:
#         raise NotImplementedError
    
#     return X_out

def similarity_rank(X, wrd2idx,top_k=None):        

    items = wrd2idx.keys()#[:max_users]
    idxs  = wrd2idx.values()#[:max_users]
    if top_k is None:
        top_k = len(idxs)
    item_ranking = np.zeros((top_k,len(idxs)))
    sim_scores = np.zeros((top_k,len(idxs)))
    
    for i, u in enumerate(idxs):
        emb = X[:,u]
        #similarities
        simz = np.dot(X.T,emb)/(np.linalg.norm(X.T)*np.linalg.norm(emb))
        #user maximally similar to itself
        simz[u] = 1
        rank = np.argsort(simz)[::-1]
        ranked_simz = simz[rank]
        item_ranking[:,i] = rank[:top_k]
        sim_scores[:,i]   = ranked_simz[:top_k]

    return items, idxs, item_ranking, sim_scores
