from collections import Counter, defaultdict
import numpy as np

def get_words_vocab(docs, max_words=None, min_freq=1, include_pad_token=False):
    """
        Compute a dictionary index mapping words into indices
    """

    words = [w for m in docs for w in m.split()]
    cnt = Counter(words)
    word_cnts = [(w,c) for (w,c) in sorted(cnt.items(), key=lambda x:x[1],reverse=True) if c>min_freq]
    #keep only the 'max_words' most frequent tokens
    if max_words:
        word_cnts = word_cnts[:max_words]
    #get the words
    words = [w[0] for w in word_cnts]       
    #prepend the padding token
    if include_pad_token: words = ['_PAD_'] + words
    return __get_vocab(words)
    
def get_labels_vocab(labels):
    """
        labels: list of labels as strings 
    """
    uniq_labels = list(set(labels))
    return __get_vocab(uniq_labels)    

def __get_vocab(items):
    vocab = {l:i for i,l in enumerate(items) }
    return vocab 

def one_hot(index, items):
    oh = np.zeros((len(items),len(index))) 
    for i,item in enumerate(items):
        oh[i,index[item]] = 1
    return oh

def idx2word(wrd2idx):
    return {i:w for w, i in wrd2idx.items()}

def label2idx(labels, vocab=None):
    if not vocab:
        vocab = get_labels_vocab(labels)
    return [vocab[y] for y in labels ], vocab
    
def docs2idx(docs, vocab=None, max_words=None, min_freq=1):
    """
        Convert documents to lists of word indices
    """
    if vocab is None:
        vocab = get_words_vocab(docs,max_words=max_words,min_freq=min_freq)    
    X = [[vocab[w] for w in m.split() if w in vocab] for m in docs]
    return X, vocab
