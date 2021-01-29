from collections import defaultdict
import numpy as np
import os
import re
import twokenize

try:
    from ipdb import set_trace
except ImportError:
    from pdb import set_trace

def getX(data):
    return [d[1] for d in data]		

def getY(data):
    return [d[0] for d in data]		

def max_repetitions(sentence, n=3):
    """
        Normalizes a string to at most n repetitions of the same character
        e.g, for n=3 and "helllloooooo" -> "helllooo"
    """
    new_sentence = ' '
    reps=0
    for c in sentence:
        if c == new_sentence[-1]:
            reps+=1
            if reps >= n:
                continue
        else:
            reps=0
        new_sentence+=c
    return new_sentence.strip()

def preprocess(d, stop_words=()):       
    d = d.lower()
    # tokens = tokenizer.tokenize(d)     
    tokens = d.split()     
    d = u' '.join([t for t in tokens if t not in set(stop_words)])
    #reduce character repetitions
    d = max_repetitions(d)
    return d

def mask_user_mentions(txt):
     #replace user mentions with token '@user'
    user_regex = r".?@.+?( |$)|<@mention>"    
    txt = re.sub(user_regex," @user ", txt, flags=re.I)
    return txt

def mask_urls(txt):
    #replace urls with token 'url'
    txt = re.sub(twokenize.url," url ", txt, flags=re.I)
    return txt

def flatten_list(l):
    return [item for sublist in l for item in sublist]    

def shuffle_split(data, split_perc=0.8, random_seed=1234):
    """
        Split the data into train and test, keeping the class proportions

        data: list of (y,x) tuples
        split_perc: percentage of training examples in train/test split
        random_seed: ensure repeatable shuffles

        returns: balanced training and test sets
    """
    #shuffle data    
    rng = np.random.RandomState(random_seed)
    rng.shuffle(data)
    #group examples by class label
    z = defaultdict(list)
    for y, x in data: z[y].append(x)
    train = []
    test = []
    for label in z.keys():
        #examples of each label
        x_label = z[label]
        split = int(len(x_label) * split_perc)
        #train split        
        train_Xs = x_label[:split]
        train_Ys = [label] * len(train_Xs)        
        train += zip(train_Ys, train_Xs)
        #test split
        test_Xs = x_label[split:]
        test_Ys = [label] * len(test_Xs)        
        test += zip(test_Ys, test_Xs)
    #reshuffle
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test

def shuffle_split_idx(Y, split_perc = 0.8, random_seed=1234):
    """
        Split the data into train and test, keeping the class proportions

        data: list of labels
        split_perc: percentage of training examples in train/test split
        random_seed: ensure repeatable shuffles

        returns: indices for balanced training and test sets 
    """
    #shuffle data
    rng=np.random.RandomState(random_seed)          
    #group examples by class class    
    z = defaultdict(list)
    for i,y in enumerate(Y): z[y].append(i)    
    train = []    
    test  = []
    for cl in z.keys():
        #indices of the examples of each class 
        idx_cl = z[cl]  
        split   = int(len(idx_cl)*split_perc)
        train += idx_cl[:split]         
        test  += idx_cl[split:]
    #reshuffle
    rng.shuffle(train)
    rng.shuffle(test)    

    return train, test

def stratified_sampling(data, n, random_seed=1234):
    """
        Get a sample of the data, keeping the class proportions

        data: list of (x,y) tuples
        n: number of samples
        random_seed: ensure repeatable shuffles

        returns: balanced sample
    """
    rng=np.random.RandomState(random_seed)          
    z = defaultdict(list)
    #shuffle data
    rng.shuffle(data)
    #group examples by class    
    z = defaultdict(list)    
    for x,y in data: z[y].append(x)    
    #compute class distribution
    class_dist = {}
    for cl, samples in z.items():
        class_dist[cl] = int((len(samples)*1./len(data)) * n)
    sample = []    
    
    for label in z.keys():
        #examples of each label 
        x_label  = z[label]            
        sample += zip(x_label[:class_dist[label]],
                    [label] * class_dist[label])             
    #reshuffle
    rng.shuffle(sample)
    return sample

def simple_split(data, split_perc=0.8, random_seed=1234):
    """
        Split the data into train and test
        data: list of (y,x) tuples
        split_perc: percentage of training examples in train/test split
        random_seed: ensure repeatable shuffles
        returns: train and test splits
    """
    #shuffle data
    rng = np.random.RandomState(random_seed)
    rng.shuffle(data)        
    split = int(len(data) * split_perc)
    #train split
    train_split = data[:split]
    test_split = data[split:]
    
    return train_split, test_split

def kfolds(n_folds, n_elements, val_set=False, shuffle=False, random_seed=1234):   
         
    if val_set: assert n_folds>2    
    X = np.arange(n_elements)
    if shuffle: 
        rng=np.random.RandomState(random_seed)      
        rng.shuffle(X)    
    X = X.tolist()
    slice_size = n_elements/n_folds
    slices =  [X[j*slice_size:(j+1)*slice_size] for j in range(n_folds)]
    #append the remaining elements to the last slice
    slices[-1] += X[n_folds*slice_size:]
    kf = []
    for i in range(len(slices)):
        train = slices[:]     
        test = train.pop(i)
        if val_set:
            #take one of the slices as the development set
            try:
                val = train.pop(i)
            except IndexError:
                val = train.pop(-1)                
            #flatten the list of lists
            train = flatten_list(train)
            kf.append([train,test,val])
        else:
            train = flatten_list(train)
            kf.append([train,test])
    return kf

def crossfolds(data, k):    
    data = np.array(data)
    folds = []
    for i, (train, test) in enumerate(kfolds(k, len(data),shuffle=True)):
        train_data = data[train].tolist()
        test_data  = data[test].tolist() 
        folds.append([train_data, test_data])   
    return folds

def read_data(path, sep="\t"):	
    data = []    
    with open(path, "r") as fid:
        for l in fid:
            splt = l.rstrip("\n").split(sep)
            data.append(splt)
    return data

def read_dataset(path, labels=None):	
    data = []
    ys = []
    with open(path, "r") as fid:
        for l in fid:
            splt = l.replace("\n", "").split("\t")
            y = splt[0]            
            x = ' '.join(splt[1:])
            data.append([y,x])	
            ys+=[y]
    if labels is not None:
        data = filter_labels(data, labels)    
    return data

def read_dataframe(df, x_label, y_label):    
    data = [ [row[y_label], row[x_label]] for _, row in df.iterrows()] 
    #shuffle data    
    random_seed = 123
    rng = np.random.RandomState(random_seed)
    rng.shuffle(data)

    return data

def save_dataset(data, out_path, labels=None):
    dirname = os.path.dirname(out_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if labels is not None: data = filter_labels(data, labels)
    with open(out_path, "w") as fod:
        for ex in data: fod.write('\t'.join(ex) + "\n")
    return data


def filter_labels(data,labels):
    #dictionaries are faster for comparisons
    labels = {l:None for l in labels}
    filtered_data = filter(lambda x:x[0] in labels, data)
    return filtered_data

