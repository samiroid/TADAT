import numpy as np
import os
from collections import Counter
from scipy.sparse import dok_matrix
import pickle


def NLSE(X, Y):
	lens = np.array([len(tr) for tr in X]).astype(int)
	st = np.cumsum(np.concatenate((np.zeros((1, )), lens[:-1]), 0)).astype(int)
	ed = (st + lens)
	x = np.zeros((ed[-1], 1))
	for i, ins_x in enumerate(X):
		x[st[i]:ed[i]] = np.array(ins_x, dtype=int)[:, None]
	X = x
	Y = np.array(Y)

	return X, Y, st, ed

def BOW(docs, vocab_size, sparse=True):
	"""
		Extract bag-of-word features
	"""		
	X = dok_matrix((len(docs), vocab_size))	
	for i, doc in enumerate(docs):
		try:
			X[i, np.array(doc)] = 1
		except IndexError:
			pass
	if sparse: 
		# print("(sparse BOW)")
		return X.tocsr()	
	# print("(dense BOW)")
	return X.todense()

def BOW_freq(docs, vocab_size, sparse=True):
	"""
		Extract bag-of-word features
	"""		
	X = dok_matrix((len(docs), vocab_size))
	for i, doc in enumerate(docs):		
		if len(doc) == 0: continue
		ctr = Counter(doc)
		cts = [ctr[w] for w in doc]		
		try:
			X[i, np.array(doc)] = np.array(cts)
		except IndexError:
			pass
	if sparse: 
		# print("(sparse BOW)")
		return X.tocsr()	
	# print("(dense BOW)")
	return X.todense()

def BOW_tfidf(docs, vocab_size, idfs, sparse=True):
	"""
		Extract bag-of-word features
	"""		
	X = lil_matrix((len(docs), vocab_size))	
	for i, doc in enumerate(docs):		
		if len(doc) == 0: continue
		ctr = Counter(doc)
		cts = np.array([ctr[w] for w in doc])
		idf = np.array(idfs[doc])
		w = cts*idf		
		try:
			X[i, np.array(doc)] = w
		except IndexError:
			pass
	if sparse: 
		# print("(sparse BOW)")
		return X.tocsr()	
	# print("(dense BOW)")
	return X.todense()

def BOE(docs, E, agg='sum'):
	"""
		Build Bag-of-Embedding features
	"""
	assert agg in ["sum", "bin"] 
	X = np.zeros((len(docs), E.shape[0]))
	if agg == 'sum':
		for i, doc in enumerate(docs):
			X[i, :] = E[:, doc].T.sum(axis=0)
	elif agg == 'bin':
		for i, doc in enumerate(docs):
			unique = list(set(doc))
			X[i, :] = E[:, unique].T.sum(axis=0)
	return X

def read_features(data_path, features):	
	with open(data_path,"rb") as fid:
		train_data = pickle.load(fid)
		X = None
		Y = np.array(train_data[1])	
		label_dict = train_data[3]		
		#remove extension from filename
		data_path = os.path.splitext(data_path)[0]		
		#get features
		for ft in features:			
			feat_suffix = "-"+ft+".npy" 					
			x = np.load(data_path+feat_suffix)
			if X is None: 
				X = x
			else:
				X = np.concatenate((X,x),axis=1)
	return X, Y, label_dict
