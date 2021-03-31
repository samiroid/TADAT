import argparse
import codecs
import pickle
# from ipdb import set_trace
import os
import sys

from tadat.core.vectorizer import docs2idx, get_words_vocab, get_labels_vocab
from tadat.core.embeddings import extract_embeddings
from tadat.core.data import read_dataset, flatten_list

def get_vocabulary(fnames, max_words=None):		
	datasets = []	
	for fname in fnames:		
		ds = read_dataset(fname)
		# set_trace()
		datasets.append(ds)
	datasets = flatten_list(datasets)
	vocab_docs = [x[1] for x in datasets]		
	vocab = get_words_vocab(vocab_docs, max_words=max_words)	
	return vocab

def get_labels(fnames):		
	datasets = []	
	for fname in fnames:		
		ds = read_dataset(fname)
		datasets.append(ds)
	datasets = flatten_list(datasets)	
	Y = [x[0] for x in datasets]			
	return get_labels_vocab(Y)

def vectorize(dataset, word_vocab, label_vocab):
	docs = [x[1] for x in dataset]
	Y = [label_vocab[x[0]] for x in dataset]
	X, _ = docs2idx(docs, word_vocab)	
	return X, Y

def main(fnames, word_vocab, label_vocab, opts):
	#read data
	datasets = []
	for fname in fnames:
		print("[reading data @ {}]".format(repr(fname)))
		ds = read_dataset(fname)
		datasets.append(ds)	
	#vectorize
	print("[vectorizing documents]")
	for name, ds in zip(fnames, datasets):
		X, Y = vectorize(ds, word_vocab, label_vocab)
		basename = os.path.splitext(os.path.basename(name))[0]
		path = opts.out_folder + basename
		N = len(Y)
		print("[saving data (N={}) @ {}]".format(N, path))
		with open(path, "wb") as fid:
			pickle.dump([X, Y, word_vocab, label_vocab], fid, -1)
	return word_vocab

def get_parser():
	par = argparse.ArgumentParser(description="Extract Indices")
	par.add_argument('-input', type=str, required=True, nargs='+', help='train data')
	par.add_argument('-out_folder', type=str, required=True, help='output folder')	
	par.add_argument('-cv', type=int, help='crossfold')
	par.add_argument('-cv_from', type=str, nargs='*', help="files for crossvalidation")
	par.add_argument('-embeddings', type=str, nargs='+', help='path to embeddings')	
	par.add_argument('-embedding_encoding', type=str, default="utf-8", help='encoding of the embedding file. default = utf-8')	
	par.add_argument('-vocab_size', type=int, \
						help='max number of types to keep in the vocabulary')
	par.add_argument('-vocab_from', type=str, nargs='*', \
						help="compute vocabulary from these files")
	return par

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()
	
	#create output folder if needed
	args.out_folder = args.out_folder.rstrip("/") + "/"
	if not os.path.exists(os.path.dirname(args.out_folder)):
	    os.makedirs(os.path.dirname(args.out_folder))

	#loop through cross-validation folds (if any)
	if args.cv is None:
		all_fnames = args.input
		print("[computing vocabulary]")
		if args.vocab_from is not None:
			words = get_vocabulary(args.vocab_from, args.vocab_size)
		else:
			words = get_vocabulary([args.input[0]], args.vocab_size)		
		print("[vocabulary size: {}]".format(len(words)))
		labels = get_labels(args.input)
		main(all_fnames, words, labels, args)
	else:
		assert args.cv > 2, "need at leat 2 folds for cross-validation"
		for cv_fold in range(1, args.cv+1):
			if args.cv_from is None:
				cv_fnames = [f+"_"+str(cv_fold) for f in args.input]
			else:
				cv_fnames = [f + "_" + str(cv_fold) for f in args.cv_from]
			print("[computing vocabulary]")
			if args.vocab_from is not None:
				cv_vocab_fnames = [f+"_"+str(cv_fold) for f in args.vocab_from]				
				words = get_vocabulary(cv_vocab_fnames, args.vocab_size)
			else:
				words = get_vocabulary([args.cv_fnames[0]], args.vocab_size)
			labels = get_labels(args.input)
			print("[vocabulary size: {}]".format(len(words)))
			main(cv_fnames, words, labels, args)						
			
	#extract embeddings
	if args.embeddings is not None:
		for vecs_in in args.embeddings:
			print("[reading embeddings @ {} | enc: {}]".format(vecs_in, args.embedding_encoding))
			vecs_out = args.out_folder + os.path.basename(vecs_in)
			print("[saving embeddings @ {}]".format(vecs_out))			
			extract_embeddings(vecs_in, vecs_out, words, encoding=args.embedding_encoding)
