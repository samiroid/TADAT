import argparse
import codecs
import pickle
import os
# from ipdb import set_trace
import numpy as np
import sys

#local
from tadat.core import features, embeddings

def run(inputs, opts):
	for dataset in inputs:
		print("[extracting features @ {}]".format(repr(dataset)))
		E = None
		with open(dataset, "rb") as fid:
			X, _, vocabulary, _ = pickle.load(fid)
			basename = os.path.splitext(os.path.basename(dataset))[0]			
			if opts.bow is not None:
				for agg in opts.bow:
					fname = basename + "-bow-" + agg.lower()
					print("\t > BOW ({})".format(fname))
					if agg == "bin":
						bow = features.BOW(X, len(vocabulary), opts.sparse_bow)
					elif agg == "freq":
						bow = features.BOW_freq(X, len(vocabulary), opts.sparse_bow)
					np.save(opts.out_folder + fname, bow)					
			if opts.boe is not None:
				for agg in opts.boe:
					fname = basename + "-boe-" + agg.lower()
					print("\t > BOE ({})".format(fname))
					E = embeddings.read_embeddings(opts.embeddings, vocab=vocabulary)
					boe = features.BOE(X, E, agg=agg)					
					np.save(opts.out_folder + fname, boe)
			if opts.w2v:				
				fname = basename + "-w2v"
				print("\t > W2V ({})".format(fname))
				E = embeddings.read_embeddings(opts.embeddings, vocab=vocabulary)
				emb = features.BOE(X, E, agg="bin")					
				np.save(opts.out_folder + fname, emb)
			if opts.u2v:				
				fname = basename + "-u2v"
				print("\t > u2v ({})".format(fname))
				E = embeddings.read_embeddings(opts.embeddings, vocab=vocabulary)
				emb = features.BOE(X, E, agg="bin")					
				np.save(opts.out_folder + fname, emb)
			if opts.nlse:
				fname = basename + "_NLSE.pkl"				
				E = embeddings.read_embeddings(opts.embeddings, vocab=vocabulary)
				np.save(fname, E)

def get_parser():
	par = argparse.ArgumentParser(description="Extract Features")
	par.add_argument('-input', type=str, required=True, nargs='+', help='train data')
	par.add_argument('-out_folder', type=str, required=True, help='output folder')
	par.add_argument('-bow', type=str, choices=['bin', 'freq'], nargs='+', help='bow features')
	par.add_argument('-boe', type=str, choices=['bin', 'sum'], nargs='+', help='boe features')
	par.add_argument('-u2v', action="store_true", help='(single) user embedding feature')
	par.add_argument('-w2v', action="store_true", help='(single) word embedding feature')
	par.add_argument('-nlse', action="store_true")
	par.add_argument('-sparse_bow', action="store_true")
	par.add_argument('-cv', type=int, help='crossfold')
	par.add_argument('-cv_from', type=str, nargs='*', \
					help="files for crossvalidation")
	par.add_argument('-embeddings', type=str, help='path to embeddings')

	return par

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()
	assert args.bow is not None or args.boe is not None or args.nlse or args.w2v or args.u2v, "please, specify some features"
	if args.boe is not None or args.nlse:
		assert args.embeddings is not None, "missing embeddings"

	#create output folder if needed
	args.out_folder = args.out_folder.rstrip("/") + "/"
	if not os.path.exists(os.path.dirname(args.out_folder)):
	    os.makedirs(os.path.dirname(args.out_folder))

	#loop through cross-validation folds (if any)
	if args.cv is None:
		fnames = args.input
		run(fnames, args)
	else:
		assert args.cv > 2, "need at leat 2 folds for cross-validation"
		for cv_fold in range(1, args.cv+1):
			if args.cv_from is None:
				cv_fnames = [f+"_"+str(cv_fold) for f in args.input]
			else:
				cv_fnames = [f + "_" + str(cv_fold) for f in args.cv_from]
			run(cv_fnames, args)			

	
