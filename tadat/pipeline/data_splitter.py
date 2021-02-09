import argparse
from bs4 import BeautifulSoup
# from ipdb import set_trace
import os
import sys
#local
from tadat.core import data as data_reader
from tadat.core.helpers import str2seed

def get_parser():
	par = argparse.ArgumentParser(description="Split Dataset")
	par.add_argument('-input', type=str, required=True, nargs='+', help='input data')
	par.add_argument('-train', type=str, required=True, help='train split path')
	par.add_argument('-test', type=str, required=True, help='test split path')
	par.add_argument('-dev', type=str, help='dev split path')
	# par.add_argument('-output', type=str, required=True, nargs=2, help='output files')
	par.add_argument('-split', type=float, default=0.8, help='data split')
	par.add_argument('-no_strat', action="store_true", help="do not stratified data for split")
	par.add_argument('-rand_seed', type=str, default="1234", help='randomization seed')
	par.add_argument('-cv', type=int, help="k-fold crossvalidation")
	return par

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()	
	try:
		seed = int(args.rand_seed)
	except ValueError:		
		seed = str2seed(args.rand_seed)	
	datasets = [data_reader.read_dataset(d)	for d in args.input]
	datasets = data_reader.flatten_list(datasets)	
	if args.cv is not None:
		print("[seed ({}) | input: {} | cv: {} | strat: {}]".format(seed, args.input, \
																	args.cv, \
																	not args.no_strat))
		folds = data_reader.crossfolds(datasets, args.cv)
		# set_trace()
		for i, (train_split, test_split) in enumerate(folds):
			tr_fname = args.train+"_"+str(i+1)
			ts_fname = args.test+"_"+str(i+1)
			if args.dev is not None:
				dev_fname = args.dev+"_"+str(i+1)
				print("[saving: {} | {} | {} ]".format(tr_fname, ts_fname, dev_fname))	
				train_split, dev_split = data_reader.shuffle_split(train_split, args.split, \
																random_seed=seed)	
				# set_trace()
				data_reader.save_dataset(dev_split, dev_fname)
			else:
				print("[saving: {} | {} ]".format(tr_fname, ts_fname))
			# set_trace()
			data_reader.save_dataset(train_split, tr_fname)
			data_reader.save_dataset(test_split, ts_fname)			
	else:		
		print("[seed ({}) | input: {} | split: {} | strat: {}]".format(seed, args.input, \
																	args.split, \
																not args.no_strat))

		train_split, test_split = data_reader.shuffle_split(datasets, args.split, \
															random_seed=seed)	
		if args.dev is not None:
			print("[saving: {} | {} | {} ]".format(args.train, args.test, args.dev))
			train_split, dev_split = data_reader.shuffle_split(train_split, args.split, \
															  random_seed=seed)	
			data_reader.save_dataset(dev_split, args.dev)
		else:
			print("[saving: {} | {} ]".format(args.train, args.test))
		data_reader.save_dataset(train_split, args.train)
		data_reader.save_dataset(test_split, args.test)
