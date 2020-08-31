import itertools
import json
import numpy as np
import os
import pandas as pd
from pdb import set_trace
import pprint
import string

def colstr(st, color, best=False):    
    if color == 'red':
        cstring = "\033[31m" + st  + "\033[0m"
    elif color == 'green':    
        cstring = "\033[32m" + st  + "\033[0m"
    else:
        cstring = st
    if best:
        cstring+=" **"
    return cstring   


def str2seed(s):
	"""
		simple heuristic to convert strings into a digit
		map each character to its index in the alphabet (e.g. 'a'->1, 'b'->2)
	"""
	assert isinstance(s,str), "input is not a string"
	seed = ""
	for c in s.lower():		
		try:
			z = string.ascii_letters.index(c)+1	
		except ValueError:
			z=0
		seed += str(z)
	#limit the seed to 9 digits
	return int(seed[:9])

def save_results(results, path, columns=None, sep=","):

	if columns is not None:
		row = [str(results[c]) for c in columns if c in results]
	else:
		columns = results.keys()
		row = [str(v) for v in results.values()]

	dirname = os.path.dirname(path)
	if not os.path.exists(dirname):
		os.makedirs(dirname)

	if not os.path.exists(path):
		with open(path, "w") as fod:
			fod.write(sep.join(columns) + "\n")
			fod.write(sep.join(row) + "\n")
	else:
		with open(path, "a") as fod:
			fod.write(sep.join(row) + "\n")

def print_results(results, columns=None):
	if columns is not None:
		row = [results[c] for c in columns if c in results]
	else:
		columns = results.keys()
		row = results.values()
	s = ["{}: {}| ".format(c, r) for c, r in zip(columns, row)]
	print("** " + "".join(s))

def read_results(df, metric="avgF1", run_ids=None, models=None, datasets=None):
	"""
		read results:
		assumes the following columns: dataset, model, [metric]
	"""
	# set_trace()
	df.drop_duplicates(subset=["model","dataset","run_id"],inplace=True)
	
	if run_ids is not None:
		#filter by run_ids 
		df = df[df["run_id"].isin(run_ids)]
	
	#get datasets 
	if datasets is not None:
		df = df[df["dataset"].isin(datasets)]		
	if models is not None:
		df = df[df["model"].isin(models)]		
	
	uniq_models = df["model"].unique().tolist()
	uniq_datasets = df["dataset"].unique().tolist()	
	dt = [[d] + df[df["dataset"]==d][metric].values.tolist() for d in uniq_datasets]	
	columns = ["dataset"] + uniq_models	
	# set_trace()
	df = pd.DataFrame(dt,columns=columns)
	# set_trace()
	if models is not None:
		#re-order model columns
		return df[["dataset"]+models]
	return df

def get_hyperparams(path, default_conf):
	confs = json.load(open(path, 'r'))	
	#add configurations that are not specified with the default values
	confs = dict(default_conf.items() + confs.items())
	params = confs.keys()
	choices = [x if isinstance(x,list) else [x] for x in confs.values()]
	combs = list(itertools.product(*choices))	
	hyperparams = [{k:v for k,v in zip(c,x)} for x, c in zip(combs, [params]*len(combs))]
	return hyperparams
