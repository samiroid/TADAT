import argparse
from collections import defaultdict
import pickle
# from ipdb import set_trace
import numpy as np
import os
from sklearn.linear_model import SGDClassifier 
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import sys

from tadat.core import helpers


def get_features(data_path, features):	
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
			x = np.load(data_path+feat_suffix, allow_pickle=True)
			#check if this is a sparse matrix (which have no 'shape')
			try:
				_ = x.shape[0]
			except IndexError:				
				x = x.item()
			if X is None: 
				X = x
			else:
				X = np.concatenate((X,x),axis=1)
	return X, Y, label_dict

def hypertune(train, dev, features, obj, hyperparams, res_path=None):
	X_train, Y_train, _ = get_features(train, features)
	X_dev,  Y_dev, _  = get_features(dev, features)			 
	best_hp = None
	best_score = 0
	for hp in hyperparams:
		#initialize model with the hyperparameters				
		model = SGDClassifier(random_state=1234,**hp)
		model.fit(X_train,Y_train)
		Y_hat = model.predict(X_dev)
		score = obj(Y_dev, Y_hat)		
		if score > best_score:
			best_score = score
			best_hp = hp
		res = {"score":round(score,3), "hyper":repr(hp)}
		if res_path is not None:
			helpers.save_results(res,res_path)
		helpers.print_results(res)	
	print("\n[best conf: {} | score: {}]".format(repr(best_hp),round(best_score,3)))
	return best_hp, best_score

def main(train, test, run_id, features, hyperparameters=None, res_path=None):	
	if not hyperparameters: hyperparameters = {}
	#train and evalute model		
	if features[0].lower() == "naive_bayes":
		X_train, Y_train, labels = get_features(train, ["bow-bin"])
		X_test,  Y_test, labels  = get_features(test, ["bow-bin"])	
		model = BernoulliNB()
		model_name = "NaiveBayes"
	elif features[0].lower() == "mlp":
		X_train, Y_train, labels = get_features(train, ["bow-bin"])
		X_test,  Y_test, labels  = get_features(test, ["bow-bin"])	
		model = MLPClassifier(solver='lbfgs', activation="logistic", hidden_layer_sizes=[400])
		model_name = "MLP"
	elif features[0].lower() == "mlp-2":
		X_train, Y_train, labels = get_features(train, ["bow-bin"])
		X_test,  Y_test, labels  = get_features(test, ["bow-bin"])			
		model = MLPClassifier(solver='lbfgs', activation="logistic", hidden_layer_sizes=[400,100])
		model_name = "MLP-2"
	else:
		X_train, Y_train, labels = get_features(train, features)
		X_test,  Y_test, labels  = get_features(test, features)	
		#initialize model with the hyperparameters	
		model = SGDClassifier(random_state=1234,**hyperparameters)
		model_name = "+".join(features)	
	model.fit(X_train,Y_train)
	Y_hat = model.predict(X_test)
	avgF1 = f1_score(Y_test, Y_hat,average="macro") 		
	acc = accuracy_score(Y_test, Y_hat)					
	res = {"acc":round(acc,3), \
			"avgF1":round(avgF1,3),	\
			"model":model_name, \
			"dataset":os.path.basename(test), \
			"run_id":run_id, \
			"train_size":X_train.shape[0], \
			"test_size":X_test.shape[0], \
			"hyper":repr(hyperparameters)}
	cols = ["dataset", "run_id", "acc", "avgF1","hyper"]
	helpers.print_results(res, columns=cols)	
	precision, recall, fscore, support = score(Y_test, Y_hat)	
	for l, i in labels.items():
		print("\n== {} ==".format(l))
		print('precision: {}'.format(round(precision[i],3)))
		print('recall: {}'.format(round(recall[i],3)))
		print('fscore: {}'.format(round(fscore[i],3)))
		print('support: {}'.format(round(support[i],3)))
		print("")
	if res_path is not None:
		cols = ["dataset", "model", "run_id", "acc", "avgF1"]
		helpers.save_results(res, res_path, sep="\t", columns=cols)
	return res
		
def get_parser():
	par = argparse.ArgumentParser(description="Document Classifier")
	par.add_argument('-train', type=str, required=True, help='train data')
	par.add_argument('-dev', type=str, help='dev data')
	par.add_argument('-test', type=str, required=True, help='test data')
	par.add_argument('-features', type=str, required=True, nargs='+', help='features')	
	par.add_argument('-run_id', type=str, help='run id')
	par.add_argument('-res_path', type=str, help='results file')
	par.add_argument('-silent', action="store_true",help='silent')
	par.add_argument('-pos_label', type=str, default="positive", \
					help='label for the positive class')
	par.add_argument('-neg_label', type=str, default="negative", \
					help='label for the negative class')
	par.add_argument('-cv', type=int, help='crossfold')
	par.add_argument('-hyperparams_path', type=str, default="", help='path to a dictionary of hyperparameters')

	return par

if __name__=="__main__":	
	parser = get_parser()
	args = parser.parse_args()  	
	#open datasets
	#train	
	print("[features: {}]".format("+".join(args.features)))
	if args.run_id is None: args.run_id = "+".join(args.features)		
	hyper_results_path = None

	hyperparams_grid = []
	if os.path.isfile(args.hyperparams_path):
		assert args.dev is not None, "Need a dev set for hyperparameter search"		
		hyperparams_grid = helpers.get_hyperparams(args.hyperparams_path, {})
		print("[tuning hyperparameters from @ {}]".format(args.hyperparams_path))
		if args.res_path is not None:            
			fname, _ = os.path.splitext(args.res_path)            
			hyper_results_path = fname+"_"+os.path.basename(args.test)+"_hyper.txt"
		else:
			hyper_results_path = None
		scorer = lambda y_true,y_hat: f1_score(y_true, y_hat,average="macro") 	
			
	if args.cv is None:			
		if len(hyperparams_grid) > 0:	
			best_hyper, _ = hypertune(args.train, args.dev, args.features, \
									scorer, hyperparams_grid, res_path=hyper_results_path)
		else:
			best_hyper = {}
		#run model with the best hyperparams
		main(args.train, args.test, args.run_id, args.features, best_hyper, args.res_path)
	else:
		assert args.cv > 2, "need at leat 2 folds for cross-validation"
		results = []
		cv_results_path = None
		if args.res_path is not None:
			#in CV experiments save the results of each fold in an external file
			fname, _ = os.path.splitext(args.res_path)
			cv_results_path = fname+"_"+os.path.basename(args.test)+"_CV.txt"		
		#loop through cross-validation folds
		for cv_fold in range(1, args.cv+1):
			if len(hyperparams_grid) > 0:
				best_hyper, _ = hypertune(args.train, args.dev, args.features, \
										scorer, hyperparams_grid, res_path=hyper_results_path)
			else:
				best_hyper = {}
			print(cv_results_path)
			#run model with the best hyperparams
			res = main(args.train+"_"+str(cv_fold), args.test+"_"+str(cv_fold), \
				args.run_id+"_"+str(cv_fold), args.features, best_hyper, cv_results_path)
			results.append(res)
		
		accs = [res["acc"] for res in results ]
		f1s = [res["avgF1"] for res in results ]
		
		cv_res = {"acc_mean":round(np.mean(accs),3), \
				"acc_std":round(np.std(accs),3), \
				"avgF1_mean":round(np.mean(f1s),3), \
				"avgF1_std":round(np.std(f1s),3), \
				"model":"+".join(args.features), \
				"dataset":os.path.basename(args.test), \
				"run_id":args.run_id}
		helpers.print_results(cv_res)
		#save the results of each run 
		if args.res_path is not None:
			colz = ["dataset", "run_id", "model", "acc_mean","acc_std","avgF1_mean","avgF1_std"]
			helpers.save_results(cv_res, args.res_path, columns=colz)
			



	
