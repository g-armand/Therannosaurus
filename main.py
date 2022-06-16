from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from scipy.sparse import coo_array
from scipy.sparse import csr_array
import numpy as np

import time
import sys
import os

#define global variables, it's easier ton work with when we need them in several methods
#global thesaurus
thesaurus = dict()
#global vocab
WORD2IDX = dict()
#global relative vocab
RELATIVE_WORD2IDX = dict()
#global index vocab
IDX2WORD = dict()
#global test corpus
test_corpus = ["corde", "sourire", "midi", "ficelle", "coq", "périple", "fruit", "fournaise", "autographe", "rivage", "automobile", "sorcier", "monticule", "four", "grimace", "instrument", "refuge", "fruit", "refuge", "moine", "cimetière", "asylum", "garçon", "coq", "verre", "magicien", "coussin", "bijou", "moine", "esclave", "refuge", "cimetière", "côte", "forêt", "grimace", "gars", "rivage", "bois", "moine", "oracle", "garçon", "sage", "automobile", "coussin", "monticule", "rivage", "gars", "sorcier", "forêt", "cimetière", "nourriture", "coq", "cimetière", "bois", "rivage", "trip", "oiseau", "bois", "côte", "colline", "fournaise", "instrument", "grue", "coq", "colline", "bois", "auto", "voyage", "cimetière", "monticule", "verre", "bijou", "magicien", "oracle", "grue", "instrument", "frère", "gars", "sage", "sorcier", "oracle", "sage", "oiseau", "grue", "oiseau", "coq", "nourriture", "fruit", "frère", "moine", "refuge", "asile", "fournaise", "four", "magicien", "sorcier", "colline", "monticule", "corde", "ficelle", "verre", "goblet", "grimace", "sourire", "serf", "esclave", "voyage", "périple", "autographe", "signature", "côte", "rivage", "forêt", "bois", "instrument", "outil", "coq", "coq", "garçon", "gars", "coussin", "oreiller", "cimetière", "cimetière", "automobile", "auto", "joyau", "bijou", "midi", "dîner" ]



def load_data(path, testing=False, only_nouns=False, consider_index=True):
	"""
	Updates global lists COLS, ROWS and global vocabularies WORD2IDX, IDX2WORD, RELATIVE_WORD2IDX

	Parameters: 
		-str(path): full path to file to read
		-bool(testing): must be True only if test() is called after, otherwise False
		-bool(only_nouns): if True, will get only nouns(rows) but will still consider other classes for context(columns)
		-bool(consider_index): if True, will store columns keys like "word-1", "word+2", if False will store "word"
	"""
	global WORD2IDX
	global IDX2WORD
	global RELATIVE_WORD2IDX
	#ROWS and COLS must be same length, they indicate coordinates of sparse matrix (scipy.sparse.coo_array)
	global ROWS
	global COLS
	global test_corpus

	#global rows and columns
	ROWS = list()
	COLS = list()

	#readfile
	with open(path, "r", encoding='utf-8', errors='ignore') as corpus:   #Chargement du corpus (nom du fichier potentiellement à changer)
		lines = corpus.readlines()

	#iteration over each line of file
	for i in range(len(lines)):

		splitline = lines[i].split("\t")
		
		#avoids empty lines	
		if (len(splitline)<10): 
			continue

		key = splitline[2].lower()

		#for tests we only need keys that are represented in test_corpus, no need to store more keys than that
		if testing:
			if key not in test_corpus:
				continue
		else:
			if only_nouns and splitline[3] != "N":
				continue

		#get index (or create it if not in WORD2IDX)
		WORD2IDX[key] = WORD2IDX.get(key, len(WORD2IDX.keys()) + 1)	   #Si la clé (lemme, cat) n'existe pas encore dans le thésaurus, on la crée
		row_index = WORD2IDX[key]

		#negative relative positions
		#i-1
		try:
			if consider_index:
				key_minus1 = lines[i-1].split("\t")[2].lower() + lines[i-1].split("\t")[3] + "-1"
			else:
				key_minus1 = lines[i-1].split("\t")[2].lower() + lines[i-1].split("\t")[3]

			#column indexes
			RELATIVE_WORD2IDX[key_minus1] = RELATIVE_WORD2IDX.get(key_minus1, len(RELATIVE_WORD2IDX.keys())+1)
			neg_col1_index = RELATIVE_WORD2IDX[key_minus1]		      #On stocke le lemme du mot voisin
			ROWS.append(row_index)
			COLS.append(neg_col1_index)
		except IndexError:
			pass
		#i-2
		try:
			if consider_index:
				key_minus2 = lines[i-2].split("\t")[2].lower() + lines[i-2].split("\t")[3] + "-2"
			else:
				key_minus2 = lines[i-2].split("\t")[2].lower() + lines[i-2].split("\t")[3]

			#column indexes
			RELATIVE_WORD2IDX[key_minus2] = RELATIVE_WORD2IDX.get(key_minus2, len(RELATIVE_WORD2IDX.keys())+1)
			neg_col2_index = RELATIVE_WORD2IDX[key_minus2]		      #On stocke le lemme du mot voisin			ROWS.append(row_index)
			ROWS.append(row_index)
			COLS.append(neg_col2_index)
		except IndexError:
			pass
		
		#positive relative positions
		#i+1
		try:
			if consider_index:
				key_plus1 = lines[i+1].split("\t")[2].lower() + lines[i+1].split("\t")[3] + "+1"
			else:
				key_plus1 = lines[i+1].split("\t")[2].lower() + lines[i+1].split("\t")[3]
			RELATIVE_WORD2IDX[key_plus1] = RELATIVE_WORD2IDX.get(key_plus1, len(RELATIVE_WORD2IDX.keys())+1)
			pos_col1_index = RELATIVE_WORD2IDX[key_plus1]		      #On stocke le lemme du mot voisin
			ROWS.append(row_index)
			COLS.append(pos_col1_index)
		except IndexError:
			pass
		#i+2
		try:
			if consider_index:
				key_plus2 = lines[i+2].split("\t")[2].lower() + lines[i+2].split("\t")[3] + "+2"
			else:
				key_plus2 = lines[i+2].split("\t")[2].lower() + lines[i+2].split("\t")[3]

			#column indexes
			RELATIVE_WORD2IDX[key_plus2] = RELATIVE_WORD2IDX.get(key_plus2, len(RELATIVE_WORD2IDX.keys())+1)
			pos_col2_index = RELATIVE_WORD2IDX[key_plus2]
			ROWS.append(row_index)
			COLS.append(pos_col2_index)
		except IndexError:
			pass
	
	#switch keys(unique) and indexes(unique)
	IDX2WORD = {WORD2IDX[key]:key for key in WORD2IDX.keys()}

#FOR TESTING
def get_var_stats(Matrix, path):
	"""
	Creates stat variables
	parameters:
		-numpy.ndarray(Matrix) : Dense matrix to run our tests on
		-str(path): full path to test sample
	"""
	global WORD2IDX	
	X = {}	#thesaurus similarities
	Y = {}	#human similarities

	#read file
	with open(path, 'r', encoding = "utf-8") as file:
		lines = file.readlines()

	for line in lines:
		splitted = line.split(",")

		#check if both words are in vocab
		if (splitted[0] in WORD2IDX.keys() and splitted[1] in WORD2IDX.keys()):
			try:
				X[splitted[0]+" "+splitted[1]] = Matrix[WORD2IDX[splitted[0]]][WORD2IDX[splitted[1]]]		#X["nchat nchien"] = M[indice de "nchat"][indice de "nchien"]
				Y[splitted[0]+" "+splitted[1]] = float(splitted[2])/4
			except IndexError:
				pass	
		else:
			pass

	return X,Y


#FOR TESTING
def sigma(X):
	"""
	Computes standard deviation (ecart-type) of X(stat variables)
	parameters:
		-dict(X): dictionnary of keys="word1 word2" and values= a float() between 0.0 and 1.0
	"""
	moyx = np.mean(list(X.values()))
	array_sum = 0
	for val in X.values():
		array_sum += (val-moyx)**2
	variance = array_sum / len(X)
	return np.sqrt(variance)	#eh oui Jamy puisque l'écart-type n'est que la racine carrée de la variance


#FOR TESTING
def covar(X, Y):
	"""
	computes covariance between X and Y (stat variables)
	parameters:
		-dict(X), dict(Y): dictionnaries of keys="word1 word2" and values= a float() between 0.0 and 1.0
	"""
	moyx = np.mean(list(X.values()))
	moyy = np.mean(list(Y.values()))
	array_sum = 0
	for key in X.keys():
		array_sum += (X[key]-moyx)*(Y[key]-moyy)
	return (array_sum/len(X))


#FOR TESTING
def pearson(X, Y):
	"""
	computes pearson's linear similarity between  X and Y (stat variables)
	parameters:
		-dict(X), dict(Y): dictionnaries of keys="word1 word2" and values= a float() between 0.0 and 1.0	
	"""
	covariance = covar(X,Y)
	etypex = sigma(X)
	etypey = sigma(Y)
	return covariance / (etypex*etypey)



#FOR TESTING
def spearman(X, Y):	
	"""
	computes spearman's linear similarity (rangs) between  X and Y (stat variables)
	parameters:
		-dict(X), dict(Y): dictionnaries of keys="word1 word2" and values= a float() between 0.0 and 1.0
	"""
	rgX = {}
	rgY = {}
	valX = sorted(list(X.values()))
	valY = sorted(list(Y.values()))

	for i in range(len(X)):
		minx = valX.pop(0)
		miny = valY.pop(0)
		for key in X.keys():
			if X[key] == minx:
				rgX[key] = i+1

		for key in Y.keys():
			if Y[key] == miny:
				rgY[key] = i+1

	return pearson(rgX, rgY)

#FOR TESTING
def test(matrix):
	"""
	Run tests on Matrix and prints results
	parameters:
		-numpy.ndarray(matrix): the dense cosine matrix
	"""
	XX,YY = get_var_stats(matrix,"evaluation_list.txt")
	print("spearman: ",spearman(XX,YY))
	print("pearson: ",pearson(XX,YY))


def find_most_similar(matrix, key):
	"""
	prints top 10 most similar words of key
	parameters:
		-scipy.sparse.csr_array(matrix): the sparse cosine matrix
		-string(key): word to compare
	"""
	global IDX2WORD
	result = list()
	i = WORD2IDX[key]

	for index in range(1,len(IDX2WORD.keys())):
		if index == 0 or index >= len(IDX2WORD.keys()):
			print("skip")
			continue
		else:
			try:
				sim = matrix.getrow(i).getcol(index).data[0]
			#in case similarity does not exists between the two words (no match at all in whole corpus)
			except IndexError:
				continue

			#get significant similarities
			if sim > 0.6 and sim < 0.99:
				tup = (IDX2WORD[i], IDX2WORD[index], sim)
				result.append(tup)

	#get 10 highest similarities
	result = sorted(result, key=lambda tup: tup[2], reverse=True,)[:10]

	for elt in result:
		print(f"{elt[1]} ({str(elt[2])[:5]})")


def load_data_and_compute_matrix(testing=False, only_nouns=False, consider_index=True, count_documents=1, sparse_cosine_matrix=True):
	"""
	Central method !
	will load data from all files
	will compute cosine similarity matrix and will return it
	parameters:  
		-bool(testing): must be True only if test() is called after, otherwise False
		-bool(only_nouns): if True, will get only nouns(rows) but will still consider other classes for context(columns)
		-bool(consider_index): if True, will store columns keys like "word-1", "word+2", if False will store "word"
		-int(count_documents): how many documents to load (max 78)(the highest the slowest)
		-bool(sparse_cosine_matrix): returns numpy.ndarray(matrix) if False, otherwise scipy.sparse.csr_array(matrix)
									must have a opposite value of testing
	"""
	global ROWS
	global COLS
	global WORD2IDX

	#reset global values
	WORD2IDX=dict()
	IDX2WORD=dict()
	RELATIVE_WORD2IDX=dict()
	ROWS =list()
	COLS =list()
	start = time.time()

	path = os.getcwd()+"\\corpus"

	count = 0
	#there are max 78 corpus in this project
	if count_documents>78:
		count_documents=1

	#load files' data
	for file in os.listdir(path):
		load_data(f"{path}\\{file}", testing, only_nouns, consider_index)
		print(f"loaded {file}, {len(WORD2IDX.keys())+1} different keys")
		count += 1

		if count == count_documents:
			break

	#build matrix
	DATA = np.ones(len(COLS))
	ROWS = np.array(ROWS,dtype = np.int64)
	COLS = np.array(COLS,dtype= np.int64)

	#coo_array is efficient to build sparse arrays
	#converting to csr_array is reccomended because csr_arrays are efficient for math operations
	finalmatrix = coo_array((DATA, (ROWS, COLS)), dtype=np.int64).tocsr()
	print(f"matrix shape before cosine computing:{finalmatrix.shape}")

	#compute cosine matrix
	if sparse_cosine_matrix:
		X = sklearn_cosine_similarity(finalmatrix, dense_output=False) #scipy.sparse.csr_array
	else:
		X = sklearn_cosine_similarity(finalmatrix, dense_output=True) #numpy.ndarray
	end = time.time()
	#logs
	print(f"Result obtained in {end-start} seconds")
	print(f"considering only nouns: {only_nouns}")
	print(f"considering only test corpus({finalmatrix.shape[0]}/65 pairs): {testing}")
	print(f"cosine matrix shape: {X.shape}")

	return X


def main():
	global WORD2IDX
	print("Welcome in Therranosaurus engine !\nPress[T] to display test results (pearson and spearman)\nPress [R] to find word similarities")
	user_input = input()

	while user_input.lower() != "t":
		user_input = input("enter R or T:\n")

	if user_input.lower() == "t":
		matrix = load_data_and_compute_matrix(testing=True, only_nouns=False, consider_index=True, count_documents=1, sparse_cosine_matrix=False)
		test(matrix)
		input()
	else:
		matrix = load_data_and_compute_matrix(testing=False, only_nouns=False, consider_index=True, count_documents=1, sparse_cosine_matrix=True)
		while True:
			user_input = input("Find 10 most similar words of: ")
			if user_input.lower() == "q":
				break
			elif user_input not in WORD2IDX.keys():
				print(f"Wrong input, {user_input} is not mapped in our thesaurus...")
				continue
			else:
				find_most_similar(matrix, user_input.lower())
				print("\n")
			
		
main()
