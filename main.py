from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from scipy.sparse import coo_array
from scipy.sparse import csr_array
import numpy as np

import time
import sys
import os

#define global variables, might switch to a normal version...
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


#enfin la bonne version !!!
#path = le chemin d'accès
#ne renvoie rien mais met à jours COLS, ROWS et les vocabulaires (WORD2IDX, IDX2WORD, RELATIVE_WORD2IDX)
def load_data(path, testing=False, only_nouns=False, consider_index=True):
	global WORD2IDX
	global IDX2WORD
	global RELATIVE_WORD2IDX
	global ROWS
	global COLS
	#ROWS et COLS doivent avoir la même longueur, ils spécifient les coordonnées non nulles de la matrice creuse
	global test_corpus

	#global rows
	ROWS = list()
	#global columns
	COLS = list()
	#readfile
	with open(path, "r", encoding='utf-8', errors='ignore') as corpus:   #Chargement du corpus (nom du fichier potentiellement à changer)
		lines = corpus.readlines()


	for i in range(len(lines)):      #Pour chaque mot:

		splitline = lines[i].split("\t")	    #Colonnes séparées par des tabulations
		
		#Evite de prendre les lignes vides
		if (len(splitline)<10): 
			continue

		#recupération de l'indice (création si non présent dans WORD2IDX)
		if testing:
			key = splitline[2].lower()
			if key not in test_corpus:
				continue
		else:
			key = splitline[2].lower()
			if only_nouns and splitline[3] != "N":
				continue
			elif only_nouns == False:
				key = splitline[2].lower()

		WORD2IDX[key] = WORD2IDX.get(key, len(WORD2IDX.keys()) + 1)	   #Si la clé (lemme, cat) n'existe pas encore dans le thésaurus, on la crée
		row_index = WORD2IDX[key]

		#negative relative positions
		#i-1
		try:
			if consider_index:
				key_minus1 = lines[i-1].split("\t")[2].lower() + lines[i-1].split("\t")[3] + "-1"
			else:
				key_minus1 = lines[i-1].split("\t")[2].lower() + lines[i-1].split("\t")[3]

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
			RELATIVE_WORD2IDX[key_plus2] = RELATIVE_WORD2IDX.get(key_plus2, len(RELATIVE_WORD2IDX.keys())+1)
			pos_col2_index = RELATIVE_WORD2IDX[key_plus2]		      #On stocke le lemme du mot voisin			ROWS.append(row_index)
			ROWS.append(row_index)
			COLS.append(pos_col2_index)
		except IndexError:
			pass
	
	#switch keys(unique) and indexes(unique)
	IDX2WORD = {WORD2IDX[key]:key for key in WORD2IDX.keys()}

def get_var_stats(Matrix, path):		
	Y = {}	#similarités humaines
	with open(path, 'r', encoding = "utf-8") as file:
		lines = file.readlines()
	for line in lines:
		splitted = line.split(",")	#vérifier si besoin de trim à cause du retour chariot / remplacer ligne par genre ligne[0:len(ligne)-1]
		if (splitted[0] in WORD2IDX.keys() and splitted[1] in WORD2IDX.keys()):		#on vérifie que les deux mots sont dans notre matrice
			X[splitted[0]+" "+splitted[1]] = Matrix[WORD2IDX[splitted[0]]][WORD2IDX[splitted[1]]]		#X["nchat nchien"] = M[indice de "nchat"][indice de "nchien"]
			Y[splitted[0]+" "+splitted[1]] = float(splitted[2])/4		
		else:
			pass
	return X,Y


#FONCTION POUR CALCULER L'ECART TYPE
def sigma(X):	#avec X évidemment une variable statistique représentée par un dico
	moyx = np.mean(list(X.values()))
	array_sum = 0
	for val in X.values():
		array_sum += (val-moyx)**2
	variance = array_sum / len(X)
	return np.sqrt(variance)	#eh oui Jamy puisque l'écart-type n'est que la racine carrée de la variance


#FONCTION POUR CALCULER LA COVARIANCE, VERSION MOINS STUPIDE
def covar(X, Y):	#X Y évidemment nos deux variables stats adorées
	moyx = np.mean(list(X.values()))
	moyy = np.mean(list(Y.values()))
	array_sum = 0
	for key in X.keys():	#puisque X.keys() == Y.keys()
		array_sum += (X[key]-moyx)*(Y[key]-moyy)
	return (array_sum/len(X))


#FONCTION POUR CALCULER LA SIMILARITE LINEAIRE
def pearson(X, Y):	#X Y nos (je te laisse compléter)
	covariance = covar(X,Y)
	etypex = sigma(X)
	etypey = sigma(Y)
	return covariance / (etypex*etypey)



#FONCTION POUR CALCULER LA SIMILARITE LINEAIRE SUR LES RANGS

def spearman(X, Y):	#tjr la même
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


def test(matrix, testing=True, only_nouns=False, consider_index=True, count_documents=1):
	XX,YY = get_var_stats(matrix,"evaluation_list.txt")
	print("spearman: ",spearman(XX,YY))
	print("pearson: ",pearson(XX,YY))


def save_matrix(cosine_matrix):
	np.save("cosine_matrix", sklearn_cosine_similarity(cosine_matrix))
	print("saved")

def load_matrix():
	return np.load("cosine_matrix.npy")

def find_most_similar(matrix, key, testing=False, only_nouns=False, consider_index=True, count_documents=1):
	global IDX2WORD
	result = list()
	#compute cosine matrix
	finalmatrix = matrix
	i = WORD2IDX[key]
	for index in range(1,len(IDX2WORD.keys())):
		if index == 0 or index >= len(IDX2WORD.keys()):
			print("skip")
			continue
		else:
			try:
				sim = finalmatrix.getrow(i).getcol(index).data[0]
			except IndexError:
				continue
			if sim > 0.6 and sim < 0.99:
				tup = (IDX2WORD[i], IDX2WORD[index], sim)
				result.append(tup)

	result = sorted(result, key=lambda tup: tup[2], reverse=True,)[:10]
	print(f"\n10 most similar words of {key}:")
	for elt in result:
		print(f"{elt[1]} ({str(elt[2])[:5]})")


def load_data_and_compute_matrix(testing=False, only_nouns=False, consider_index=True, count_documents=1, sparse_cosine_matrix=True):
	global ROWS
	global COLS
	
	#reset global values
	WORD2IDX=dict()
	IDX2WORD=dict()
	RELATIVE_WORD2IDX=dict()
	ROWS =list()
	COLS =list()

	start = time.time()
	#parse corpuses
	path = "C:\\Users\\garri\\Documents\\estrepublicain.a.outmalt.tar\\estrepublicain.a.outmalt"
	path = os.getcwd()+"\\corpus"
	count = 0
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
	finalmatrix = coo_array((DATA, (ROWS, COLS)), dtype=np.int64).tocsr()
	print(f"matrix shape before cosine computing:{finalmatrix.shape}")

	#compute cosine matrix
	if sparse_cosine_matrix:
		X = sklearn_cosine_similarity(finalmatrix, dense_output=False)
	else:
		X = sklearn_cosine_similarity(finalmatrix, dense_output=True)
	end = time.time()
	#logs
	print(f"Result obtained in {end-start} seconds")
	print(f"considering only nouns: {only_nouns}")
	print(f"considering only test corpus({finalmatrix.shape[0]}/65 pairs): {testing}")
	print(f"cosine matrix shape: {X.shape}")

	return X

def run_all_tests():
	global WORD2IDX
	global IDX2WORD
	global RELATIVE_WORD2IDX
	global ROWS
	global COLS

	#compute every tests
	print("[TEST 1]")
	test(testing=True, only_nouns=False, consider_index=False)
	
	#reset global values
	WORD2IDX=dict()
	IDX2WORD=dict()
	RELATIVE_WORD2IDX=dict()
	ROWS =list()
	COLS =list()
	print("\n\n[TEST 2]")
	test(testing=True, only_nouns=True, consider_index=False)
	
	#reset global values
	WORD2IDX=dict()
	IDX2WORD=dict()
	RELATIVE_WORD2IDX=dict()
	ROWS =list()
	COLS =list()
	print("\n\n[TEST 3]")
	test(testing=True, only_nouns=False, consider_index=True)
		
	#reset global values
	WORD2IDX=dict()
	IDX2WORD=dict()
	RELATIVE_WORD2IDX=dict()
	ROWS =list()
	COLS =list()
	print("\n\n[TEST 4]")
	test(testing=True, only_nouns=True, consider_index=True)	

	#reset global values
	WORD2IDX=dict()
	IDX2WORD=dict()
	RELATIVE_WORD2IDX=dict()
	ROWS =list()
	COLS =list()
	print("\n\n[TEST 5]")
	test(testing=True, only_nouns=False, consider_index=False, count_documents=10)
	
	#reset global values
	WORD2IDX=dict()
	IDX2WORD=dict()
	RELATIVE_WORD2IDX=dict()
	ROWS =list()
	COLS =list()
	print("\n\n[TEST 6]")
	test(testing=True, only_nouns=True, consider_index=False, count_documents=10)
	
	#reset global values
	WORD2IDX=dict()
	IDX2WORD=dict()
	RELATIVE_WORD2IDX=dict()
	ROWS =list()
	COLS =list()
	print("\n\n[TEST 7]")
	test(testing=True, only_nouns=False, consider_index=True,count_documents=10)
		
	#reset global values
	WORD2IDX=dict()
	IDX2WORD=dict()
	RELATIVE_WORD2IDX=dict()
	ROWS =list()
	COLS =list()
	print("\n\n[TEST 8]")
	test(testing=True, only_nouns=True, consider_index=True,count_documents=10)	

	

def main():
	global WORD2IDX
	print("Welcome in Therranosaurus engine !\nPress[T] to display test results (pearson and spearman)\nPress [R] to find word similarities")
	user_input = input() 

	while user_input.lower() not in ["t", "r"]:
		user_input = input("enter R or T:\n")

	if user_input.lower() == "t":
		matrix = load_data_and_compute_matrix(testing=True, only_nouns=False, consider_index=False, count_documents=1, sparse_cosine_matrix=False)
		test(matrix)
	else:
		matrix = load_data_and_compute_matrix(testing=False, only_nouns=False, consider_index=False, count_documents=1, sparse_cosine_matrix=True)
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
