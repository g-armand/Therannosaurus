import sklearn.preprocessing as pp
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from scipy.sparse import coo_array
import numpy as np

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
#global rows
ROWS = list()
#global columns
COLS = list()
#global test corpus
test_corpus = ["corde", "sourire", "midi", "ficelle", "coq", "périple", "fruit", "fournaise", "autographe", "rivage", "automobile", "sorcier", "monticule", "four", "grimace", "instrument", "refuge", "fruit", "refuge", "moine", "cimetière", "asylum", "garçon", "coq", "verre", "magicien", "coussin", "bijou", "moine", "esclave", "refuge", "cimetière", "côte", "forêt", "grimace", "gars", "rivage", "bois", "moine", "oracle", "garçon", "sage", "automobile", "coussin", "monticule", "rivage", "gars", "sorcier", "forêt", "cimetière", "nourriture", "coq", "cimetière", "bois", "rivage", "trip", "oiseau", "bois", "côte", "colline", "fournaise", "instrument", "grue", "coq", "colline", "bois", "auto", "voyage", "cimetière", "monticule", "verre", "bijou", "magicien", "oracle", "grue", "instrument", "frère", "gars", "sage", "sorcier", "oracle", "sage", "oiseau", "grue", "oiseau", "coq", "nourriture", "fruit", "frère", "moine", "refuge", "asile", "fournaise", "four", "magicien", "sorcier", "colline", "monticule", "corde", "ficelle", "verre", "goblet", "grimace", "sourire", "serf", "esclave", "voyage", "périple", "autographe", "signature", "côte", "rivage", "forêt", "bois", "instrument", "outil", "coq", "coq", "garçon", "gars", "coussin", "oreiller", "cimetière", "cimetière", "automobile", "auto", "joyau", "bijou", "midi", "dîner" ]


#enfin la bonne version !!!
#path = le chemin d'accès
#ne renvoie rien mais met à jours COLS, ROWS et les vocabulaires (WORD2IDX, IDX2WORD, RELATIVE_WORD2IDX)
def load_data3(path):
	global WORD2IDX
	global IDX2WORD
	global RELATIVE_WORD2IDX

	#ROWS et COLS doivent avoir la même longueur, ils spécifient les coordonnées non nulles de la matrice creuse
	global ROWS
	global COLS
	global test_corpus

	#readfile
	with open(path, "r", encoding='utf-8', errors='ignore') as corpus:   #Chargement du corpus (nom du fichier potentiellement à changer)
		lines = corpus.readlines()


	for i in range(len(lines)):      #Pour chaque mot:
		splitline = lines[i].split("\t")	    #Colonnes séparées par des tabulations
		
		#Evite de prendre les lignes vides
		if (len(splitline)<10): 
			continue

		#Evite de prendre les mots qui ne nous servent pas (mots hors corpus de test)
		if splitline[2].lower() not in test_corpus:
			continue

		#recupération de l'indice (création si non présent dans WORD2IDX)
		key = splitline[2].lower()+splitline[3][0]
		WORD2IDX[key] = WORD2IDX.get(key, len(WORD2IDX.keys()) + 1)	   #Si la clé (lemme, cat) n'existe pas encore dans le thésaurus, on la crée
		row_index = WORD2IDX[key]

		#negative relative positions
		#i-1
		try:
			key_minus1 = lines[i-1].split("\t")[2].lower() + lines[i-1].split("\t")[3] + "-1"
			RELATIVE_WORD2IDX[key_minus1] = RELATIVE_WORD2IDX.get(key_minus1, len(RELATIVE_WORD2IDX.keys())+1)
			neg_col1_index = RELATIVE_WORD2IDX[key_minus1]		      #On stocke le lemme du mot voisin
			ROWS.append(row_index)
			COLS.append(neg_col1_index)
		except IndexError:
			continue

		#i-2
		try:
			key_minus2 = lines[i-2].split("\t")[2].lower() + lines[i-2].split("\t")[3] + "-2"
			RELATIVE_WORD2IDX[key_minus2] = RELATIVE_WORD2IDX.get(key_minus2, len(RELATIVE_WORD2IDX.keys())+1)
			neg_col2_index = RELATIVE_WORD2IDX[key_minus2]		      #On stocke le lemme du mot voisin			ROWS.append(row_index)
			ROWS.append(row_index)
			COLS.append(neg_col2_index)
		except IndexError:
			continue
		
		#positive relative positions
		#i+1
		try:
			key_plus1 = lines[i+1].split("\t")[2].lower() + lines[i-1].split("\t")[3] + "+1"
			RELATIVE_WORD2IDX[key_plus1] = RELATIVE_WORD2IDX.get(key_plus1, len(RELATIVE_WORD2IDX.keys())+1)
			pos_col1_index = RELATIVE_WORD2IDX[key_plus1]		      #On stocke le lemme du mot voisin
			ROWS.append(row_index)
			COLS.append(pos_col1_index)
		except IndexError:
			continue
		#i+2
		try:
			key_plus2 = lines[i+2].split("\t")[2].lower() + lines[i-2].split("\t")[3] + "+2"
			RELATIVE_WORD2IDX[key_plus2] = RELATIVE_WORD2IDX.get(key_plus2, len(RELATIVE_WORD2IDX.keys())+1)
			pos_col2_index = RELATIVE_WORD2IDX[key_plus2]		      #On stocke le lemme du mot voisin			ROWS.append(row_index)
			ROWS.append(row_index)
			COLS.append(pos_col2_index)
		except IndexError:
			continue
	
	#switch keys(unique) and indexes(unique)
	IDX2WORD = {WORD2IDX[key]:key for key in WORD2IDX.keys()}


#parse corpuses
path = "C:\\Users\\garri\\Documents\\estrepublicain.a.outmalt.tar\\estrepublicain.a.outmalt"
for file in os.listdir(path):
	load_data3(f"{path}\\{file}")
	print(path+"\\"+file+ " DONE", len(WORD2IDX.keys()) , sys.getsizeof(WORD2IDX))
	#break
	
#build matrix
DATA = np.ones(len(COLS))
ROWS = np.array(ROWS,dtype = np.int64)
COLS = np.array(COLS,dtype= np.int64)
finalmatrix = coo_array((DATA, (ROWS, COLS)), dtype=np.int64).tocsr()

#compute cosine matrix
finalmatrix = sklearn_cosine_similarity(finalmatrix)


#test
for i in range(1,len(IDX2WORD.keys())):
	if i >= len(IDX2WORD.keys()):
		print("skipping index: ",i)
		continue
	else:
		for index in range(1,len(IDX2WORD.keys())):
			if index == 0 or index >= len(IDX2WORD.keys()):
				print("skip")
				continue
			else:
				if finalmatrix[i][index] > 0.85 and finalmatrix[i][index]<0.98:
					print(IDX2WORD[i], IDX2WORD[index], finalmatrix[i][index])
