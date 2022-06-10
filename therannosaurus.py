import os
import locale
import numpy as np
from numpy.linalg import norm
import sys
os.environ["PYTHONIOENCODING"] = "utf-8"
scriptLocale=locale.setlocale(category=locale.LC_ALL, locale="en_GB.UTF-8")

global thesaurus
thesaurus = dict()

def load_data():
	global thesaurus
	with open("test_corpus10k.txt", "r", encoding='utf-8', errors='ignore') as corpus:   #Chargement du corpus (nom du fichier potentiellement à changer)
		lines = corpus.readlines()

	for i in range(len(lines)):      #Pour chaque mot:
		splitline = lines[i].split("\t")	    #Colonnes séparées par des tabulations
		if (len(splitline)<10): continue	#Evite de prendre les lignes vides
		lemme = splitline[2]
		cat = splitline[3]
		thesaurus[(lemme, cat)] = thesaurus.get((lemme, cat), dict())	   #Si la clé (lemme, cat) n'existe pas encore dans le thésaurus, on la crée
		for j in range(-2, 3):					  #On parcourt les mots voisins
			if ((i+j<0) or (i+j>=len(lines)) or (j==0)):
				continue			#Evite la out of bounds error si i est en début/fin de corpus
			splitj = lines[i+j].split("\t")
			if (len(splitj) < 10):
				continue			#On ne s'intéresse au voisin que s'il ne s'agit pas d'une ligne vide (c'est mieux)
			voisin = splitj[2]				      #On stocke le lemme du mot voisin
			thesaurus[(lemme, cat)][(voisin, j)] = thesaurus[(lemme, cat)].get((voisin, j), 0) + 1      #On stocke dans l'entrée lemme/cat le nombre d'apparition du voisin à la même position

#computes thesaurus more than 2 times faster than load_data()
def load_data2(path):
	global thesaurus

	with open(path, "r", encoding='utf-8', errors='ignore') as corpus:   #Chargement du corpus (nom du fichier potentiellement à changer)
		lines = corpus.readlines()


	for i in range(len(lines)):      #Pour chaque mot:
		splitline = lines[i].split("\t")	    #Colonnes séparées par des tabulations
		if (len(splitline)<10): 
			continue	#Evite de prendre les lignes vides
		key = splitline[2]+splitline[3] #on ajoutera l'indice après

		thesaurus[key] = thesaurus.get(key, dict())	   #Si la clé (lemme, cat) n'existe pas encore dans le thésaurus, on la crée


		try:
			voisin1 = lines[i-2].split("\t")[2] + lines[i-2].split("\t")[3]			      #On stocke le lemme du mot voisin
			thesaurus[voisin1] = thesaurus.get(voisin1, dict())	   #Si la clé (lemme, cat) n'existe pas encore dans le thésaurus, on la crée
			thesaurus[key][voisin1] = thesaurus[key].get(voisin1, 0) + 1      #On stocke dans l'entrée lemme/cat le nombre d'apparition du voisin à la même position
			thesaurus[voisin1][key] = thesaurus[voisin1].get(key, 0) + 1      #On stocke dans l'entrée lemme/cat le nombre d'apparition du voisin à la même position
		except IndexError:
			continue

		try:
			voisin2 = lines[i-2].split("\t")[2] + lines[i-2].split("\t")[3]			      #On stocke le lemme du mot voisin
			thesaurus[voisin2] = thesaurus.get(voisin2, dict())	   #Si la clé (lemme, cat) n'existe pas encore dans le thésaurus, on la crée
			thesaurus[key][voisin2] = thesaurus[key].get(voisin2, 0) + 1      #On stocke dans l'entrée lemme/cat le nombre d'apparition du voisin à la même position
			thesaurus[voisin2][key] = thesaurus[voisin2].get(key, 0) + 1      #On stocke dans l'entrée lemme/cat le nombre d'apparition du voisin à la même position
		except IndexError:
			continue


#global vocab
WORD2IDX = dict()
#global rows
ROWS = list()
#global columns
COLS = list()
def load_data3(path):
	global WORD2IDX
	global ROWS
	global COLS
	with open(path, "r", encoding='utf-8', errors='ignore') as corpus:   #Chargement du corpus (nom du fichier potentiellement à changer)
		lines = corpus.readlines()


	for i in range(len(lines)):      #Pour chaque mot:
		splitline = lines[i].split("\t")	    #Colonnes séparées par des tabulations
		if (len(splitline)<10): 
			continue	#Evite de prendre les lignes vides

		key = splitline[2]+splitline[3] 
		WORD2IDX[key] = WORD2IDX.get(key, len(WORD2IDX.keys()) + 1)	   #Si la clé (lemme, cat) n'existe pas encore dans le thésaurus, on la crée
		row_index = WORD2IDX[key]

		try:
			col1_index = WORD2IDX[lines[i-1].split("\t")[2] + lines[i-1].split("\t")[3]]		      #On stocke le lemme du mot voisin
			ROWS.append(row_index)
			COLS.append(col1_index)
			ROWS.append(col1_index)
			COLS.append(row_index)
		except IndexError:
			continue

		try:
			col2_index = WORD2IDX[lines[i-2].split("\t")[2] + lines[i-2].split("\t")[3]]		      #On stocke le lemme du mot voisin
			ROWS.append(row_index)
			COLS.append(col2_index)
			ROWS.append(col2_index)
			COLS.append(row_index)
		except IndexError:
			continue

path = "C:\\Users\\garri\\Documents\\estrepublicain.a.outmalt.tar\\estrepublicain.a.outmalt"
for file in os.listdir(path):
	load_data3(f"{path}\\{file}")
	print(path+"\\"+file+ " DONE", len(WORD2IDX.keys()) , sys.getsizeof(WORD2IDX))
	break
	
DATA = np.ones(len(ROWS))
ROWS = np.array(ROWS,dtype = np.int32)
COLS = np.array(COLS,dtype= np.int32)
finalmatrix = coo_array((DATA, (ROWS, COLS)), dtype=np.int16)

			
			
"""
for key in thesaurus.keys():
	print("(" + key[0] + ", " + key[1] + ") : ")
	for keyInt in thesaurus[key].keys():
		print("\t(" + keyInt[0] + ", " + str(keyInt[1]) + ") : " + str(thesaurus[key][keyInt]))
"""

def make_vectors():
	"""
	mets à jour le thésaurus, un dictionnaire où:
	clé = tuple(mot, catégorie)
	valeur = vecteur de fréquence des contextes
	"""
	global thesaurus
	#liste des tokens (mot, position relative)
	vocab = set()
	for key in thesaurus.keys():
		for context_key in thesaurus[key]:
			vocab.add(context_key)
	temp_thes = dict()
	for key in thesaurus:
		temp_thes[key] = thesaurus[key]
	thesaurus = temp_thes
	#matrice diagonale
	diag_matrix = np.eye(len(vocab))

	#dictionnaire des indices de chaque token
	word2idx = {token: index for (index, token) in enumerate(vocab)} 

	#matrice de vecteurs(vides pour le moment) à renvoyer
	context_vectors = {token: np.zeros(len(vocab)) for token in thesaurus.keys()}

	#mise à jour des vecteurs
	for token in thesaurus:
		for context_token in thesaurus[token]:
			try:
				#màj
				context_vectors[token][word2idx[context_token]] = thesaurus[token][context_token] #decent for now 
			except KeyError:
				pass
	thesaurus = context_vectors

def cos_similarity(a,b):
	return np.dot(a,b)/( norm(a) * norm(b))

def better_cos_similarity(a,b):
	"""
	returns cosine similarity of SPARSE array with hopefully shorter time than cos_similarity
	a and b must be dictionnary where:
	key = word
	value = relative frequency
	"""
	scalar_product = 0
	for key in a:
		if key in b:
			scalar_product += a[key] * b[key]
	
	norm_a = 0
	norm_b = 0
	for val in a.values():
		norm_a += val*val
	for val in b.values():
		norm_b += val*val
	norm_a = np.sqrt(norm_a)
	norm_b = np.sqrt(norm_b)

	return scalar_product/ (norm_a*norm_b)



#fonction obligatoirement appelée, crée le dictionnaire de contextes
load_data()

#cette fonction doit être appelée quand on choisis la méthode avec les vecteurs, elle ne doit pas être appelée quand on traite directement le dictionnaire "thesaurus" car elle le modifie
#make_vectors()

#déclaration des listes sur lesquelles on va travailler
main_list = []
liste = list(thesaurus.keys())

#on parcourt la liste de clés du dictionnaire
for index, key in enumerate(liste):
	#on traite la clé key avec toutes celles qui suivent dans la liste: permet de comparer toutes les paires possibles
	for other_key in liste[index+1:]:
		#replace better_cos_similarity() with cos_similarity() for version with vectors
		sim = better_cos_similarity(thesaurus[key], thesaurus[other_key])
		#main_list va stocker toutes les paires du thésaurus, dans des tuples comprenant (première clé, deuxième clé, similarité entre première clé et deuxième clé)
		tup = (key, other_key, sim)
		main_list.append(tup)

#on print toutes les paires ayant une similarité significative
for tup in main_list:
	if tup[2] > 0.85 and tup[2]<1:
		print(tup)

#exemples de temps de calculs pour différents corpus avec la fonction better_cos_similarity et avec cos_similarity
#[100% of corpus_10K Finished in 15.7s], [10% of corpus_100K Finished in 174.6s],[100% of corpus_100K Finished in 465.6s] with better_cos_similarity
#[100% of corpus_10K Finished in 12.3s], [10% of corpus_100K Finished in 356.3s] [100% of corpus_100K Finished in 1856.8s]with cos_similarity
