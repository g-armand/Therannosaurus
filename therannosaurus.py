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
	with open("corpus_10k.txt", "r", encoding='utf-8', errors='ignore') as corpus:   #Chargement du corpus (nom du fichier potentiellement à changer)
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




load_data()

#uncomment for version with vectors
#make_vectors()

main_list = []
liste = list(thesaurus.keys())
for index, key in enumerate(liste):
	for other_key in liste[index+1:]:
		#replace better_cos_similarity() with cos_similarity() for version with vectors
		sim = better_cos_similarity(thesaurus[key], thesaurus[other_key])
		tup = (key, other_key, sim)
		main_list.append(tup)

#[100% of corpus_10K Finished in 15.7s], [10% of corpus_100K Finished in 174.6s],[100% of corpus_100K Finished in 465.6s] with better_cos_similarity
#[100% of corpus_10K Finished in 12.3s], [10% of corpus_100K Finished in 356.3s] [100% of corpus_100K Finished in 1856.8s]with normal cos_similarity
