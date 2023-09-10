"""
 10/09/2023

@author Leandro Paolo De Persiis

Si parte da un embeding e da due liste di parole, si misurano le distanze di ogni elemento della prima lista con ciascun elemento della seconda lista.
L'embedding deve essere in un file di testo in cui il primo elemento è la parola ed i successivi i valori del suo vettore.
I risultati vengono salvati su un file csv in cui le colonne sono: primo elemento, secondo elemento coseno dell'angolo tra i due vettori che rappresentano quelle due parole.
"""

import numpy as np
import os
import csv
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


MY_EMBEDDING_PATH = "C:\\Users\\lpdepersiis\\PycharmProjects\\embedding_analysis\\dehuman\\embedding_2019_2021_with_glove2.txt"
# MY_EMBEDDING_PATH = "C:\\Users\\lpdepersiis\\PycharmProjects\\embedding_analysis\\dehuman\\embedding_2018_2022_with_glove.txt"
# MY_EMBEDDING_PATH = "C:\\Users\\lpdepersiis\\PycharmProjects\\autoencoderNlp\\embedding\\en\\glove\\glove.6B.100d.txt"

print(MY_EMBEDDING_PATH)

object_animals = ["animal", "dog", "pig", "vermin"]

targets = ["lgbt", "lgb&t", "lgbtq", "lgbtqia", "lgbtq2", "lgbtq+", "lgbtiqapd", "glbt", "gay", "homosexual",
           "queer", "lesbian", "bisexual", "androgyne", "allosexual", "demisexual",
           "enby", "femme", "folx", "ftm", "neutral", "greygender", "greysexual", "heteroflexible", "homoaro",
           "intersectionality", "intersex", "misgender", "mtf", "non-binary", "panromantic", "pansexual", "passing",
           "polyamory", "polygender", "qpoc", "questioning", "straight", "terfs", "third-gender", "two-spirit",
           "trans", "transfeminine", "transgender", "transmasculine", "transition", "transvestite", "wimmin", "wlw",
           "womxn", "womyn", "zi", "hir"]


print("\nCaricamento embedding.\n Attendere...\n")

model = KeyedVectors.load_word2vec_format(MY_EMBEDDING_PATH, binary=False, no_header=True, encoding='utf-8', unicode_errors='ignore')

def cos_similarity(word1, word2):
    v1 = model[word1]
    v2 = model[word2]
    cos = np.dot(v1, v2) /( np.sqrt(np.dot(v1, v1)) *  np.sqrt(np.dot(v2, v2)) )
    return cos

print("   Eliminazione parole non presenti    ")
non_presenti = []
for t in targets:
    try:
        v = model[t]
        if set(v) == {0.}:
            non_presenti.append(t)
            print("Rimosso", t, v)

    except:
        non_presenti.append(t)
        print("Rimosso", t)

for t in non_presenti:
    targets.remove(t)

print(targets)

non_presenti = []
for o in object_animals:
    try:
        v = model[o]
        if set(v) == {0.}:
            non_presenti.append(o)
            print("Rimosso", o, v)

    except:
        non_presenti.append(o)
        print("Rimosso", o)

for o in non_presenti:
    object_animals.remove(o)

print(object_animals)

print("   Ricerca vicinanza vettori    ")

with open("similarity_target.csv", "w", newline='', encoding='UTF-8', errors='ignore') as file:
    writer = csv.writer(file)
    for o in object_animals:
        for t in targets:
            similarity = cos_similarity(o, t)
            writer.writerow([o, t, similarity])
            print(o, t, similarity)


