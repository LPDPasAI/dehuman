"""
 04/08/2023

@author Leandro Paolo De Persiis

Viene utilizzato gensim per analizzare lo spazio vettoriale dell'embedding passato.
L'embedding deve essere in un file di testo in cui il primo elemento è la parola ed i successivi i valori del suo vettore.
Di ogni parola vengono mostrate le 500 parole con vettori più vicini e vengono calcolate le medie di valence, arousal e dominance
di queste 500 parole in base ai valori del lexicon realizzato dal Dr. Saif M. Mohammad presso il National Research Council Canada (NRC)
Le medie vengono calcolate sia semplici che pesate in base alla vicinanza delle parole.
"""

import numpy as np
import os
import csv
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

MY_EMBEDDING_PATH = "C:\\Users\\lpdepersiis\\PycharmProjects\\embedding_analysis\\dehuman\\embedding_2018_2022_with_glove.txt"
# MY_EMBEDDING_PATH = "C:\\Users\\lpdepersiis\\PycharmProjects\\autoencoderNlp\\embedding\\en\\glove\\glove.6B.100d.txt"
PATH_LEXICON = "NRC-VAD-Lexicon.txt"

print(MY_EMBEDDING_PATH)

print("\nCaricamento embedding.\n Attendere...\n")

model = KeyedVectors.load_word2vec_format(MY_EMBEDDING_PATH, binary=False, no_header=True, encoding='utf-8', unicode_errors='ignore')

def get_similar(word, model=model, limite=0.5):
    print(f'Parole più simili a {word}: ')
    vicine = model.most_similar(word, topn=500)
    if vicine[0][1] <= limite: # stampa almeno la prima
        print(vicine[0][0], vicine[0][1])
    res = []
    for p in vicine:
        if p[1] > limite:
            p_v = (p[0], p[1])
            # print(p_v)
            res.append(p_v)
    return res


lexicon = {}
with open(PATH_LEXICON, 'r', newline='') as file_lexicon:
    reader = csv.reader(file_lexicon, delimiter='\t')
    n = 0
    for l in reader:
        n += 1
        vad = [float(l[1]), float(l[2]), float(l[3])]
        lexicon[l[0]] = vad
        # print(l)
    print(n)


def vad_mean(lexicon, words: list):
    v = 0
    a = 0
    d = 0
    v_p = 0
    a_p = 0
    d_p = 0

    n = 0
    t = 0
    for w in words:
        print(w)
        vad = lexicon.get(w[0])
        print("vad:", vad)
        if vad != None and len(vad) > 0:
            n += 1
            t += w[1]
            v += vad[0]
            a += vad[1]
            d += vad[2]

            v_p += vad[0] * w[1]
            a_p += vad[1] * w[1]
            d_p += vad[2] * w[1]
    v = v / n
    a = a / n
    d = d / n
    v_p = v_p / t
    a_p = a_p / t
    d_p = d_p / t

    #print("n:", n)
    #print("t:", t)
    print("valence:", v)
    print("arousal:", a)
    print("dominance:", d)
    print("valence weighed:", v_p)
    print("arousal weighed:", a_p)
    print("dominance weighed:", d_p)


print("          Ricerca parole vicine     ")
print("\nVengono mostrate le 500 parole con vettore più vicino a quello della parola indicata ")
print("\nIndicare la parola per la quale si cercano i vettori più vicini ")
print("Rispondere con - per terminare")

richiesta = '+'
limite_d = 0.01
while richiesta != '-':
    richiesta = input("\nParola: ")
    try:
        if richiesta == '-':
            print("\n  End")
            break
        elif richiesta == 'limite' or richiesta == '<limite>':
            limite_d = float( input("Nuovo limite: "))
        else:
            similar = get_similar(richiesta, limite=limite_d)
            vad_mean(lexicon, similar)

    except Exception as e:
        print(e)
