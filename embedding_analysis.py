"""
 04/08/2023

@author Leandro Paolo De Persiis

Viene utilizzato gensim per analizzare lo spazio vettoriale dell'embedding passato.
L'embedding deve essere in un file di testo in cui il primo elemento è la parola ed i successivi i valori del suo vettore.
Una volta caricato dal path indicato nella variabile globale MY_EMBEDDING_PATH, vengono mostrati i vettori più vicini ad alcune parole di esempio.
Di ogni parola vengono mostrate solo quelle con vettori più vicini, in cui l coseno del loro angolo sia superiore ad una certa soglia.
Nei primi esempi la soglia di è di 0.5 in una scala che va da 0 (completamente ortogonali) a 1 (perfetta identità).
Nel caso in cui non ci sia nessuna parola che superi la soglia, viene comunque indicata la più vicina.
In seguito inizia un ciclo in cui viene data la possibilità di indicare altre parole con le quali cercare vicinanze.
La soglia di partenza è 0.4, ma sarà possibile modificarla rispondendo <limite>
ed impostando quindi il nuovo limite
"""

import numpy as np
import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

MY_EMBEDDING_PATH = "C:\\Users\\lpdepersiis\\PycharmProjects\\embedding_analysis\\dehuman\\embedding_new.txt"

print(MY_EMBEDDING_PATH)

print("\nCaricamento embedding.\n Attendere...\n")

model = KeyedVectors.load_word2vec_format(MY_EMBEDDING_PATH, binary=False, no_header=True)

def get_similar(word, model=model, limite=0.5):
    print(f'Parole più simili a {word}: ')
    vicine = model.most_similar(word, 20)
    if vicine[0][1] <= limite: # stampa almeno la prima
        print(vicine[0][0], vicine[0][1])
    for p in vicine:
        if p[1] > limite:
            print(p[0], p[1])


print("          Ricerca parole vicine     ")
print("\nVengono mostrate le parole con vettore più vicino a quello della parola indicata ")
print("mostrando solo quelli con una vicinanza (in base al coseno del loro angolo) superiore ad una certa soglia. ")
print("In questi esempi abbiamo impostato la soglia di 0.5 in una scala che va da 0 (completamente ortogonali) a 1 (perfetta identità).  ")
print("Nel caso in cui non ci sia nessuna parola che superi la soglia, viene comunque indicata la più vicina.")
print("\nEsempi: ")

get_similar("wasp")
print("")

get_similar("psychology")
print("")

get_similar("bird")
print("")

print("Ora verrà data la possibilità di indicare altre parole con le quali cercare similitudini.")
print("La soglia di partenza è 0.4, ma sarà possibile modificarla rispondendo <limite>")
print("ed impostando quindi il nuovo limite")
print("\nIndicare la parola per la quale si cercano similitudini")
print("Rispondere con - per terminare")

risposta = '+'
limite_d = 0.4
while risposta != '-':
    risposta = input("\nParola: ")
    try:
        if risposta == '-':
            print("\n  Ciao")
            break
        elif risposta == 'limite' or risposta == '<limite>':
            limite_d = float( input("Nuovo limite: "))
        else:
            get_similar(risposta, limite=limite_d)
    except Exception as e:
        print(e)
