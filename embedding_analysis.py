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

MY_EMBEDDING_PATH = "C:\\Users\\lpdepersiis\\PycharmProjects\\embedding_analysis\\dehuman\\embedding_2019_2021_with_glove2.txt"
# MY_EMBEDDING_PATH = "C:\\Users\\lpdepersiis\\PycharmProjects\\embedding_analysis\\dehuman\\embedding_2018_2022_with_glove.txt"
# MY_EMBEDDING_PATH = "C:\\Users\\lpdepersiis\\PycharmProjects\\autoencoderNlp\\embedding\\en\\glove\\glove.6B.100d.txt"
PATH_LEXICON = "NRC-VAD-Lexicon.txt"
limite_d = 0.01
num_parole = 100

print(MY_EMBEDDING_PATH)

print("\nCaricamento embedding.\n Attendere...\n")

model = KeyedVectors.load_word2vec_format(MY_EMBEDDING_PATH, binary=False, no_header=True, encoding='utf-8', unicode_errors='ignore')

def get_similar(word, model=model, limite=0.5, num_word=500):
    print(f'Parole più simili a {word}: ')
    vicine = model.most_similar(word, topn=num_word)
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
    v_pr = 0
    a_pr = 0
    d_pr = 0

    n = 0
    t = 0
    tr = 0
    sum_num = int((len(words) + 1) * len(words) / 2)
    pos = 0
    for w in words:
        pos += 1
        print(w)
        vad = lexicon.get(w[0])
        print(n, "vad:", vad)
        if vad != None and len(vad) > 0:
            n += 1
            t += w[1]
            v += vad[0]
            a += vad[1]
            d += vad[2]

            v_p += vad[0] * w[1]
            a_p += vad[1] * w[1]
            d_p += vad[2] * w[1]

            rank_ratio = (len(words) - pos) / sum_num
            rank_ratio = (w[1] + rank_ratio) / 2
            tr += rank_ratio
            print(rank_ratio, tr)
            v_pr += vad[0] * rank_ratio
            a_pr += vad[1] * rank_ratio
            d_pr += vad[2] * rank_ratio
    if n> 0 and t > 0 and tr > 0:
        v = v / n
        a = a / n
        d = d / n
        v_p = v_p / t
        a_p = a_p / t
        d_p = d_p / t

        v_pr = v_pr / tr
        a_pr = a_pr / tr
        d_pr = d_pr / tr
        #print("n:", n)
        #print("t:", t)
        print("valence:", v)
        print("arousal:", a)
        print("dominance:", d)
        print("valence weighed:", v_p)
        print("arousal weighed:", a_p)
        print("dominance weighed:", d_p)
        print("valence rank weighed:", v_pr)
        print("arousal rank weighed:", a_pr)
        print("dominance rank weighed:", d_pr)
        return v, a, d, v_p, a_p, d_p, v_pr, a_pr, d_pr
    else:
        print(f"Valutazione non possibile per la parola indicata, in quanto nessuna delle parole vicine trovate è presente nel lexicon ")
        return 0,0,0,0,0,0,0,0,0

print("          Ricerca parole vicine     ")
print(f"\nVengono mostrate le {num_parole} parole con vettore più vicino a quello della parola indicata ")

print(f"Ora viene effettuato in automatico il calcolo sulle parole della lista label ")

labels = ['lgbt', 'lgbtq', 'lgbtqia', 'lgbtq2', 'glbt', 'gay', 'homosexual', 'queer', 'lesbian', 'bisexual',
         'androgyne', 'demisexual', 'enby', 'femme', 'folx', 'ftm', 'neutral', 'heteroflexible', 'intersectionality',
         'intersex', 'misgender', 'mtf', 'non-binary', 'panromantic', 'pansexual', 'passing', 'polyamory', 'qpoc',
         'questioning', 'straight', 'terfs', 'third-gender', 'two-spirit', 'trans', 'transfeminine', 'transgender',
         'transmasculine', 'transition', 'transvestite', 'wlw', 'womxn', 'womyn', 'zi', 'hir']

print("   Eliminazione parole della lista label non presenti nell'embedding   ")
non_presenti = []
for t in labels:
    try:
        v = model[t]
        if set(v) == {0.}:
            non_presenti.append(t)
            print("Rimosso", t, v)

    except:
        non_presenti.append(t)
        print("Rimosso", t)

for t in non_presenti:
    labels.remove(t)

print("Parole rimaste:", labels)

nome_file_csv_medie = "lexicon_means_target.csv"
with open(nome_file_csv_medie, "w", newline='', encoding='UTF-8', errors='ignore') as file:
    writer = csv.writer(file)
    writer.writerow(["word", "valence", "arousal", "dominance", "valence_weighed", "arousal_weighed", "dominance_weighed", "valence_rank_weighed", "arousal_rank_weighed", "dominance_rank_weighed"])
    for p in labels:
        similar = get_similar(p, limite=limite_d, num_word=num_parole)
        v, a, d, v_p, a_p, d_p, v_pr, a_pr, d_pr = vad_mean(lexicon, similar)
        if(set([v, a, d, v_p, a_p, d_p, v_pr, a_pr, d_pr])!={0}):
            writer.writerow([p, v, a, d, v_p, a_p, d_p, v_pr, a_pr, d_pr])

print("\nCompletata scrittura del file", nome_file_csv_medie)

print("\nPer effettuare le stesse valutazioni su altre parole ")
print("scrivere la parola per la quale si cercano i vettori più vicini ")
print("Rispondere con - per terminare")

richiesta = '+'

while richiesta != '-':
    richiesta = input("\nParola: ")
    try:
        if richiesta == '-':
            print("\n  End")
            break
        elif richiesta == 'limite' or richiesta == '<limite>':
            limite_d = float( input("Nuovo limite: "))
        else:
            similar = get_similar(richiesta, limite=limite_d, num_word=num_parole)
            vad_mean(lexicon, similar)

    except Exception as e:
        print(e)
