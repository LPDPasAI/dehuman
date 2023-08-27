"""
 23/08/2023

@author Leandro Paolo De Persiis
"""

import spacy
from spacy import displacy
from spacy.cli import download
import re
import csv

nlp = spacy.load ("en_core_web_lg")

def normalization(text):
    text = text.replace(":-)", "smile")
    text = text.replace(":)", "smile")
    text = text.replace(":D", "smile")
    text = text.replace("b4", "before")
    text = text.replace("\r\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("nbsp;", " ")
    text = text.replace(" ", " ")
    text = text.replace("\xc2\xa0", " ")
    text = text.replace("xc2xa0", " ")
    # text = text.replace(",", " , ")
    # text = text.replace(".", " . ")
    # text = text.replace(":", " : ")
    # text = text.replace(";", " ; ")
    # text = text.replace("   ", " ")
    # text = text.replace("  ", " ")
    # text = text.replace("  ", " ")
    return text


def noise_removal(text):
    text = re.sub(r"[^A-Za-z0-9 àèéìòù.,:;?!()'\-]+", '', text)
    return text


def preprocess_text(text):
    text = text.lower()
    text = normalization(text)
    text = noise_removal(text)
    # print(text)

    return text

neg_adj = []
with open("NRC-VAD-Lexicon.txt", newline='', encoding='UTF-8', errors='ignore') as file_nrc:
    reader = csv.reader(file_nrc, delimiter='\t')
    for r in reader:
        if float(r[1]) < 0.4:
            neg_adj.append(r[0])

targets = ["lgbt", "lgb&t", "lgbtq", "lgbtqia", "lgbtq2", "lgbtq+", "lgbtiqapd", "glbt", "gay", "homosexual",
           "queer", "lesbian", "bisexual", "androgyne", "allosexual", "demisexual",
           "enby", "femme", "folx", "ftm", "neutral", "greygender", "greysexual", "heteroflexible", "homoaro",
           "intersectionality", "intersex", "misgender", "mtf", "non-binary", "panromantic", "pansexual", "passing",
           "polyamory", "polygender", "qpoc", "questioning", "straight", "terfs", "third-gender", "two-spirit",
           "trans", "transfeminine", "transgender", "transmasculine", "transition", "transvestite", "wimmin", "wlw",
           "womxn", "womyn", "zi", "hir"]


def essentialism_filter(testo):
    doc = preprocess_text(testo)
    lst_doc = doc.split(". ")
    essenzialiste = []
    possibili_essenz = []
    subj = ''
    for s in lst_doc:
        f = nlp(s)
        ag = []
        adj = ''
        aux = False
        post_aux = False
        adv = []
        noun = []
        for token in f:
            if len(adv) > 0:
                if token.pos_ == "ADJ" and (token.lemma_ in neg_adj or adv[0] in neg_adj):
                    ag.append("ADJN")
                    # if token.text in neg_adj:
                        # print(token.text)
                    # else:
                       # adv[0]
                else:
                    adv = []
            if aux == True:
                aux = False
                post_aux = True
                if token.pos_ == "ADJ" and token.lemma_ in neg_adj:
                    adj = token.text
                    ag.append("ADJN")
                    # print(token.text, "ADJ")
                elif token.pos_ == "ADV":
                    adv.append(token.lemma_)
                    # print(token.text, "ADV")
            else:
                if token.dep_ == 'nsubj':
                    subj = token.lemma_
                ag.append(token.pos_)
                if token.pos_ == "NOUN" and post_aux == False:
                    noun.append(token.lemma_)
                    # print(token.lemma_)
            # print(token.text, token.pos_, token.tag_, token.dep_)

            if token.pos_ == "AUX":
                aux = True
        if ("DET" in ag or "ADJ" in ag or "NOUN" in ag) and "AUX" in ag and "ADJN" in ag and subj in targets:
            possibili_essenz.append(s)
            # print(s, noun, ag)
            if len(noun) > 0 and noun[-1] in targets:
                print(s)
                print(ag)
                essenzialiste.append(s)
                print("---------------------")

    return essenzialiste, possibili_essenz


def essentialism_filter_social(da=0, num_max=3):
    essenzialiste_tot = []
    possibili_essenz_tot = []
    nome_file_txt = 'social-2019_2021_essentialist.txt'
    nome_file_txt2 = 'social-2019_2021_poss_essentialist.txt'
    print("Scrittura file:", nome_file_txt)
    files_tot = ['fb_2019_jan_may.csv', 'fb_2019_june_dec.csv', 'fb_2020_jan_may.csv', 'fb_2020_june_dec.csv', 'fb_2021_jan_may.csv', 'fb_2021_june_dec.csv', 'ig_2019.csv', 'ig_2020.csv', 'ig_2021.csv']

    csv.field_size_limit(131072 * 2) # impostiamo una lunghezza del campo maggiore
    files = files_tot[da:da+num_max]
    print("file in elaborazione:", files)
    with open(nome_file_txt, 'w', newline=None, encoding='UTF-8', errors='strict') as file_txt:
        with open(nome_file_txt2, 'w', newline=None, encoding='UTF-8', errors='strict') as file_txt2:
            for nome_file in files:
                with open(nome_file, 'r', newline=None, encoding='UTF-8', errors='ignore') as file_social:
                    reader = csv.DictReader(file_social)
                    for r in reader:
                        if r is not None and len(r)>2:
                            test = ""
                            if nome_file.startswith("fb_"):
                                testo = r['Message']
                            if nome_file.startswith("ig_"):
                                testo = r['Description'] + " " + r['Image Text']
                                #file_txt.write(preprocess_text(r['Description'] + " " + r['Image Text']) +  " \n ")
                            if len(testo) > 10:
                                essenzialiste, possibili_essenz = essentialism_filter(testo)
                                for f in essenzialiste:
                                    file_txt.write(f + " \n ")
                                for f in possibili_essenz:
                                    file_txt2.write(f + " \n ")
                                essenzialiste_tot.extend(essenzialiste)
                                possibili_essenz_tot.extend(possibili_essenz)
    print("Completato file " + nome_file_txt)
    return essenzialiste_tot, possibili_essenz_tot

essenzialiste, possibili_essenz = essentialism_filter_social(0, 9)
print(essenzialiste)

