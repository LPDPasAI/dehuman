"""
 28/11/23
 
@author Leandro Paolo De Persiis
"""

"""
 23/08/2023

@author Leandro Paolo De Persiis
"""
# ! pip install spacy
# !python -m spacy download en_core_web_lg

# essentialism_articles1983.log

import spacy
from spacy import displacy
from spacy.cli import download
import re
import csv
from tqdm import tqdm

nlp = spacy.load ("en_core_web_lg")

cosa_vuoi = "articles"
# nome_file = "prova"
nome_file = "new_articles-2018_2018"


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

targets = ['lgbt', 'lgb&t', 'lgbtq', 'lgbtqia', 'lgbtq2', 'lgbtq+', 'lgbtiqapd', 'glbt', 'gay', 'homosexual', 'queer',
           'lesbian', 'bisexual', 'androgyne', 'allosexual', 'demisexual', 'enby', 'folx', 'ftm', 'greygender',
           'greysexual', 'heteroflexible', 'homoaro', 'intersectionality', 'intersex', 'mtf', 'non-binary',
           'panromantic', 'pansexual', 'polyamory', 'polygender', 'qpoc', 'questioning', 'third-gender', 'two-spirit',
           'trans', 'transfeminine', 'transgender', 'transmasculine', 'transvestite', 'wimmin', 'wlw', 'womxn', 'womyn',
           'zi', 'hir', 'lgbts', 'lgb&ts', 'lgbtqs', 'lgbtqias', 'lgbtq2s', 'lgbtq+s', 'lgbtiqapds', 'glbts', 'gays',
           'homosexuals', 'queers', 'lesbians', 'bisexuals', 'androgynes', 'allosexuals', 'demisexuals', 'enbys',
           'folxs', 'ftms', 'greygenders', 'greysexuals', 'heteroflexibles', 'homoaros', 'intersectionalitys',
           'intersexs', 'mtfs', 'non-binarys', 'panromantics', 'pansexuals', 'passings', 'polyamorys', 'polygenders',
           'qpocs', 'questionings', 'third-genders', 'two-spirits', 'transs', 'transfeminines', 'transgenders',
           'transmasculines', 'transvestites', 'wimmins', 'wlws', 'womxns', 'womyns', 'zis', 'hirs', 'misgendered',
           'misgender', 'misgenders', 'terfs', 'homoromantic', 'biromantic', 'asexual', 'aromantic', 'nonbinary',
           'genderfluid', 'homoromantics', 'biromantics', 'asexuals', 'aromantics', 'nonbinarys', 'genderfluids',
           '-lgbt-']

neutri =['straight',
         'hetero',
         'heterosexual',
         'cisgender',
         'heteroromantic',
         'straights',
         'heteros',
         'heterosexuals',
         'cisgenders',
         'heteroromantics',
         '-neutral-']



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
        # if ("DET" in ag or "ADJ" in ag or "NOUN" in ag) and "AUX" in ag and "ADJN" in ag and subj in targets:
        if ("DET" in ag or "ADJ" in ag or "NOUN" in ag) and "AUX" in ag and "ADJ" in ag and subj in targets:
            possibili_essenz.append(s)
            # print(s, noun, ag)
            if len(noun) > 0 and noun[-1] in targets:
                print(s)
                print(ag)
                essenzialiste.append(s)
                print("---------------------")

    return essenzialiste, possibili_essenz


def essentialism_filter_sempl(testo):
    doc = preprocess_text(testo)
    # lst_doc = doc.split(". ")
    lst_doc = [i for i in nlp(doc).sents]
    numero_sentence = len(lst_doc)
#    print("numero frasi:" + str(numero_sentence))
    essenzialiste = []
    possibili_essenz = []
    essenz_neutre = []
    possib_essenz_neutre = []

    subj = ''
    num_con_target = 0
    num_con_neutre = 0
    for f in lst_doc:
        #f = nlp(s)
        al = ''
        tar_pres = False
        net_pres = False
        for token in f:
            if token.text in targets:
                al += "TAR-"
                tar_pres = True
            elif token.text in neutri:
                al += "TARNEU-"
                net_pres = True
            else:
                al += token.pos_ + "-"

            if token.text in neutri:
                net_pres = True

        if net_pres:
            num_con_neutre += 1
            if 'TARNEU-AUX-ADJ' in al or 'TARNEU-ADV-AUX-ADJ' in al or 'TARNEU-AUX-ADV-ADJ' in al or 'TARNEU-ADV-AUX-ADV-ADJ' in al:
                possib_essenz_neutre.append(str(f))
                if 'DET-TARNEU-AUX-ADJ' in al or 'DET-TARNEU-ADV-AUX-ADJ' in al or 'DET-TARNEU-AUX-ADV-ADJ' in al or 'DET-TARNEU-ADV-AUX-ADV-ADJ' in al:
                    essenz_neutre.append(str(f))
                print('')
                print(f)
                print(al)
                print("---------------------")

        if tar_pres:
            num_con_target += 1
            if 'TAR-AUX-ADJ' in al or 'TAR-ADV-AUX-ADJ' in al or 'TAR-AUX-ADV-ADJ' in al or 'TAR-ADV-AUX-ADV-ADJ' in al:
                possibili_essenz.append(str(f))
                if 'DET-TAR-AUX-ADJ' in al or 'DET-TAR-ADV-AUX-ADJ' in al or 'DET-TAR-AUX-ADV-ADJ' in al or 'DET-TAR-ADV-AUX-ADV-ADJ' in al:
                    essenzialiste.append(str(f))
                print('')
                print(f)
                print(al)
                print("---------------------")

    return essenzialiste, possibili_essenz, possib_essenz_neutre, essenz_neutre, numero_sentence, num_con_target, num_con_neutre


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
                            testo = ""
                            if nome_file.startswith("fb_"):
                                testo = r['Message']
                            if nome_file.startswith("ig_"):
                                testo = r['Description'] + " " + r['Image Text']
                                #file_txt.write(preprocess_text(r['Description'] + " " + r['Image Text']) +  " \n ")
                            sentences = [i for i in nlp(testo).sents]
                            num_sentences_tot += len(sentences)
                            for sentence in sentences:
                                if len(sentence) > 10:
                                    essenzialiste, possibili_essenz = essentialism_filter_sempl(sentence)
                                    for f in essenzialiste:
                                        file_txt.write(f + " \n ")
                                    for f in possibili_essenz:
                                        file_txt2.write(f + " \n ")
                                    essenzialiste_tot.extend(essenzialiste)
                                    possibili_essenz_tot.extend(possibili_essenz)
    print("Completato file " + nome_file_txt)
    return essenzialiste_tot, possibili_essenz_tot, num_sentences_tot


def essentialism_filter_articles(nome_file):
    essenzialiste_tot = []
    possibili_essenz_tot = []
    essenz_neut_tot = []
    possibili_essenz_neut_tot = []

    nome_file_txt = nome_file + '_new_essentialist.txt'
    nome_file_txt2 = nome_file + '_new_poss_essentialist.txt'
    nome_file_neut_txt = nome_file + '_neut_new_essentialist.txt'
    nome_file_neut_txt2 = nome_file + '_neut_new_poss_essentialist.txt'

    print("Scrittura file:", nome_file_txt)
    files = [nome_file + ".txt"]
    print("file in elaborazione:", files)
    num_sentences_tot= 0
    possib_essenz_neutre_tot = 0
    essenz_neutre_tot = 0
    numero_sentence_tot = 0
    num_con_target_tot = 0
    num_con_neutre_tot = 0
    with open(nome_file_txt, 'w', newline=None, encoding='UTF-8', errors='strict') as file_txt:
        with open(nome_file_txt2, 'w', newline=None, encoding='UTF-8', errors='strict') as file_txt2:
            with open(nome_file_neut_txt, 'w', newline=None, encoding='UTF-8', errors='strict') as file_neut_txt:
                with open(nome_file_neut_txt2, 'w', newline=None, encoding='UTF-8', errors='strict') as file_neut_txt2:
                    for nome_file in files:
                        with open(nome_file, 'r', newline=None, encoding='UTF-8', errors='ignore') as file_articles:
                            for articolo in tqdm(file_articles):
        #                        sentences = [i for i in nlp(articolo).sents]
        #                        num_sentences_tot += len(sentences)
        #                        for sentence in sentences:
        #                            sentence= str(sentence)
                                if articolo is not None and len(articolo)>2:
                                    # if len(sentence) > 10:
                                    essenzialiste, possibili_essenz, possib_essenz_neutre, essenz_neutre, numero_sentence, num_con_target, num_con_neutre = essentialism_filter_sempl(articolo)
                                    num_sentences_tot += numero_sentence
                                    num_con_target_tot += num_con_target
                                    num_con_neutre_tot += num_con_neutre
                                    for f in essenzialiste:
                                        file_txt.write(f + " \n ")
                                    file_txt.flush()
                                    for f in possibili_essenz:
                                        file_txt2.write(f + " \n ")
                                    file_txt2.flush()
                                    for f in essenz_neutre:
                                        file_neut_txt.write(f + " \n ")
                                    file_neut_txt.flush()
                                    for f in possib_essenz_neutre:
                                        file_neut_txt2.write(f + " \n ")
                                    file_neut_txt2.flush()

                                    essenzialiste_tot.extend(essenzialiste)
                                    possibili_essenz_tot.extend(possibili_essenz)
                                    essenz_neut_tot.extend(essenz_neutre)
                                    possibili_essenz_neut_tot.extend(possib_essenz_neutre)

                    proporzione_essenzialiste = len(essenzialiste_tot)/num_sentences_tot
                    proporzione_possibili_essenzialiste = len(possibili_essenz_tot)/num_sentences_tot
                    proporzione_essenzialiste_corr = len(essenzialiste_tot)/num_con_target_tot
                    proporzione_possibili_essenzialiste_corr = len(possibili_essenz_tot)/num_con_target_tot

                    proporzione_essenz_neutre = len(essenz_neut_tot)/num_sentences_tot
                    proporzione_possibili_essenz_neutre = len(possibili_essenz_neut_tot)/num_sentences_tot
                    proporzione_essenz_neutre_corr = len(essenz_neut_tot)/num_con_neutre_tot
                    proporzione_possibili_essenz_neutre_corr = len(possibili_essenz_neut_tot)/num_con_neutre_tot

                    file_txt.write("\n---------------------------- \n num_sentences_tot: " + str(num_sentences_tot))
                    file_txt.write("\n num_con_target_tot: " + str(num_con_target_tot))
                    file_txt.write("\n proporzione_essenzialiste: " + str(proporzione_essenzialiste))
                    file_txt.write("\n proporzione_essenzialiste_corr: " + str(proporzione_essenzialiste_corr))

                    file_txt2.write("\n---------------------------- \n num_sentences_tot: " + str(num_sentences_tot))
                    file_txt2.write("\n num_con_target_tot: " + str(num_con_target_tot))
                    file_txt2.write("\n proporzione_possibili_essenzialiste: " + str(proporzione_possibili_essenzialiste))
                    file_txt2.write("\n proporzione_possibili_essenzialiste_corr: " + str(proporzione_possibili_essenzialiste_corr))

                    file_neut_txt.write("\n---------------------------- \n num_sentences_tot: " + str(num_sentences_tot))
                    file_neut_txt.write("\n num_con_neutre_tot: " + str(num_con_neutre_tot))
                    file_neut_txt.write("\n proporzione_essenz_neutre: " + str(proporzione_essenz_neutre))
                    file_neut_txt.write("\n proporzione_essenz_neutre_corr: " + str(proporzione_essenz_neutre_corr))

                    file_neut_txt2.write("\n---------------------------- \n num_sentences_tot: " + str(num_sentences_tot))
                    file_neut_txt2.write("\n num_con_neutre: " + str(num_con_neutre_tot))
                    file_neut_txt2.write("\n proporzione_possibili_essenz_neutre: " + str(proporzione_possibili_essenz_neutre))
                    file_neut_txt2.write("\n proporzione_possibili_essenz_neutre_corr: " + str(proporzione_possibili_essenz_neutre_corr))

    print("Completato file " + nome_file_txt)
    return essenzialiste_tot, possibili_essenz_tot, num_sentences_tot

essenzialiste = ""
possibili_essenz = ""
num_sentences_tot = 0
if cosa_vuoi == "articles":
    essenzialiste, possibili_essenz, num_sentences_tot = essentialism_filter_articles(nome_file)

else:
    essenzialiste, possibili_essenz, num_sentences_tot = essentialism_filter_social(0,9)

print(essenzialiste)
print(possibili_essenz)

proporzione_essenzialiste = len(essenzialiste)/num_sentences_tot
proporzione_possibili_essenzialiste = len(possibili_essenz)/num_sentences_tot

print(proporzione_essenzialiste, proporzione_possibili_essenzialiste)
