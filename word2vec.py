"""
 31/07/2023

@author Leandro Paolo De Persiis

"""
import time
import datetime
ms_iniz = time.time() * 1000
ms_prec = 0

import io
import re
import string
import tqdm
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import csv

print(tf.version.VERSION)

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
EMBEDDING_SIZE = 100
MY_EMBEDDING_PATH = "C:\\Users\\lpdepersiis\\PycharmProjects\\autoencoderNlp\\embedding\\en\\glove\\"
WITH_GLOVE = True  # se impostato a True utilizza come base l'embedding glove

YEARS_FROM = 2019
YEARS_TO = 2021
# I due parametri seguenti determinano l'origine dei dati
ARTICLES_SOCIAL = 'SOCIAL' # Con SOCIAL oppure ARTICLES viene prima creato un file a partire da queste raccolte. YEARS_FROM e YEARS_TO sono utilizzati solo se si imposta ARTICLES
# se ARTICLES_SOCIAL è impostato a FILE, non verrà creato un nuovo file in base a SOCIAL o ARTICLES, ma utilizzato questo file direttamente
FILE = 'social-2019_2021.txt' # Impostando un file non viene fatta la creazione di un file in base a SOCIAL o ARTICLES, ma viene utilizzato questo file direttamente

vocab_size = 100000 # vocabulary size
sequence_length = 10 # number of words in a sequence.
num_ns = 4 # number of negative samples per positive context

context_class = tf.reshape(tf.constant(2, dtype="int64"), (1, 1))
negative_sampling_candidates = 2
sampling_table = 0

def print_time(ms_iniziali=ms_iniz, ms_az_prec=ms_prec):
    ms = round((time.time()*1000) - ms_iniziali)
    ms_pr = round(ms - ms_az_prec)
    print(datetime.datetime.now(), "dall'inizio", ms, "millisecondi,", "dalla stampa precedente", ms_pr, "millisecondi")
    return ms

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    print("Numero di sequences:", len(sequences))
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
    context_class = tf.constant([1], dtype="int64")
    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):
        # print("*************** seq:", sequence)
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name="negative_sampling")

        # Build context and label vectors (for one target word)
        try:
            context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")
            # print("############## target:", target_word, " context:", context)
            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
        except Exception as e:
            try:
                print("Exception " + str(e))
            except:
                print("Exception ")

    return targets, contexts, labels

def get_word_matrix(nome_file_embedding=None):
    """
    Questa funzione ci serve per crearci un dizionario avente come indice la parola e come valore il vettore dell'embedding corrispondente
    """
    if nome_file_embedding==None:
        nome_file_embedding = MY_EMBEDDING_PATH + 'glove.6B.'+str(EMBEDDING_SIZE)+'d.txt'
    print("nome file embedding di partenza:", nome_file_embedding)
    word_matrix = {}
    with open(nome_file_embedding, 'r', encoding='UTF-8') as file_emb:
        for row in file_emb: # leggo ogni riga del file di testo contenente l'embedding
            row = row.split() # la divido nei suoi elementi
            try:
                word_matrix[row[0]] = np.array(row[1:], dtype='float32') # il primo è la parola e sarà l'indice di questa voce, gli altri andranno a formare il vettore
            except Exception as e:
                print(row)
                print(e)
    return word_matrix


def get_embedding_matrix(embeddings_index, word_index, dim_embeddings=EMBEDDING_SIZE):
    """
    Tramite questa funzione creiamo una matrice in cui le righe siano nello stesso ordine dell'indice ottenuto dal tokenizer
    e che contenga solo i vettori relativi alle parole in esso contenute

    :param embeddings_index: il dizionario, ottenuto dall'embedding, avente le parole come indice ed i vettori come valore
    :param word_index:  La lista di parole ottenuta dal tokenizer
    :param dim_embeddings: la lunghezza dei vettori dell'embedding che stiamo utilizzando
    :return: la matrice dei vettori dell'embedding ordinata come il nostro indice
    """
    embedding_matrix = np.zeros((len(word_index) + 1,
                                 dim_embeddings))  # creiamo la matrice di zeri avente tante righe quante sono le parole (più una) e tante colonne quante sono quelle dei vettori
    print(embedding_matrix.shape)
    num = 0
    for word in word_index:  # Scorriamo le parole dell'indice del tokenizer
        embedding_vector = embeddings_index.get(word)  # estraiamo il vettore corrispondente
        if embedding_vector is not None:  # verifichiamo che esista (anche se il nostro dizionario è più piccolo di quello dell'embedding potrebbe contenere parole non presenti in esso)
            # se la parola è presente andiamo avanti (se non è presente, in corrispondenza di questo indice, rimarrà il vettore formato da zeri)
            embedding_matrix[num] = embedding_vector  # impostiamo nella matrice quella riga con il vettore corrispondente alla parola
        num += 1
    return embedding_matrix


def get_articles(anno_inizio, anno_fine):
    anni = range(anno_inizio, anno_fine+1)
    mesi = range(1, 13)
    nome_file_txt = 'articles-'+str(anni[0])+ '_' +str(anni[-1])+'.txt'
    print("Scrittura file:", nome_file_txt)
    csv.field_size_limit(131072 * 2) # impostiamo una lunghezza del campo maggiore, perché alcuni articoli superano la lunghezza di default
    with open(nome_file_txt, 'w', newline=None, encoding='UTF-8', errors='strict') as file_txt:
        for a in anni:
            for m in mesi:
                nome_file = str(a) + '-' + str(m) + '-articles'
                with open(nome_file + ".csv", 'r', newline=None, encoding='UTF-8', errors='ignore') as file_nyt:
                    reader = csv.reader(file_nyt)
                    for r in reader:
                        if r is not None and len(r)>2 and len(r[1]) > 50:
                            file_txt.write(preprocess_text(r[1]) + " \n ")
    print("Completato file " + nome_file_txt)
    return nome_file_txt


def get_social(da=0, num_max=3):
    """

    :param da: Indica l'indice del file da cui partire
    :param num_max: indica il numero massimo di file da elaborare
    :return:
    """
    nome_file_txt = 'social-2019_2021_'+str(da)+'_'+str(num_max)+'.txt'
    print("Scrittura file:", nome_file_txt)
    files_tot = ['fb_2019_jan_may.csv', 'fb_2019_june_dec.csv', 'fb_2020_jan_may.csv', 'fb_2020_june_dec.csv', 'fb_2021_jan_may.csv', 'fb_2021_june_dec.csv', 'ig_2019.csv', 'ig_2020.csv', 'ig_2021.csv']
    # files = ['fb_2019_jan_may.csv', 'fb_2019_june_dec.csv', 'fb_2020_jan_may.csv', 'fb_2020_june_dec.csv', 'fb_2021_jan_may.csv', 'fb_2021_june_dec.csv']
    # files = ['ig_2019.csv', 'ig_2020.csv', 'ig_2021.csv']

    csv.field_size_limit(131072 * 2) # impostiamo una lunghezza del campo maggiore
    files = files_tot[da:da+num_max]
    print("file in elaborazione:", files)
    with open(nome_file_txt, 'w', newline=None, encoding='UTF-8', errors='strict') as file_txt:
        for nome_file in files:
            with open(nome_file, 'r', newline=None, encoding='UTF-8', errors='ignore') as file_social:
                reader = csv.DictReader(file_social)
                for r in reader:
                    if r is not None and len(r)>2:
                        if nome_file.startswith("fb_"):
                            file_txt.write(preprocess_text(r['Message']) + " \n ")
                        if nome_file.startswith("ig_"):
                            file_txt.write(preprocess_text(r['Description'] + " " + r['Image Text']) +  " \n ")
    print("Completato file " + nome_file_txt)
    return nome_file_txt

def noise_removal(text):
    text = re.sub(r"[^A-Za-z0-9 àèéìòù.,:;?!()'\-]+", '', text)
    return text

def normalization(text):
    text = text.replace(":-)", "smile")
    text = text.replace(":)", "smile")
    text = text.replace(":D", "lol")
    text = text.replace("b4", "before")
    text = text.replace("\r\n", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.replace("nbsp;", " ")
    text = text.replace(" ", " ")
    text = text.replace("\xc2\xa0", " ")
    text = text.replace("xc2xa0", " ")
    text = text.replace(",", " , ")
    text = text.replace(".", " . ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace(":", " : ")
    text = text.replace(";", " ; ")
    text = text.replace("   ", " ")
    text = text.replace("  ", " ")
    text = text.replace("  ", " ")
    return text


def preprocess_text(text):
    text = text.lower()
    text = normalization(text)
    text = noise_removal(text)
    return text


ms_prec = print_time()
nome_file_txt = ''
if ARTICLES_SOCIAL == 'SOCIAL':
    nome_file_txt = get_social(num_max=9)
elif ARTICLES_SOCIAL == 'ARTICLES':
    nome_file_txt = get_articles(YEARS_FROM, YEARS_TO)
else:
    nome_file_txt = FILE

text_ds = tf.data.TextLineDataset(nome_file_txt).filter(lambda x: tf.cast(tf.strings.length(x), bool))
ms_prec = print_time()

# Now, create a custom standardization function to lowercase the text and
# remove punctuation.
def custom_standardization(input_data):
    lowercase = input_data.lower()
    lowercase = lowercase.replace("nbsp;", " ")
    lowercase = lowercase.replace(" ", " ")
    lowercase = lowercase.replace("\xc2\xa0", " ")
    lowercase = lowercase.replace("xc2xa0", " ")
    lowercase = lowercase.replace("￼", " ")
    lowercase = tf.strings.lower(lowercase)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')


class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix):
        super(Word2Vec, self).__init__()
        self.target_embedding = self.getEmbeddingLayer(vocab_size, embedding_dim, "w2v_embedding", 1, embedding_matrix=embedding_matrix, t='T')
        self.context_embedding = self.getEmbeddingLayer(vocab_size, embedding_dim, "context_embedding", num_ns+1, embedding_matrix=embedding_matrix, t='C')

    def getEmbeddingLayer(self, vocab_size, embedding_dim, name, input_length, embedding_matrix=None, t='C'):
        if WITH_GLOVE and t=='T':
            embedding_layer = layers.Embedding(vocab_size +1,
                                          embedding_dim,
                                          input_length=input_length,
                                          weights=[embedding_matrix],
                                          name=name)
        else:
            embedding_layer = layers.Embedding(vocab_size +1,
                                          embedding_dim,
                                          input_length=input_length,
                                          name=name)
        return embedding_layer

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots


# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the
# same length.
vectorize_layer = layers.TextVectorization(
    standardize=None,  #'lower_and_strip_punctuation', # custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    encoding='utf-8',
    vocabulary=None,
    output_sequence_length=sequence_length)

print(type(vectorize_layer))
print(vectorize_layer)

vectorize_layer.adapt(text_ds.batch(BATCH_SIZE))
print("vocabulary_size", vectorize_layer.vocabulary_size())

inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])
with open("inverse_vocab.txt", 'w', encoding='utf-8') as voc:
    for v in inverse_vocab:
        voc.write(v + "\n")

print("len(inverse_vocab))", len(inverse_vocab))


def crea_embedding(text_ds=text_ds, vectorize_layer=vectorize_layer, inverse_vocab=inverse_vocab, vocab_size=vocab_size,
                   embedding_dim=EMBEDDING_SIZE, da=0, num_max=9, nome_embedding_prec=None):
    nome_file_embedding = None
    passaggio = ''
    if ARTICLES_SOCIAL != 'ARTICLES':
        passaggio = da
        nome_file_txt = get_social(da=da, num_max=num_max)  #
        text_ds = tf.data.TextLineDataset(nome_file_txt).filter(lambda x: tf.cast(tf.strings.length(x), bool))

    text_vector_ds = text_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

    sequences = list(text_vector_ds.as_numpy_iterator())
    print(len(sequences))

    for seq in sequences[:5]:
        print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=2,
        num_ns=4,
        vocab_size=vocab_size,
        seed=SEED)

    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)

    print('\n')
    print(f"targets.shape: {targets.shape}")
    print(f"contexts.shape: {contexts.shape}")
    print(f"labels.shape: {labels.shape}")

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    print("dataset pre:", dataset)

    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    print("dataset post:", dataset)
    try:
        print("cardinality dataset:", dataset.cardinality())
        print("__sizeof__ dataset:", dataset.__sizeof__())
        print("__len__ dataset:", dataset.__len__())
        print("len dataset:", len(dataset))
    except Exception as e:
        print(e)
    embedding_matrix = []
    if WITH_GLOVE:
        word_matrix = get_word_matrix(nome_embedding_prec)
        embedding_matrix = get_embedding_matrix(word_matrix, inverse_vocab)
        print("embedding_matrix:", embedding_matrix)
        print(embedding_matrix[:3])

    word2vec = Word2Vec(vocab_size, embedding_dim, embedding_matrix)
    word2vec.compile(optimizer='adam',
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    word2vec.fit(dataset, epochs=25) #, callbacks=[tensorboard_callback])

    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    emb_with_glove = ''
    if WITH_GLOVE:
        emb_with_glove = 'with_glove_'
    nome_embedding = 'embedding_'+str(YEARS_FROM)+ '_' +str(YEARS_TO)+ '_' + emb_with_glove+ str(passaggio) +'.txt'
    with open(nome_embedding, 'w', encoding='utf-8') as emb:
        with open("log_embedding.log", 'w', encoding='utf-8') as log:
            for index, word in enumerate(vocab):
                log.write(str(index) + ' - ' + word+ "\n")
                if index == 0:
                    continue  # skip 0, it's padding.
                vec = weights[index]
                vec_str = ' '.join([str(x) for x in vec])
                emb.write(word + ' ' + vec_str + "\n")

    print("Completato salvataggio dell'embedding", nome_embedding)
    return nome_embedding

if ARTICLES_SOCIAL != 'ARTICLES':
    ms_prec = print_time()
    nome_emb_prec = None
    inverse_vocabolary = inverse_vocab
    num_files = 1
    for passaggio in range(0, 9, num_files):
        print("passaggio", passaggio)
        ms_prec = print_time()
        vectorize_layer = layers.TextVectorization(
            standardize=None,  # 'lower_and_strip_punctuation',  # custom_standardization,
            max_tokens=vocab_size,
            output_mode='int',
            encoding='utf-8',
            vocabulary=inverse_vocabolary,
            output_sequence_length=sequence_length)
        nome_emb_prec = crea_embedding(vectorize_layer=vectorize_layer, inverse_vocab=inverse_vocab, vocab_size=vocab_size,
                                       embedding_dim=EMBEDDING_SIZE, da=passaggio, num_max=num_files, nome_embedding_prec=nome_emb_prec)
else:
    crea_embedding(text_ds, vectorize_layer, inverse_vocab, vocab_size)

ms_prec = print_time()