"""
 31/07/2023

@author Leandro Paolo De Persiis

"""

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

vocab_size = 100000 # vocabulary size
sequence_length = 10 # number of words in a sequence.
num_ns = 4 # number of negative samples per positive context

context_class = tf.reshape(tf.constant(2, dtype="int64"), (1, 1))
negative_sampling_candidates = 2
sampling_table = 0

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for `vocab_size` tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):

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
        context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
        label = tf.constant([1] + [0]*num_ns, dtype="int64")

        # Append each element from the training example to global lists.
        targets.append(target_word)
        contexts.append(context)
        labels.append(label)

    return targets, contexts, labels

def get_word_matrix():
    """
    Questa funzione ci serve per crearci un dizionario avente come indice la parola e come valore il vettore dell'embedding corrispondente
    """
    word_matrix = {}
    with open(MY_EMBEDDING_PATH + 'glove.6B.'+str(EMBEDDING_SIZE)+'d.txt', 'r', encoding='UTF-8') as file_emb:
        for row in file_emb: # leggo ogni riga del file di testo contenente l'embedding
            row = row.split() # la divido nei suoi elementi
            word_matrix[row[0]] = np.array(row[1:], dtype='float32') # il primo è la parola e sarà l'indice di questa voce, gli altri andranno a formare il vettore
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


anni = range(2018, 2023)
mesi = range(1, 13)
nome_file_txt = 'articles-'+str(anni[0])+ '_' +str(anni[-1])+'.txt'
print("nome_file_txt:", nome_file_txt)
with open(nome_file_txt, 'w', newline=None, encoding='UTF-8', errors='strict') as file_txt:
    for a in anni:
        for m in mesi:
            nome_file = str(a) + '-' + str(m) + '-articles'
            with open(nome_file + ".csv", 'r', newline=None, encoding='UTF-8', errors='ignore') as file_nyt:
                reader = csv.reader(file_nyt)
                for r in reader:
                    if r is not None and len(r)>2 and len(r[1]) > 50:
                        file_txt.write(r[1] + " \n ")

print("Completato file " + nome_file_txt)
# path_to_file = tf.keras.utils.get_file(nome_file + ".txt", 'file:' + nome_file + ".txt")

text_ds = tf.data.TextLineDataset(nome_file_txt).filter(lambda x: tf.cast(tf.strings.length(x), bool))

# Now, create a custom standardization function to lowercase the text and
# remove punctuation.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')


# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the
# same length.
vectorize_layer = layers.TextVectorization(
    standardize='lower_and_strip_punctuation', # custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    encoding='utf-8',
    output_sequence_length=sequence_length)

print(type(vectorize_layer))
print(vectorize_layer)

vectorize_layer.adapt(text_ds.batch(BATCH_SIZE))
print("vocabulary_size", vectorize_layer.vocabulary_size())

inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])

print("len(inverse_vocab))", len(inverse_vocab))

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
print(dataset)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)

embedding_matrix = []
if WITH_GLOVE:
    word_matrix = get_word_matrix()
    embedding_matrix = get_embedding_matrix(word_matrix, inverse_vocab)
    print("embedding_matrix:", embedding_matrix)
    print(embedding_matrix[:3])


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


word2vec = Word2Vec(vocab_size, EMBEDDING_SIZE, embedding_matrix)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

word2vec.fit(dataset, epochs=22) #, callbacks=[tensorboard_callback])

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

emb_with_glove = ''
if WITH_GLOVE:
    emb_with_glove = '_with_glove'
with open('embedding_'+str(anni[0])+ '_' +str(anni[-1])+ emb_with_glove +'.txt', 'w', encoding='utf-8') as emb:
    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        vec_str = ' '.join([str(x) for x in vec])
        emb.write(word + ' ' + vec_str + "\n")


print("Completato salvataggio dell'embedding")