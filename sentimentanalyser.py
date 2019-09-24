import plac
import pathlib
from keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.optimizers import Adam
from spacy.compat import pickle
import spacy
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf

nlp = spacy.load("en_vectors_web_lg")

def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype="int32")

def plot_series(series_1, series_2, format="-", title=None, legend=None):
    plt.plot(series_1)
    plt.plot(series_2)
    plt.title(title)
    plt.legend(legend, loc='upper left')
    plt.show()  
    
 
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.99):
    #if(logs.get('loss')<0.4):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True
      
def get_vectors(docs, max_length):
    docs = list(docs)
    Xs = np.zeros((len(docs), max_length), dtype="int32")
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs


def train_model(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    lstm_shape,
    lstm_settings,
    lstm_optimizer,
    batch_size=100,
    nb_epoch=5,
    by_sentence=False,
):

    print("Loading spaCy")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    embeddings = get_embeddings(nlp.vocab)
    model = compile_model(embeddings, lstm_shape, lstm_settings)
    
    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_texts))
    val_docs = list(nlp.pipe(val_texts))
    if by_sentence:
        train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
        val_docs, val_labels = get_labelled_sentences(val_docs, val_labels)

    train_X = get_vectors(train_docs, lstm_shape["max_length"])
    val_X = get_vectors(val_docs, lstm_shape["max_length"])
    
    callbacks = my_Callback()
    
    estimator = model.fit(
        train_X,
        train_labels,
        validation_data=(val_X, val_labels),
        epochs=nb_epoch,
        batch_size=batch_size, 
        callbacks=[callbacks]
    )
    
    plot_series(estimator.history['acc'], estimator.history['val_acc'], title='model accuracy', legend=['train', 'valid'])
    plot_series(estimator.history['loss'], estimator.history['val_loss'], title='model loss', legend=['train', 'valid'])
      
    predicted_prob = model.predict(val_X)

    prediction = np.where(predicted_prob >=0.5, 1, 0)
    
    count=0
    for i in range(len(val_labels)):
        #print(prediction[i], val_labels.iloc[i])
        if (prediction[i] != val_labels.iloc[i]):
            if count ==0:
                print('Here is the list of misclassified texts:\n')
            count+=1
            print(val_docs[i], '\n')
    print('We got ', count, 'out of ', val_labels.shape[0], 'misclassified texts')
    
    return model

def compile_model(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[0],
            embeddings.shape[1],
            input_length=shape["max_length"],
            trainable=False,
            weights=[embeddings],
            mask_zero=True,
        )
    )
    model.add(TimeDistributed(Dense(shape["nr_hidden"], use_bias=False)))
    model.add(
        Bidirectional(
            LSTM(
                shape["nr_hidden"],
                recurrent_dropout=settings["dropout"],
                dropout=settings["dropout"], 
                return_sequences=True
            )
        )
    )
    model.add(
        Bidirectional(
            LSTM(
                shape["nr_hidden"],
                recurrent_dropout=settings["dropout"],
                dropout=settings["dropout"],
                return_sequences=True,
            )
        )
    )
    model.add(
        Bidirectional(
            LSTM(
                shape["nr_hidden"],
                recurrent_dropout=settings["dropout"],
                dropout=settings["dropout"],
                return_sequences=True,
            )
        )
    )
    model.add(
        Bidirectional(
            LSTM(
                shape["nr_hidden"],
                recurrent_dropout=settings["dropout"],
                dropout=settings["dropout"],
            )
        )
    )
    model.add(Dense(shape["nr_class"], activation="sigmoid"))
    model.compile(
        optimizer=Adam(lr=settings["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_embeddings(vocab):
    return vocab.vectors.data

def cleanup_text(docs, logging=False):
    docs = docs.str.strip().replace("\n", " ").replace("\r", " ")
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-' and tok.pos_ !='NUM' and tok.pos_ !='PUNCT']
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

def read_data(data_dir, training_portion):
    texts = pd.DataFrame()
    for filename in pathlib.Path(data_dir).iterdir():
        with filename.open(encoding='latin-1') as file_:
            if not file_.name.endswith('DS_Store'):
                text = pd.read_csv(file_, usecols=[1, 2], encoding='latin-1')
                texts = texts.append(text, ignore_index=True)
    texts = texts.sample(frac=1)
    text_cln = cleanup_text(texts.iloc[:, 1], logging=True)
    sentiments = np.asarray(texts.iloc[:, 0].unique())
    for i in range(len(sentiments)):
        texts.iloc[:, 0].replace(sentiments[i], i, inplace=True)
        
    train_size = int(len(texts) * training_portion)
    train_texts, train_labels = text_cln[:train_size], texts.iloc[:train_size, 0]
    val_texts, val_labels = text_cln[train_size:], texts.iloc[train_size:, 0]
    return train_texts, train_labels, val_texts, val_labels

@plac.annotations(
    train_dir=("Location of training file or directory"),
    model_dir=("Location of output model directory",),
    nr_hidden=("Number of hidden units", "option", "u", int),
    max_length=("Maximum sentence length", "option", "l", int),
    dropout=("Dropout", "option", "d", float),
    learn_rate=("Learn rate", "option", "e", float),
    nb_epoch=("Number of training epochs", "option", "n", int),
    batch_size=("Size of minibatches for training LSTM", "option", "b", int),
)
def main(
    model_dir='/Users/masha/Data/Model',
    train_dir='/Users/masha/Data/Train',
    nr_hidden=128,
    max_length=100,  
    dropout=0.2,
    learn_rate=0.0001,  
    nb_epoch=150,
    batch_size=64,
    #nr_examples=-1,
    training_portion = .8,
):  # Training params
    if model_dir is not None:
        model_dir = pathlib.Path(model_dir)
    if train_dir is None:
        print('Please provide training directory!')
    train_texts, train_labels, val_texts, val_labels = read_data(train_dir, training_portion)
     
    model = train_model(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        {"nr_hidden": nr_hidden, "max_length": max_length, "nr_class": 1},
        {"dropout": dropout, "lr": learn_rate},
        {},
         nb_epoch=nb_epoch,
        batch_size=batch_size
        )
        
        
    weights = model.get_weights()
    if model_dir is not None:
        with (model_dir / "model").open("wb") as file_:
            pickle.dump(weights[1:], file_)
        with (model_dir / "config.json").open("w") as file_:
            file_.write(model.to_json())


if __name__ == "__main__":
    plac.call(main)
