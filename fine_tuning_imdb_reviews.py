import bert_model

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

import pandas as pd

import re
import os
import csv


def load_imdb_dataset(train_slice, test_slice, as_supervised=False):
    (train_data, test_data), info = tfds.load('imdb_reviews',
                                              split=['train[{}]'.format(train_slice), 'test[{}]'.format(test_slice)],
                                              as_supervised=as_supervised,
                                              with_info=True
                                              )

    return train_data, test_data


def get_label_name(label_id):
    labels = ["Negative", "Positive"]
    return labels[label_id]


def convert_dataset_into_dataframe(tfds_data, info=None):
    df = tfds.as_dataframe(tfds_data, info)
    df['text'] = df['text'].str.decode('utf8')
    return df


def save_dataframe_to_csv(df, filepath):
    df.to_csv(filepath)
    return


def clean_text(text):
    text = re.sub("@\S+", " ", text) # Remove Mentions
    text = re.sub("https*\S+", " ", text) # Remove URL
    text = re.sub("#\S+", " ", text) # Remove Hastags
    text = re.sub('&lt;/?[a-z]+&gt;', '', text) # Remove special Charaters
    text = re.sub('#39', ' ', text) # Remove special Charaters
    text = re.sub('<.*?>', '', text) # Remove html
    text = re.sub(' +', ' ', text) # Merge multiple blank spaces

    return text


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1,len(acc) +1)

    plt.plot(epochs,acc, 'bo' , label="Training acc")
    plt.plot(epochs,val_acc, 'b' , label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.figure()

    plt.plot(epochs,loss, 'bo' , label="Training loss")
    plt.plot(epochs,val_loss, 'b' , label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()
    return

"""
def bert_fine_tuning():
    label_list = [0, 1]
    n_out_units = 1

    activation_function = "sigmoid"
    dropout_perc = 0.3
    max_seq_length = 256  # maximum length of (token) input sequences

    learning_rate = 2e-5
    epochs = 3

    exp_description = "training bert with 100% of the training examples on imdb_reviews-dataset for 3 epochs, with 0.2 dropout and learning_rate 2e-5 and max seq lenght of 256"

    model = bert_model.BertModel()

    model.set_experiment_description(exp_description)

    model.load_pre_trained_bert_tf_hub_from_dir("saved_models/pre_trained/bert_en_uncased_L-12_H-768_A-12_3")

    model.create_model(1, [0, 1], max_seq_length, activation_function, dropout_perc)

    model.compile(learning_rate=learning_rate)

    history = model.fit(train_data, test_data, epochs=epochs, batch_size=8, verbose=1, n_train=len(df_train),
                        n_test=len(df_test))

    out_path = model.save_model(experiment_description="imdb_reviews_exp_1")

    df_train.to_csv(os.path.join(out_path, "df_train.csv"))
    df_test.to_csv(os.path.join(out_path, "df_test.csv"))

    return
"""

def load_fine_tuned_model(directory_path):
    # Create the model wrapper object
    model = bert_model.BertModel()

    # Load the model from disk
    model.load_model(directory_path)

    # Get the tokenizer
    tokenizer = model.get_tokenizer()

    return model, tokenizer


def load_dataset_from_csv(dataset_path):
    df = pd.read_csv(dataset_path)
    return df


def get_label_given_probabilities(p):
    if p >= 0.5:
        return "Positive"
    return "Negative"


if __name__ == "__main__":

    model, tokenizer = load_fine_tuned_model("saved_models/fine_tuned/20201117_bert_model_imdb_reviews_exp_0")

    df = load_dataset_from_csv("datasets/df_test.csv")

    print(df.shape)

    input_texts = ["This film was very awful", "This film was very good", "The film is very bad, however the book is wonderful", "The film is not as good as the book!"]

    predictions = model.predict(input_texts)

    # Examples of predictions
    print("{} -> probability: {} -> label: {} ".format(input_texts[0], predictions[0], get_label_given_probabilities(predictions[0])))
    print("{} -> probability: {} -> label: {} ".format(input_texts[1], predictions[1], get_label_given_probabilities(predictions[1])))
    print("{} -> probability: {} -> label: {} ".format(input_texts[2], predictions[2], get_label_given_probabilities(predictions[2])))
    print("{} -> probability: {} -> label: {} ".format(input_texts[3], predictions[3], get_label_given_probabilities(predictions[3])))

    # Examples of tokenization

    #features, output_path = extract_and_save_embeddings(df,
    #                                                    model,
    #                                                    filename="train_embeddings_0_1_2_3.hdf5",
    #                                                    output_dir="/Users/salvatore/PycharmProjects/ConceptDriftManager/saved_embeddings/20201116_bert_model_0_1_2_3_embeddings"
    #                                                    )


