import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import itertools
from collections import Counter

MAX_SEQUENCE_LENGTH = 150
MAX_NB_WORDS = 100000


def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)
    replace_numbers = re.compile(r'\d+', re.IGNORECASE)
    # 字母小写
    text = text.lower().split()

    # 去停用词
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
    text = " ".join(text)

    # 正则处理
    text = special_character_removal.sub('', text)
    text = replace_numbers.sub('n', text)

    # 提取词干
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    return text


def text2sequence(train_df, test_df):
    list_sentences_train = train_df["comment_text"].fillna("NA").values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    train_label = train_df[list_classes].values
    list_sentences_test = test_df["comment_text"].fillna("NA").values

    comments = []
    for text in list_sentences_train:
        comments.append(text_to_wordlist(text))

    test_comments = []
    for text in list_sentences_test:
        test_comments.append(text_to_wordlist(text))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(comments + test_comments)

    sequences = tokenizer.texts_to_sequences(comments)
    test_sequences = tokenizer.texts_to_sequences(test_comments)

    word_index = tokenizer.word_index
    print('Found %s unique tokens' % len(word_index))

    train_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', train_data.shape)
    print('Shape of label tensor:', y.shape)

    test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of test_data tensor:', test_data.shape)

    return train_data, train_label, test_data


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
