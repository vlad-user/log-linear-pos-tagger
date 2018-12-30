"""Contains functions for training and evaluation of models."""
import pickle
import os
import sys
from time import time
import random
import json

import numpy as np

import nlp_utils
import inference

def train_model1(datadir=None,
                 rare_word_thresh=2,
                 train_data_fname='train.wtag',
                 transformed_fname='transformed.pkl',
                 vectorizer_fname='vectorizer.pkl',
                 raw_feature_converter=None,
                 verbose=False,
                 store_intermediate=False,
                 init=None,
                 weights_fname='x.pkl'):
    """Trains model 1.
    
    This function uses already fitted `vectorizer` and `transformed_data`
    if there are exists in `datadir`. If not, the `vectorizer` is
    fitted and the data is transformed.

    Args:
        `datadir`: A path on disk where data and all auxiliary files
            is/will be stored.
        `rare_word_thresh`: A integer indicating a minimum number
            of occurences of a feature to be included as model's
            feature.
        `train_data_fname`: A filename containing training data.
        `transformed_fname`: A filename containing transformed by
            vectorizer data.
        `vectorizer_fname`: A filename containg vectorizer.
        `raw_feature_converter`: See `nlp_utils.Vectorizer`.
        `verbose`: If `True`, prints the progress to `stdout`.
        `store_intermediate`: If `True`, stores weights after each
            iteration of the optimization algorithm.
        `init`: A initialization vector for optimization algorithm.
        `weights_fname`: A filename for weights. If
            `store_intermediate` is `False`, this arguments is ignored.

    Returns:
        Trained model.

    Raises:
        `FileNotFoundError`: If there's no train data file in `datadir`.
    """
    processor = nlp_utils.TextProcessor_v2()

    if datadir is None:
        dirname = os.path.abspath(os.path.dirname(__file__))
    else:
        dirname = os.path.abspath(datadir)

    # load and normalize train data
    train_data_path = os.path.join(dirname, train_data_fname)
    try:
        with open(os.path.join(dirname, train_data_fname)) as fo:
            text = fo.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError('No '+train_data_fname+' file in '+dirname) from exc
    train_data = [processor.word_pos_tokenize(sent)
                  for sent in processor.sent_tokenize(text)]

    # create/load/store vectorizer
    vectorizer_path = os.path.join(dirname, vectorizer_fname)
    if not os.path.exists(vectorizer_path):
        if verbose: print('Fitting vectorizer...')
        vectorizer = nlp_utils.Vectorizer(
                rare_word_thresh=rare_word_thresh,
                raw_feature_converter=raw_feature_converter)
        vectorizer.fit(train_data)
        with open(vectorizer_path, 'wb') as fo:
            pickle.dump(vectorizer, fo, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if verbose: print('Found vectorizer...')
        with open(vectorizer_path, 'rb') as fo:
            vectorizer = pickle.load(fo)
    if verbose:
        print('vectorizer has', vectorizer._n_features, 'features')
        print('vectorizer has following tags:')
        for t in vectorizer.list_available_tags():
            print(t)
    # create/load/store transformed data
    transformed_path = os.path.join(dirname, transformed_fname)
    if not os.path.exists(transformed_path):
        if verbose: print('Transforming data...')
        transformed = vectorizer.transform(train_data)
        with open(transformed_path, 'wb') as fo:
            pickle.dump(transformed, fo, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        if verbose: print('Found transformed data...')
        with open(transformed_path, 'rb') as fo:
            transformed = pickle.load(fo)

    # train log linear model
    print('Training...')
    model = nlp_utils.LogLinearModel(vectorizer, dirname)
    model.minimize(data=transformed,
                   datapath=dirname,
                   verbose=verbose,
                   init=init,
                   store_intermediate=store_intermediate,
                   weights_fname=weights_fname)

    return model

def train_model2(datadir=None,
                 rare_word_thresh=1,
                 train_data_fname='train2.wtag',
                 transformed_fname='transformed2.pkl',
                 vectorizer_fname='vectorizer2.pkl',
                 raw_feature_converter=None,
                 verbose=False,
                 store_intermediate=False,
                 init=None,
                 weights_fname='x_model2.pkl'):
    """Trains model 2. See `train_model1()`."""
    return train_model1(
            datadir=datadir, rare_word_thresh=rare_word_thresh,
            train_data_fname=train_data_fname,
            transformed_fname=transformed_fname,
            vectorizer_fname=vectorizer_fname,
            raw_feature_converter=nlp_utils.Vectorizer._convert_to_raw_hist_model2,
            verbose=verbose, store_intermediate=store_intermediate,
            init=init, weights_fname=weights_fname)

def compute_accuracy(model, datadir=None, 
                     test_data_fname='test.wtag',
                     verbose=False):
    """Computes accuracy over the whole test dataset.

    Args:
        `model`: A fitted model instance.
        `datadir`: A path on disk where data and all auxiliary files
            is/will be stored.
        `test_data_fname`: A filename containing test data.
        `verbose`: If `True`, prints the progress to `stdout`.

    Returns:
        A tuple (`y_true`, `y_pred`, `accuracy`)

    Raises:
        `FileNotFoundError`: If `test_data_fname` not in `datadir`.

    """
    if datadir is None:
        dirname = os.path.abspath(os.path.dirname(__file__))
    else:
        dirname = os.path.abspath(datadir)

    test_data_path = os.path.join(dirname, test_data_fname)
    try:
        with open(test_data_path) as fo:
            text = fo.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError('No '+test_data_fname+' file in '+dirname) from exc
    processor = nlp_utils.TextProcessor_v2()
    test_data = [processor.word_pos_tokenize(sent)
                 for sent in processor.sent_tokenize(text)]
    test_data = [t for t in test_data if t != []]
    viterbi_infer = inference.ViterbiInference(model)
    accuracies = []
    times = []
    y_true = []
    y_pred = []
    data_len = len(test_data)

    for words_and_tags in test_data:
        start_time = time()
        predicted = viterbi_infer.predict([w[0] for w in words_and_tags])
        tags = [w[1] for w in words_and_tags]
        res = [x == y for x, y in zip(predicted, tags)]
        try:
            accuracies.append(sum(res)/len(res))
        except ZeroDivisionError:
            print('sample num:', len(result))
            continue
        y_true += tags
        y_pred += predicted
        times.append(time() - start_time)
        if verbose:
            log = {
                '[curr_acc': str(np.mean(accuracies)) + ']',
                '[time': str(np.mean(times)) + ' sec/sentence]',
                '[completed': str(len(accuracies)) + '/' + str(data_len) + ']'
               }
            print_log(log)

    return y_true, y_pred, np.mean(accuracies)
    
def label_model1(fname2label='comp.words',
                 model_fname='model1.pkl',
                 datadir=None,
                 verbose=False):
    """Helper for labeling raw sentences for model 1.

    Args:
        `fname2label`: A filename that contains a sentences to label.
        `model_fname`: A filename of the fitted model (pickle binary
            object).
        `datadir`: A directory where the filenames/models are stored.
        `verbose`: If `True`, prints progress log.

    Returns:
        A list of tagged sentences.
    """
    if datadir is None:
        dirname = os.path.abspath(os.path.dirname(__file__))
    else:
        dirname = os.path.abspath(datadir)

    fname2label = os.path.join(dirname, fname2label)
    with open(fname2label) as fo:
        text = fo.read()

    processor = nlp_utils.TextProcessor_v2()
    sents = processor.sent_tokenize(text)
    
    words_list = [processor.word_tokenize(sent) for sent in sents]
    
    with open(os.path.join(dirname, model_fname), 'rb') as fo:
        model = pickle.load(fo)
    
    viterbi_infer = inference.ViterbiInference(model)
    
    result = []
    for words in words_list:
        tags = viterbi_infer.predict(words)
        result.append(processor.merge_words_and_tags(words, tags))    
        if verbose:
            print_log({'[completed': str(len(result)) + '/' + str(len(words_list)) + ']'})
    return result

def label_model2(fname2label='comp2.words',
                 model_fname='model2.pkl',
                 datadir=None,
                 verbose=False):
    """See `label_model1()`"""
    return label_model1(fname2label=fname2label,
                        model_fname=model_fname,
                        datadir=datadir,
                        verbose=verbose)

def create_confusion_matrix(y_pred, y_true, cm_fname='cm.png'):
    """Generates confusion matrix image.

    This package uses `pandas` and `matplotlib` packages.

    ## Usage example:
    ```python
    # Suppose `model` is a fitted log-linear model object.
    y_true, y_pred, accuracy = compute_accuracy(model, verbose=True)
    create_confusion_matrix(y_pred, y_true)
    ```

    Args:
        `y_true`: A list of true tags.
        `y_pred`: A list of predicted tags.
        `cm_fname`: A name of an image with confusion matrix.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    actual = pd.Series(y_true, name='Actual')
    predicted = pd.Series(y_pred, name='Predicted')
    df = pd.crosstab(actual, predicted)

    plt.matshow(df, cmap=plt.cm.gray_r)
    plt.colorbar()
    tick_marks = np.arange(len(df.columns))
    plt.xticks(tick_marks, df.columns, rotation=45)
    plt.yticks(tick_marks, df.index)
    plt.ylabel(df.index.name)
    plt.xlabel(df.columns.name)
    plt.rcParams['figure.figsize'] = (250, 250)
    plt.savefig(cm_fname, bbox_inches='tight')
    plt.close()

def print_log(dict_):
    """Helper for printing logs."""
    buff = '\r' + '|'.join(sorted([k+':'+str(v) for k, v in dict_.items()]))
    sys.stdout.write(buff)
    sys.stdout.flush()
