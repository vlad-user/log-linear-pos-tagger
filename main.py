"""Trains, evaluates model1 and stores two models."""
import pickle

import train_and_eval

def main():
    """Trains, evaluates and stores two model2.

    To test the model during training, use `store_intermediate=True`
    flag as in following example:
    ```python
    # start training from one script:
    # ...
    # model = train_and_eval.train_model1(verbose=True,
                                          store_intermediate=True)
    # ...
    # and from other script run the following (after vectorization is
    # completed):
    import pickle
    import numpy as np

    import nlp_utils
    import train_and_evaluate

    with open('x.pkl', 'rb') as fo:
        x = pickle.load(fo)
    with open('vectorizer.pkl', 'rb') as fo:
        vectorizer = pickle.load(fo)
    model = nlp_utils.LogLinearModel(vectorizer)
    model.weights = x
    accuracy = train_and_eval.compute_accuracy(model, verbose=True)

    ```
    """
    
    model1 = train_and_eval.train_model1(verbose=False,
                                         store_intermediate=False)
    with open('model1.pkl', 'wb') as fo:
        pickle.dump(model1, fo, protocol=pickle.HIGHEST_PROTOCOL)
    accuracy = train_and_eval.compute_accuracy(model, verbose=True)
    print('accuracy for model1:', accuracy[2])

    model2 = train_and_eval.train_model2(verbose=False,
                                         store_intermediate=False)
    with open('model2.pkl', 'wb') as fo:
        pickle.dump(model2, fo, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()