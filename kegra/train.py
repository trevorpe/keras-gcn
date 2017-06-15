from __future__ import print_function, division

import argparse

import numpy as np

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.utils import accuracy_metric, load_data, preprocess_adj, \
    normalized_laplacian, chebyshev_polynomial, rescale_laplacian, get_splits

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience


def main():
    # Parse command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset',
                            default=DATASET,
                            choices=['cora'],
                            help='the data set to load')
    arg_parser.add_argument('--filter',
                            default=FILTER,
                            choices=['localpool', 'chebyshev'],
                            help='the filter to use')
    arg_parser.add_argument('--no-norm-symmetry',
                            dest='sym_norm',
                            action='store_false',
                            help='disable symmetric normalization')
    arg_parser.add_argument('-e', '--epochs',
                            type=int,
                            default=NB_EPOCH,
                            help='the number of epochs to train for')
    arg_parser.add_argument('--patience',
                            type=int,
                            default=PATIENCE,
                            help='the patience for early-stopping')
    arg_parser.add_argument('--max-degree',
                            type=int,
                            default=MAX_DEGREE,
                            help='maximum polynomial degree')
    args = arg_parser.parse_args()

    # Get data
    X, A, y = load_data(dataset=args.dataset)
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_splits(y)

    # Normalize X
    X = np.diag(1 / np.array(X.sum(1)).flatten()).dot(X)

    if FILTER == 'localpool':
        # Local pooling filters
        # (see 'renormalization trick' in Kipf & Welling, arXiv 2016)
        print('Using local pooling filters...')
        A_ = preprocess_adj(A, args.sym_norm)
        support = 1
        graph = [X, A_]
        G = [Input(shape=tuple(A_.shape[1:]), sparse=True)]
    elif FILTER == 'chebyshev':
        # Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)
        print('Using Chebyshev polynomial basis filters...')
        L = normalized_laplacian(A, args.sym_norm)
        L_scaled = rescale_laplacian(L)
        T_k = chebyshev_polynomial(L_scaled, args.max_degree)
        support = args.max_degree + 1
        graph = [X] + T_k
        G = [Input(shape=tuple(T_k.shape[i + 1]), sparse=True)
             for i in range(support)]

    else:
        raise ValueError('Invalid filter type.')

    X_in = Input(shape=(X.shape[1],))

    # Define model architecture
    H = Dropout(0.5)(X_in)
    H = GraphConvolution(16, support,
                         activation='relu',
                         W_regularizer=l2(5e-4))([H] + G)
    H = Dropout(0.5)(H)
    Y = GraphConvolution(y.shape[1], support, activation='softmax')([H] + G)

    # Compile model
    model = Model(inputs=[X_in] + G,
                  outputs=Y)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01),
                  metrics=[accuracy_metric])

    model.fit(graph, y_train,
              sample_weight=train_mask,
              validation_data=(graph, y_val, val_mask),
              batch_size=A.shape[0],
              epochs=args.epochs,
              shuffle=False,
              verbose=1)

    # Testing
    test_loss, test_acc = model.evaluate(graph, y_test,
                                         sample_weight=test_mask,
                                         batch_size=A.shape[0])
    print("Test set results:",
          "loss= {}".format(test_loss),
          "accuracy= {}".format(test_acc))


if __name__ == '__main__':
    main()
