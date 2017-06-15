from __future__ import print_function, division

import time
import argparse

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from kegra.layers.graph import GraphConvolution
from kegra.utils import *

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
                  optimizer=Adam(lr=0.01))

    # Helper variables for main training loop
    wait = 0
    preds = None
    best_val_loss = 99999

    # Fit
    for epoch in range(1, args.epochs + 1):

        # Log wall-clock time
        t = time.time()

        # Single training iteration
        # (we mask nodes without labels for loss calculation)
        model.fit(graph, y_train,
                  sample_weight=train_mask,
                  batch_size=A.shape[0],
                  epochs=1,
                  shuffle=False,
                  verbose=0)

        # Predict on full dataset
        preds = model.predict(graph, batch_size=A.shape[0])

        # Train / validation scores
        train_val_loss, train_val_acc = evaluate_preds(preds, [y_train, y_val],
                                                       [train_mask, val_mask])
        print("Epoch: {:04d}".format(epoch),
              "train_loss= {:.4f}".format(train_val_loss[0]),
              "train_acc= {:.4f}".format(train_val_acc[0]),
              "val_loss= {:.4f}".format(train_val_loss[1]),
              "val_acc= {:.4f}".format(train_val_acc[1]),
              "time= {:.4f}".format(time.time() - t))

        # Early stopping
        if train_val_loss[1] < best_val_loss:
            best_val_loss = train_val_loss[1]
            wait = 0
        else:
            if wait >= args.patience:
                print('Epoch {}: early stopping'.format(epoch))
                break
            wait += 1

    # Testing
    test_loss, test_acc = evaluate_preds(preds, [y_test], [test_mask])
    print("Test set results:",
          "loss= {:.4f}".format(test_loss[0]),
          "accuracy= {:.4f}".format(test_acc[0]))


if __name__ == '__main__':
    main()
