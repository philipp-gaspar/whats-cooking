import sys
import os

import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import StratifiedKFold

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

def load_model(data):
    """
    Function to create the Neural Network Model
    """
    K.clear_session()

    # creating the Deep Neural Net Model
    model = Sequential()

    # layer 1
    model.add(Dense(units=128,
                    activation='relu',
                    input_shape=(data.shape[1], )))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))

    # layer 2
    model.add(Dense(units=64,
                    activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))

    # output layer
    model.add(Dense(units=n_classes,
                    activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.005),
                  metrics=['acc'])

    return model

if __name__ == '__main__':
    # =======
    # FOLDERS
    # =======
    package_path = os.path.dirname(os.getcwd())
    data_path = os.path.join(package_path, 'data')
    experiments_path = os.path.join(package_path, 'experiments')

    # =========
    # LOAD DATA
    # =========
    input_file = os.path.join(data_path, 'train.json')
    df = pd.read_json(input_file)

    # creates a Tokenizer object
    tokenizer = Tokenizer(num_words=6000, split=', ', lower=True)

    # builds the word index
    tokenizer.fit_on_texts(df['ingredients'])

    # directly get the representations
    train_data = tokenizer.texts_to_matrix(df['ingredients'], mode='tfidf')

    n_samples = train_data.shape[0]
    n_features = train_data.shape[1]

    print('The training dataset with the new representation have:')
    print('  - %i entries/recipes' % n_samples)
    print('  - %i features/ingredients\n' % n_features)

    # construct the target vector
    # categorical target (one-hot encoded)
    lb = LabelBinarizer()
    target_cat = lb.fit_transform(df['cuisine'])

    # integer target, used in the StratifiedKfold class
    # in order to make each fold with balanced classes
    le = LabelEncoder()
    target = le.fit_transform(df['cuisine'])

    n_classes = len(np.unique(target))
    print('The dataset has %i unique classes.' % n_classes)

    # ================
    # CROSS VALIDATION
    # ================
    n_splits = 5
    seed = 2018
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_data, target))

    # ===============
    # MODEL CALLBACKS
    # ===============
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    weights_file = os.path.join(experiments_path, 'deep_model.hdf5')
    checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True)

    callbacks = [early_stop, checkpoint]

    # ===========
    # TRAIN MODEL
    # ===========
    cv_scores = []
    cv_hist = []

    for fold, (trn_idx, val_idx) in enumerate(folds):
        print('>> Fold %i# <<' % int(fold+1))

        # get training and validation data folds
        X_trn = train_data[trn_idx, :]
        y_trn = target_cat[trn_idx, :]
        X_val = train_data[val_idx, :]
        y_val = target_cat[val_idx, :]

        print('  Training on %i examples.' % X_trn.shape[0])
        print('  Validating on %i examples.' % X_val.shape[0])

        model = load_model(X_trn)

        # serialize model to JSON
        if fold == 0:
            model_json = model.to_json()
            model_file = os.path.join(experiments_path, 'deep_model.json')
            with open(model_file, 'w') as json_file:
                json_file.write(model_json)

        hist = model.fit(X_trn, y_trn,
                         validation_data=(X_val, y_val),
                         batch_size=32,
                         epochs=100,
                         callbacks=callbacks,
                         verbose=0)

        scores = model.evaluate(X_val, y_val)
        print('  This model has %1.2f validation accuraccy.\n' % scores[1])

        cv_scores.append(scores)
        cv_hist.append(hist)

    # ==================
    # EVALUATE THE MODEL
    # ==================
    val_acc = []
    for metric in cv_scores:
        val_acc.append(metric[1])

    print('Accuracy = %1.4f +- %1.4f' % (np.mean(val_acc), np.std(val_acc)))

    # ======================
    # GETTING THE BEST MODEL
    # ======================
    model.load_weights(weights_file)

    # ===================
    # CREATING SUBMISSION
    # ===================
    input_file = os.path.join(data_path, 'test.json')
    df = pd.read_json(input_file)

    # directly get the representations
    data_test = tokenizer.texts_to_matrix(df['ingredients'], mode='tfidf')

    n_samples = data_test.shape[0]
    n_features = data_test.shape[1]

    print('The test dataset with the new representation have:')
    print('  - %i entries/recipes' % n_samples)
    print('  - %i features/ingredients' % n_features)

    # predict classes using test data
    predict = model.predict_classes(data_test)

    # map each integer to the string labels
    cat = pd.factorize(le.classes_)

    # create the column
    df['cuisine'] = cat[1][predict]

    submissions_path = os.path.join(package_path, 'submissions')
    submissions_file = os.path.join(submissions_path, 'first_deep_model.csv')

    df[['id', 'cuisine']].to_csv(submissions_file, index=False)
