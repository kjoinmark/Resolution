import glob

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    features = []
    labels = []

    all_ = glob.glob(".\\results\\source\\моя оценка\\*")
    print(glob.glob(".\\results\\source\\моя оценка\\*"))
    for file in all_:
        print(file)
    for pack in all_:
        if pack.find('.ini') != -1:
            continue
        all_files = glob.glob(pack + '\\*.txt')
        images = []
        results = []
        for i in range(0, 42):
            file = glob.glob(pack + f'\\img_{i}_*.txt')
            if len(file) == 0:
                continue
            path = file[0]
            features.append(np.loadtxt(path, delimiter=','))
            labels.append(int(path[path.rfind('.txt') - 1:path.rfind('.txt')]))

            # if res != 'N':
            # print(res)
            #   results.append(int(path[path.rfind('result') + 6:path.rfind('result') + 7]))

    features = np.asarray(features)
    labels = np.asarray(labels)
    print(features[0])
    print(np.shape(features))
    print(labels[0])
    print(np.shape(labels))

    num_classes = 2
    input_shape = (115, 115, 1)

    features = np.expand_dims(features, -1)
    print("x_train shape:", features.shape)
    print(features.shape[0], "train samples")

    print("y_train shape:", labels.shape)
    print(labels.shape[0], "train samples")
    print(labels)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(1000, use_bias=True, activation="relu"),
            layers.Dense(100, activation="relu"),
            # layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    print(model.summary())

    batch_size = 64
    epochs = 50

    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  # categorical_crossentropy
                  metrics=["accuracy"])

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print(X_test.shape)
    predict = model.predict(X_test)
    # print(predict)
    # print(y_test)
    predict_0 = np.where(predict > 0.5, 1, 0)
    counter = 0
    for i in range(len(predict)):
        #        counter += abs(y_test[i][0] - predict_0[i][0])
        counter += abs(y_test[i] - predict_0[i])

    print(counter, len(predict))

    for i in predict: print("%.2f" % i)

    model.save('./model/model_newreg')


def predict():
    features = []
    labels = []

    path = ".\\results\\source\\test\\44839_Г3-2_В3-3"
    all_ = glob.glob(path)
    print(glob.glob(path))

    model = keras.models.load_model('./model/model2')
    print(path)
    all_files = glob.glob(path + '\\*.txt')
    images = []
    results = []
    files = []
    for file in all_files:
        files.append(file)
        features.append(np.loadtxt(file, delimiter=','))
        labels.append(file[file.rfind('.txt') - 1:file.rfind('.txt')])

    print(path)

    X_test = np.asarray(features)
    y_test = np.asarray(labels)

    X_test = np.expand_dims(X_test, -1)

    y_test = tf.keras.utils.to_categorical(y_test, 2)

    predict = model.predict(X_test)
    print(predict)
    predict_0 = np.where(predict > 0.5, 1, 0)
    counter = 0
    for i in range(len(predict)):
        print(y_test[i], predict_0[i])
        counter += abs(y_test[i][0] - predict_0[i][0])

    print(counter, len(predict))

    print(path + '\\log.dat')

    save = open(path + '\\log.dat', 'w')

    save.write('name \t original result \t predicted result 1.0 \t predicted result\n')

    for i in range(len(files)):
        save.write(files[i] + ' ' + str(y_test[i]) + ' ' + str(predict[i]) + ' ' + str(predict_0[i]) + '\n')

    save.close()


create_model()


def model_builder(hp):
    num_classes = 2
    input_shape = (115, 115, 1)

    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(layers.Flatten())

    hp_units_1 = hp.Int('units1', min_value=10, max_value=1000, step=10)
    model.add(layers.Dense(units=hp_units_1, use_bias=True, activation="relu"))

    hp_units_2 = hp.Int('units2', min_value=10, max_value=1000, step=10)
    model.add(layers.Dense(units=hp_units_2, use_bias=True, activation="relu"))

    model.add(keras.layers.Dense(num_classes, activation="softmax"))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


import keras_tuner as kt


def find_model():
    features = []
    labels = []

    all_ = glob.glob(".\\results\\source\\моя оценка\\*")
    print(glob.glob(".\\results\\source\\моя оценка\\*"))
    for pack in all_:
        if pack.find('.ini') != -1:
            continue
        # print(pack)
        all_files = glob.glob(pack + '\\*.txt')
        if len(all_files) == 0:
            continue
        for i in range(0, 24):
            file = glob.glob(pack + f'\\img_{i}_*.txt')
            path = file[0]
            features.append(np.loadtxt(path, delimiter=','))
            labels.append(path[path.rfind('.txt') - 1:path.rfind('.txt')])
            # print(file)

    features = np.asarray(features)
    labels = np.asarray(labels)
    print(features[0])
    print(np.shape(features))
    print(labels[0])
    print(np.shape(labels))

    num_classes = 2
    input_shape = (115, 115, 1)

    features = np.expand_dims(features, -1)
    print("x_train shape:", features.shape)
    print(features.shape[0], "train samples")

    labels = tf.keras.utils.to_categorical(labels, num_classes)

    print("y_train shape:", labels.shape)
    print(labels.shape[0], "train samples")
    print(labels)

    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=20,
                         factor=3,
                         directory='new_dir',
                         project_name='intro_')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(features, labels, epochs=50, validation_split=0.2, callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units1')} and {best_hps.get('units2')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(features, labels, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

# find_model()
