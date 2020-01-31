import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, adam

# Define function

def showimage(dataframe, imnumber):

    plt.imshow(dataframe.iloc[imnumber].values.reshape(28, 28), cmap='gray')


# Load data
def load_data():
    traindata = 'data/train.csv'
    df = pd.read_csv(traindata)

    images = df.iloc[:, 1:]
    labels = df.iloc[:, :1]
    labels = labels.astype('category')

    return labels, images


def plot_image(image):
    # Plot an image
    # image = images.iloc[x, :]>
    img = image.values
    img = img.reshape((28, 28))
    plt.imshow(img, cmap='gray')


def plot_histogram(image):
    # plot a histogram
    # image = images.iloc[x, :]>
    plt.hist(image.values)


# Create lambda function and applymap to change dataframe values
def apply_threshold(images, threshold):
    # Instead of seperate scaler, we apply this
    new_images = images.applymap(lambda x: 1 if x > threshold else 0)

    return new_images

# Create train test split


def setup_train_test_split(images, labels, test_size):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size)
    n_rows, n_cols = X_train.shape

    # Shape and scale the train data
    Xnp = X_train.values
    Xnp = Xnp / 255
    Xnp = Xnp.reshape(-1, n_cols)
    Ynp = keras.utils.to_categorical(y_train, num_classes=10)

    # Shape and scale the validation data
    Xev = X_test.values
    Xev = Xev / 255
    Xev = Xev.reshape(-1, n_cols)
    Yev = keras.utils.to_categorical(y_test, num_classes=10)

    return Xnp, Ynp, Xev, Yev


def setup_network(nr_dense_layers):
    # FUTURE VERSION: includes dense_layer_nodes_multi in input
    # FUTURE VERSION: includes drop_percentages in input
    n_cols = 28 # Statis for MNIST
    dense_layer_nodes_multi = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    drop_percentages = [0.2, 0, 0, 0.2, 0, 0.2]
    # Setup the network and corresponding layers
    model = Sequential()
    # Input layer
    model.add(Dense(n_cols * dense_layer_nodes_multi[0], activation='relu', input_dim=784))
    model.add(Dropout(drop_percentages[0]))
    for i in range(nr_dense_layers - 1):
        model.add(Dense(n_cols * dense_layer_nodes_multi[i+1], activation='relu'))
        if drop_percentages[i+1] != 0:
            model.add(Dropout(drop_percentages[i+1]))

    # Add output layer
    model.add(Dense(10, activation='softmax'))

    return model


def optimize_compile(model, optimizer, learning_rate):
    # Define optimizer and compile model
    if optimizer == 'adam':
        opt = adam(lr=learning_rate, decay=1e-6)
    else:
        opt = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def train_model(model, Xnp, Ynp, epochs, early_stopping):
    # Fit the model to the training data
    early_stopping_monitor = EarlyStopping(patience=early_stopping)
    model.fit(Xnp, Ynp, epochs=epochs, callbacks=[early_stopping_monitor])

    return model


def evaluate_model(model, Xev, Yev):
    # Evaluate model
    eval = model.evaluate(Xev, Yev, verbose=True)

    return eval


# Test the predictions

#testval = X_test[0, :]
#testval = testval.reshape(-1, n_cols)

#predictions = model.predict_proba(testval)

def predictimage(image, model, n_cols):
    Xp = image.values
    Xp = Xp / 255
    Xp = Xp.reshape(-1, n_cols)
    #plt.imshow(Xp.reshape(28, 28),cmap='gray')
    predictions = model.predict_proba(Xp)
    return print(predictions*100)

def save_model(model, filename):
    # Save keras model
    model.save_weights(filename)
    return True

# Load model weights
#model.load_weights('competition/MNIST/model_190317.h5')

