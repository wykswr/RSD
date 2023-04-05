import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError, CosineSimilarity
from tensorflow.keras.optimizers import Adam


def train(mod, X, y):
    loss = keras.losses.CosineSimilarity(axis=1)
    epoch = 100
    batch_size = 32
    lr = 0.005
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    mtc = [MeanSquaredError(name="mse"),
           MeanAbsoluteError(name="mae"),
           CosineSimilarity(name='cos', axis=1)]
    mod.compile(loss=loss, optimizer=Adam(learning_rate=lr), metrics=mtc)
    es = EarlyStopping(monitor='val_loss',
                       min_delta=0.0001,
                       mode='min',
                       patience=10,
                       restore_best_weights=True)
    return mod.fit(X_train, y_train, batch_size=batch_size, epochs=epoch,
                   verbose=1,
                   shuffle=True,
                   callbacks=[es, ],
                   validation_data=(X_val, y_val))

