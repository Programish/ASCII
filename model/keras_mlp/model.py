import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import SGD


class OcrModel(object):
    def __init__(self, shape_pixels, num_classes):
        # flattend input shape
        self.num_pixels = shape_pixels[0] * shape_pixels[1]

        self.model = Sequential()
        self.model.add(Dense(self.num_pixels * 2))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=SGD(lr=1, momentum=0.9, nesterov=True))

    def flatten_pixels(self, inputs):
        return inputs.reshape((-1, self.num_pixels))

    def train(self, inputs, labels, epochs=1):
        history = self.model.fit(self.flatten_pixels(inputs),
                                 labels,
                                 batch_size=inputs.shape[0],
                                 epochs =epochs,
                                 verbose=0)
        # return loss of last epoch
        return history.history['loss'][-1]

    def predict(self, inputs):
        results =  self.model.predict(self.flatten_pixels(inputs), verbose=0)
        return np.argmax(results,1)
