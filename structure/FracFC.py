from structure.DaeCov import *
from structure.DaeFC import *


class FracFC(keras.Model):
    def __init__(self, **kwargs):
        super(FracFC, self).__init__()
        self.dae = kwargs['dae']
        self.fc1 = FcBn(125, activation='elu')
        self.fc2 = FcBn(125, activation='elu')
        self.fc3 = layers.Dense(2)
        self.softmax = layers.Softmax()

    def call(self, inputs, training=None, mask=None):
        x = self.dae.get_features(inputs)
        x = self.fc1(x)
        x = layers.Dropout(rate=0.3)(x)
        x = self.fc2(x)
        x = layers.Dropout(rate=0.3)(x)
        x = self.fc3(x)
        out = self.softmax(x)
        return out

    def build_graph(self):
        x = Input(shape=(30129,))
        return tf.keras.Model(inputs=x, outputs=self.call(x))


if __name__ == '__main__':
    mod = FracFC(dae=DaeFC(gene_num=30129)).build_graph()
    print(mod.summary())