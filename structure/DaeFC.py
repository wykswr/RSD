import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model


class DaeFC(keras.Model):
    def __init__(self, **kwargs):
        super(DaeFC, self).__init__()
        self.input_size = kwargs['gene_num']
        self.encoder_FC1 = FcBn(1000, 'elu', name='Encoder_FC1')
        self.encoder_FC2 = FcBn(500, 'tanh', name='Encoder_FC2')
        self.latent = layers.Dense(125, name='latent')
        self.decoder_FC1 = FcBn(500, 'tanh', name='Decoder_FC1')
        self.decoder_FC2 = FcBn(1000,  'elu', name='Decoder_FC2')
        self.out_FC = layers.Dense(self.input_size, activation='sigmoid', name='Decoder_FC3')

    def call(self, inputs, training=None, mask=None):
        x = self.encoder_FC1(inputs)
        x = layers.Dropout(rate=0.3)(x)
        x = self.encoder_FC2(x)
        x = layers.Dropout(rate=0.3)(x)
        features = self.latent(x)
        x = self.decoder_FC1(features)
        x = layers.Dropout(rate=0.3)(x)
        x = self.decoder_FC2(x)
        x = layers.Dropout(rate=0.3)(x)
        out = self.out_FC(x)
        return out

    def get_features(self, inputs):
        x = self.encoder_FC1(inputs)
        x = layers.Dropout(rate=0.3)(x)
        x = self.encoder_FC2(x)
        x = layers.Dropout(rate=0.3)(x)
        features = self.latent(x)
        return features

    def build_graph(self):
        input_ = tf.keras.layers.Input(shape=self.input_size)
        return tf.keras.Model(inputs=[input_], outputs=self.call(input_))


class FcBn(layers.Layer):
    def __init__(self, units: int, activation: str, **kwargs):
        super(FcBn, self).__init__(**kwargs)
        self.fc = layers.Dense(units=units)
        self.bn = layers.BatchNormalization()
        self.activate = layers.Activation(activation=activation)

    def call(self, inputs, **kwargs):
        x = self.fc(inputs)
        x = self.bn(x)
        out = self.activate(x)
        return out


if __name__ == '__main__':
    mod = DaeFC(gene_num=30129)
    plot_model(mod.build_graph(), show_shapes=True, to_file=r'C:\Users\Wangyk\Desktop\model.png')
