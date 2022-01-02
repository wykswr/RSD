import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input


class DaeConv(keras.Model):
    def __init__(self, **kwargs):
        super(DaeConv, self).__init__()
        self.input_shapes = (kwargs['gene_num'],)
        self.original_len = self.input_shapes[0]
        self.mat2d_len = self.get_length(self.original_len)
        self.add_len = self.mat2d_len * self.mat2d_len - self.original_len

        self.encoder_cov1 = layers.Conv2D(4, (4, 4), strides=2, activation='relu', name='cov1')
        self.encoder_avep1 = layers.AveragePooling2D(pool_size=(2, 2), name='pl1')
        self.encoder_cov2 = layers.Conv2D(8, (4, 4), strides=2, activation='sigmoid', name='cov2')
        self.encoder_avep2 = layers.AveragePooling2D(pool_size=(2, 2), name='pl2')
        # self.encoder_cov3 = layers.Conv2D(16, (4, 4), strides=2, activation='sigmoid')
        # self.encoder_avep3 = layers.AveragePooling2D(pool_size=(2, 2))

        # self.decoder_up1 = layers.UpSampling2D((2, 2))
        # self.decoder_dconv1 = layers.Conv2DTranspose(8, (4, 4), strides=2, activation='elu')
        self.decoder_up2 = layers.UpSampling2D((2, 2), name='up2')
        self.decoder_dconv2 = layers.Conv2DTranspose(4, (5, 5), strides=2,activation='sigmoid', name='dcov2')
        self.decoder_up3 = layers.UpSampling2D((2, 2), name='up3')
        self.decoder_dconv3 = layers.Conv2DTranspose(1, (4, 4), strides=2, activation='relu', name='dcov3')

    def call(self, inputs, training=None, mask=None):
        x = tf.pad(inputs, [[0, 0], [0, self.add_len]])
        x = tf.reshape(x, [-1, self.mat2d_len, self.mat2d_len, 1])
        x = self.encoder_cov1(x)
        x = self.encoder_avep1(x)
        x = self.encoder_cov2(x)
        # x = self.encoder_avep2(x)
        # x = self.encoder_cov3(x)
        features = self.encoder_avep2(x)

        # x = self.decoder_up1(features)
        # x = self.decoder_dconv1(x)
        x = self.decoder_up2(features)
        x = self.decoder_dconv2(x)
        x = self.decoder_up3(x)
        x = self.decoder_dconv3(x)
        out = tf.reshape(x, [-1, self.original_len + self.add_len])[:, :self.original_len]
        return out

    def get_features(self, inputs):
        x = tf.pad(inputs, [[0, 0], [0, self.add_len]])
        x = tf.reshape(x, [-1, self.mat2d_len, self.mat2d_len, 1])
        x = self.encoder_cov1(x)
        x = self.encoder_avep1(x)
        x = self.encoder_cov2(x)
        # x = self.encoder_avep2(x)
        # x = self.encoder_cov3(x)
        features = self.encoder_avep2(x)
        features = keras.layers.Flatten()(features)
        return features

    def build_graph(self):
        input_ = tf.keras.layers.Input(shape=self.input_shapes)
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))

    def model(self):
        x = Input(shape=self.input_shapes)
        return tf.keras.Model(inputs=x, outputs=self.call(x))

    @staticmethod
    def get_length(len_1d: int) -> int:
        i = 0
        while i * i < len_1d:
            i += 1
        return i


if __name__ == '__main__':
    mod = DaeConv(gene_num=30129).model()
    print(mod.summary())


