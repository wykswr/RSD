import tensorflow.keras as keras


class FinalModel(keras.Model):
    def __init__(self, **kwargs):
        super(FinalModel, self).__init__()
        self.dae = kwargs['dae']
        self.frac = kwargs['frac']

    def call(self, inputs, training=None, mask=None):
        out1 = self.dae(inputs)
        out2 = self.frac(inputs)
        return [out1, out2]
