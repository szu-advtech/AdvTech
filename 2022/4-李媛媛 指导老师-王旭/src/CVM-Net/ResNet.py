from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation


class Resnet():
    def __init__(self, filters, strides=1, residual_path=False):
        super(Resnet, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding="same", use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation("relu")

        self.c2 = Conv2D(filters, (3, 3), strides==1, padding="same", use_bias=False)
        self.b2 = BatchNormalization()

        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding="same", use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation("relu")

    def call(self, inputs):
        residual = inputs

        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)
        return out
