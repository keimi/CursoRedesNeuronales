class Perceptron:
    w1=0.0
    w2=0.0
    bias=0.0

    def __init__(self, w1, w2, bias):
        self.w1 = w1
        self.w2 = w2
        self.bias = bias

    def set_weights(self, w1, w2):
        self.w2=w2
        self.w1=w1

    def set_bias(self, bias):
        self.bias = bias

    def process_input(self, i1, i2):

        if self.w1*i1 + self.w2*i2 + self.bias >= 0:
            return 1
        else:
            return 0

