if __name__ == "__main__":
    import paddle.fluid as fluid
    import numpy as np
    import paddle
    num_filters = 3
    stride = 1
    filter_size = 2
    groups = 1
    name = "Conv"
    import unittest

    class TestDygraph(unittest.TestCase):
        def test(self):

            startup = fluid.Program()
            startup.random_seed = 111
            main = fluid.Program()
            main.random_seed = 111
            scope = fluid.core.Scope()
            place = fluid.CPUPlace()
            np.random.seed(1333)

            #input_np = np.random.random((1,3,2)).astype('float32')
            input_np = np.random.random((1, 1, 3, 2)).astype('float32')
            with fluid.scope_guard(scope):
                paddle.enable_static()
                fluid.default_startup_program().random_seed = 10
                fluid.default_main_program().random_seed = 10
                np.random.seed(1333)

                from paddle.fluid.param_attr import ParamAttr

                input = fluid.layers.data(
                    name="input", shape=[1, 3, 2], dtype="float32")

                conv = fluid.layers.conv2d(
                    input=input,
                    num_filters=3,
                    filter_size=2,
                    dilation=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    act=None,
                    bias_attr=False,
                    param_attr=ParamAttr(
                        initializer=fluid.initializer.Constant(1.23)),
                    name=name + '.conv2d.output.1')

                #print(conv)

                #linear = fluid.layers.fc(input=input, size=10, bias_attr=False, param_attr=None)

                exe = fluid.Executor(place=fluid.CPUPlace())
                exe.run(fluid.default_startup_program())
                #for p in fluid.default_startup_program().global_block().all_parameters():
                #    print(p.name)
                ret = exe.run(fetch_list=[conv, "Conv.conv2d.output.1.w_0"],
                              feed={"input": input_np})
                print("st conv:", np.sum(ret[0]))
                print("weights:", ret[1])

            with fluid.dygraph.guard(place):
                fluid.default_startup_program().random_seed = 10
                fluid.default_main_program().random_seed = 10
                np.random.seed(1333)

                class ConvBNLayer(paddle.nn.Layer):
                    def __init__(self,
                                 in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 groups=1,
                                 act=None,
                                 name="DY"):
                        super(ConvBNLayer, self).__init__()
                        self.conv = paddle.nn.Conv2D(
                            in_channels=in_channels,
                            out_channels=3,
                            kernel_size=2,
                            dilation=1,
                            stride=1,
                            padding=0,
                            groups=1,
                            weight_attr=ParamAttr(
                                initializer=fluid.initializer.Constant(1.23)),
                            bias_attr=False)

                    def forward(self, input):
                        print("dy:", self.conv.weight.numpy())
                        out = self.conv(input)
                        return out

                class Linear(paddle.nn.Layer):
                    def __init__(self):
                        super(Linear, self).__init__()
                        self.fc1 = paddle.nn.Linear(
                            in_features=6,
                            out_features=10,
                            bias_attr=False,
                            weight_attr=None)

                    def forward(self, input):
                        input = paddle.flatten(input, 0, -1)
                        print("shape:", input.shape)
                        out = self.fc1(input)
                        print("fc weights:", self.fc1.weight.numpy())
                        print("fc weights.shape;", self.fc1.weight.shape)
                        print("fc bias:", self.fc1.bias)
                        return out

                conv = ConvBNLayer(in_channels=1, out_channels=3, kernel_size=2)
                #linear = Linear()
                input = paddle.to_tensor(input_np)
                out = conv(input)
                #out = linear(input)
                print("dy:", np.sum(out.numpy()))

    unittest.main()
