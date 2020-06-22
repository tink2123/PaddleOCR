import paddle.fluid as fluid
from config import *
import ResNet


class CRNN_paddle():
    def __init__(self):
        pass

    def ocr_convs(self,
                  img,
                  regularizer=None,
                  gradient_clip=None,
                  is_test=False,
                  use_cudnn=False):
        b = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0))
        w0 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0005))
        w1 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.01))
        tmp = img
        tmp = self.conv_bn_pool(
            tmp,
            2, [16, 16],
            param=w1,
            bias=b,
            param_0=w0,
            is_test=is_test,
            use_cudnn=use_cudnn)

        tmp = self.conv_bn_pool(
            tmp,
            2, [32, 32],
            param=w1,
            bias=b,
            is_test=is_test,
            use_cudnn=use_cudnn)
        tmp = self.conv_bn_pool(
            tmp,
            2, [64, 64],
            param=w1,
            bias=b,
            is_test=is_test,
            use_cudnn=use_cudnn)
        conv_features = self.conv_bn_pool(
            tmp,
            2, [128, 128],
            param=w1,
            bias=b,
            is_test=is_test,
            pooling=False,
            use_cudnn=use_cudnn)
        return conv_features

    def conv_bn_pool(self,
                     input,
                     group,
                     out_ch,
                     act="relu",
                     param=None,
                     bias=None,
                     param_0=None,
                     is_test=False,
                     pooling=True,
                     use_cudnn=False):
        tmp = input
        for i in range(group):
            tmp = fluid.layers.conv2d(
                input=tmp,
                num_filters=out_ch[i],
                filter_size=3,
                padding=1,
                param_attr=param if param_0 is None else param_0,
                act=None,  # LinearActivation
                use_cudnn=use_cudnn)
            tmp = fluid.layers.batch_norm(
                input=tmp,
                act=act,
                param_attr=param,
                bias_attr=bias,
                is_test=is_test)
        if pooling:
            tmp = fluid.layers.pool2d(
                input=tmp,
                pool_size=2,
                pool_type='max',
                pool_stride=2,
                use_cudnn=use_cudnn,
                ceil_mode=True)

        return tmp


class CRNN():
    def __init__(self):
        pass

    def ocr_convs(self,
                  img,
                  regularizer=None,
                  gradient_clip=None,
                  is_test=False,
                  use_cudnn=False):
        b = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0))
        w0 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0005))
        w1 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.01))

        #32
        x = fluid.layers.conv2d(
            img,
            64, (3, 3),
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=w1,
            bias_attr=b,
            name="backbone_block1_conv1")
        x = fluid.layers.pool2d(
            x, (2, 2),
            "max",
            pool_stride=2,
            use_cudnn=use_cudnn,
            name="backbone_block1_pool1")
        #16
        x = fluid.layers.conv2d(
            x,
            128, (3, 3),
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=w1,
            bias_attr=b,
            name="backbone_block2_conv1")
        x = fluid.layers.pool2d(
            x, (2, 2),
            "max",
            pool_stride=2,
            use_cudnn=use_cudnn,
            name="backbone_block2_pool1")
        #8
        x = fluid.layers.conv2d(
            x,
            256, (3, 3),
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=w1,
            bias_attr=b,
            name="backbone_block3_conv1")
        x = fluid.layers.batch_norm(
            x, param_attr=w1, bias_attr=b, name="backbone_block3_bn1")
        x = fluid.layers.conv2d(
            x,
            256, (3, 3),
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=w1,
            bias_attr=b,
            name="backbone_block3_conv2")
        x = fluid.layers.batch_norm(
            x, param_attr=w1, bias_attr=b, name="backbone_block3_bn2")
        x = fluid.layers.pool2d(
            x, (2, 2),
            "max",
            pool_stride=(2, 1),
            use_cudnn=use_cudnn,
            name="backbone_block3_pool1")
        #4
        x = fluid.layers.conv2d(
            x,
            512, (3, 3),
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=w1,
            bias_attr=b,
            name="backbone_block4_conv1")
        x = fluid.layers.batch_norm(
            x, param_attr=w1, bias_attr=b, name="backbone_block3_bn1")
        x = fluid.layers.conv2d(
            x,
            512, (3, 3),
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=w1,
            bias_attr=b,
            name="backbone_block4_conv2")
        x = fluid.layers.batch_norm(
            x, param_attr=w1, bias_attr=b, name="backbone_block3_bn2")
        x = fluid.layers.pool2d(
            x, (2, 2),
            "max",
            pool_stride=(2, 1),
            use_cudnn=use_cudnn,
            name="backbone_block3_pool1")
        #2
        x = fluid.layers.conv2d(
            x,
            512, (2, 2),
            padding=0,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=w1,
            bias_attr=b,
            name="backbone_block5_conv1")
        x = fluid.layers.batch_norm(
            x, param_attr=w1, bias_attr=b, name="backbone_block5_bn1")
        #x shape is [512,1,*]
        ret = fluid.layers.conv2d(
            input=x,
            num_filters=ModelHyperParams.d_model,
            filter_size=1,
            stride=1,
            act='relu',
            bias_attr=False,
            name='d_model')
        return ret


def resnet_50(img):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    mm = ResNet.ResNet50()
    conv = mm.net(img)
    ret = fluid.layers.conv2d(
        input=conv,
        num_filters=ModelHyperParams.d_model,
        filter_size=1,
        stride=1,
        act='relu',
        bias_attr=False,
        name='d_model')
    return ret


class conv8():
    def __init__(self):
        pass

    def net(self,
            input,
            class_dim=1000,
            regularizer=None,
            gradient_clip=None,
            is_test=False):
        b = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0))
        w0 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.0005))
        w1 = fluid.ParamAttr(
            regularizer=regularizer,
            gradient_clip=gradient_clip,
            initializer=fluid.initializer.Normal(0.0, 0.01))

        tmp = input
        # ====group 1======
        conv2 = conv_bn_pool(
            # tmp, 5, [16, 16, 16, 16, 16], is_test=is_test) #param=w1, bias=b, param_0=w0, is_test=is_test)
            # tmp, 5, [16, 16, 16, 16, 16], param=w1, bias=b, param_0=w0, is_test=is_test)
            tmp,
            2,
            [16, 16],
            is_test=is_test)
        # conv2 = self.ATT(conv2, atteMap)
        # print(tmp)
        # ====group 2======

        conv3 = conv_bn_pool(conv2, 2, [32, 32], is_test=is_test)
        # print(tmp)
        ##tmp = conv_bn_pool_senet(tmp, 5, [32, 32, 32, 32, 32], param=w1, bias=b, is_test=is_test, pooling=False)
        ###tmp = conv_bn_pool_senet(tmp, 5, [32, 32, 32, 32, 32], is_test=is_test, pooling=False)

        ##for i in range(3):
        ##    tmp = bottleneck_block(tmp, 32, 1, is_test=is_test)
        ##tmp = max_pool(tmp, 2, 2)
        # conv3 = self.ATT(conv3, atteMap)

        # ====group 3======

        conv4 = conv_bn_pool(conv3, 2, [64, 64], is_test=is_test)
        # conv4 = self.ATT(conv4, atteMap)
        # print(tmp)
        ##tmp = conv_bn_pool_senet(tmp, 5, [64, 64, 64, 64, 64], param=w1, bias=b, is_test=is_test, pooling=False)

        # tmp = conv_bn_pool_senet(tmp, 5, [64, 64, 64, 64, 64], is_test=is_test, pooling=False)
        ##for i in range(4):
        ##    tmp = bottleneck_block(tmp, 64, 1, is_test=is_test)
        ###tmp = max_pool(tmp, [2, 1], [2, 1])
        ##tmp = max_pool(tmp, 2, 2)

        # ====group 4======
        conv5 = conv_bn_pool(
            conv4, 2, [128, 128], is_test=is_test, pooling=False)
        # conv5 = self.ATT(conv5, atteMap)
        # print(tmp)
        ##tmp = conv_bn_pool_senet(tmp, 5, [128, 128, 128, 128, 128], param=w1, bias=b, is_test=is_test, pooling=False)
        # tmp = conv_bn_pool_senet(tmp, 5, [128, 128, 128, 128, 128], is_test=is_test, pooling=False)
        ##for i in range(6):
        ##    tmp = bottleneck_block(tmp, 128, 1, is_test=is_test)

        ###tmp = fluid.layers.control_flow.Print(tmp, print_tensor_lod=False)
        ###add dropout
        ##tmp = fluid.layers.dropout(x=tmp, dropout_prob=0.5, is_test=is_test)

        ret = fluid.layers.conv2d(
            input=conv5,
            num_filters=ModelHyperParams.d_model,
            filter_size=1,
            stride=1,
            act='relu',
            bias_attr=False,
            name='d_model')
        return ret


w_nolr = fluid.ParamAttr(trainable=True)


def conv_bn_pool(input,
                 group,
                 out_ch,
                 act="relu",
                 param=None,
                 bias=None,
                 param_0=None,
                 is_test=False,
                 pooling=True,
                 pooling2x1=False):
    tmp = input
    for i in range(group):
        tmp = fluid.layers.conv2d(
            input=tmp,
            num_filters=out_ch[i],
            filter_size=3,
            padding=1,
            param_attr=w_nolr,  #param if param_0 is None else param_0,
            bias_attr=w_nolr,
            act=None,  # LinearActivation
            use_cudnn=True)
        tmp = fluid.layers.batch_norm(
            input=tmp,
            act=act,
            param_attr=w_nolr,
            bias_attr=w_nolr,
            is_test=is_test)
    if pooling and not pooling2x1:
        tmp = fluid.layers.pool2d(
            input=tmp,
            pool_size=2,
            pool_type='max',
            pool_stride=2,
            use_cudnn=True,
            ceil_mode=True)
    if pooling and pooling2x1:
        tmp = fluid.layers.pool2d(
            input=tmp,
            pool_size=[2, 1],
            pool_type='max',
            pool_stride=[2, 1],
            use_cudnn=True,
            ceil_mode=True)

    return tmp


if __name__ == "__main__":

    import data_generator as dg

    batchsize = 2
    filelist = [('tensorflow.JPG', 'tensorflow')]
    max_len = 30
    vocab_fpath = 'vocab.txt'
    train_imgshape = (3, 224, 224)
    # train_dg = dg.datagen(filelist, batchsize,vocab_fpath, train_imgshape)
    #
    # data_generator = train_dg
    # tu = train_dg.__getitem__(0)
    # img_data = tu[0]
    # print(img_data.shape)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    train_prog = fluid.Program()
    startup_prog = fluid.Program()
    mycrnn = CRNN()

    myconv8 = conv8()
    with fluid.program_guard(train_prog, startup_prog):
        img = fluid.layers.data(
            name='image', shape=[1, 48, 512], dtype='float32')
        res = myconv8.net(img)
        print(res)
    exe.run(startup_prog)
    # exe.run(train_prog,feed={'image':img_data},fetch_list=[res])
