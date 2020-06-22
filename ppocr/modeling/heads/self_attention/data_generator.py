import numpy as np
import random
from PIL import Image as IM
from config import *
import copy
import os
import logging
import cmdparser
from config import ModelHyperParams
import read_eng_table
import data_reader


def load_vocab(vocab_fpath):
    word_dict = {}
    word_dict1 = {}
    fdict = [
        '<s>', '<e>', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+',
        ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
        'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c',
        'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
        'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '<unk>'
    ]
    # fdict=['0','1','2','3','4',
    #         '5',
    #         '6',
    #         '7',
    #         '8',
    #         '9',
    #        'a',
    #         'b',
    #         'c',
    #         'd',
    #         'e',
    #         'f',
    #         'g',
    #         'h',
    #         'i',
    #         'j',
    #         'k',
    #         'l',
    #         'm',
    #         'n',
    #         'o',
    #         'p',
    #         'q',
    #         'r',
    #         's',
    #         't',
    #         'u',
    #         'v',
    #         'w',
    #         'x',
    #         'y',
    #         'z',
    #       '<s>',
    #        '<e>',
    #        '<unk>'
    # ]
    for idx, line in enumerate(fdict):
        word_dict[idx] = line
        word_dict1[line] = idx
    return word_dict1, word_dict


class datagen():
    def __init__(self, filepath, batchsize, vocab_fpath, train_imgshape,
                 imgs_dir):
        self.imgs_dir = imgs_dir
        self.char2idx, self.idx2char = load_vocab(vocab_fpath)
        self.filepath = filepath
        self.filelist = self.metafilerecord(
        )  #path,label,shape such as(\data\img.png,"helloworld",(115,30))
        self.batchsize = batchsize
        self.train_imgshape = train_imgshape
        logging.debug(self.train_imgshape)

        self.item = -1
        self.expand()
        #logging.debug(str(self.filelist))
        self.idxlist = self.shuffle()

        dg = data_reader.DataGenerator()
        self._reader = dg.train_reader(
            batchsize=batchsize,
            img_root_dir=imgs_dir,
            cycle=True,
            img_label_list=filepath)
        self.reader = self._reader()
        #print(next(self.reader))

    def metafilerecord(self):
        f = open(self.filepath)
        filelist = []
        # initial =['0','1','2','3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
        #                 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        initial = read_eng_table.get_map()
        #print(initial)
        if 'string' in self.filepath:
            for line in f.readlines():
                ret = line.strip().split()
                _label = ret[3]
                # _label= ''.join([initial[int(num)] for num in ret[3].split(',')])
                filelist.append((ret[2], _label, (int(ret[0]), int(ret[1]))))
        else:
            for line in f.readlines():
                ret = line.strip().split()
                # _label = ret[3]
                _label = ''.join([initial[num] for num in ret[3].split(',')])
                filelist.append((ret[2], _label, (int(ret[0]), int(ret[1]))))

        f.close()
        return filelist

    def expand(self):
        idxlist = list(range(0, len(self.filelist), self.batchsize))
        while len(self.filelist) < idxlist[-1] + self.batchsize:
            self.filelist.append(random.choice(self.filelist))

    def shuffle(self):
        random.shuffle(self.filelist)
        # logging.debug("the len of filelist is%d"%len(self.filelist))
        idxlist = list(range(0, len(self.filelist), self.batchsize))
        return idxlist

    def on_epoch_end(self):
        self.idxlist = self.shuffle()

    def __len__(self):
        return len(self.idxlist)

    def __iter__(self):
        return self

    def __getitem__(self, item):
        input = np.zeros(
            (self.batchsize, self.train_imgshape[0], self.train_imgshape[2],
             self.train_imgshape[1]),
            dtype='float32')
        y = []
        decoder_input = []
        labels = []

        i = self.idxlist[item]
        for j in range(self.batchsize):
            file = self.filelist[i][0]
            label = self.filelist[i][1]
            img = self.read(file)
            input[j] = img.transpose([2, 0, 1]).astype(
                np.float32)  #w,h,c ==> c,w,h
            y_single, decoder_i_single = self.decode(label)
            y.append(decoder_i_single)
            decoder_input.append(y_single)
            labels.append(label)
            i = i + 1

        #databuffer =  next(self.reader)
        #print(databuffer)
        #if databuffer ==None:
        #    raise StopIteration
        # [img, [SOS] + label, label + [EOS]] = databuffer
        #img_list = []
        #for elem in databuffer:
        #    img_list.append( np.expand_dims(elem[0],0) )
        #    decoder_input.append( elem[1] )
        #    y.append( elem[2] )
        #    labels.append(''.join([self.idx2char[num] for num in elem[1][1:]]))

        #input = np.concatenate(img_list,axis=0).astype('float32')
        return input, decoder_input, y, labels

    def __next__(self):
        if self.item + 1 < self.__len__():
            self.item = self.item + 1
            return self.__getitem__(self.item)
        else:
            self.item = -1
            self.on_epoch_end()
            raise StopIteration

    def next(self):
        try:
            return self.__next__()
        except Exception as StopIteration:
            raise StopIteration

    def read(self, filepath):
        img = IM.open(os.path.join(self.imgs_dir, filepath))
        #sample robust
        img = img.convert('L')
        img_new = img.resize(self.train_imgshape[1:])  #reshape the size
        img_np = np.asarray(img_new)
        r, c = img_np.shape
        img_np = np.reshape(img_np, (r, c, 1))
        #print(img_np.shape)
        if random.random() > 0.5:  # reverse the color of front and background
            img_np = 255 - img_np
        return img_np

    def decode(self, label):
        ret = []
        ret.append(ModelHyperParams.bos_idx)  # at the begin of label,insert <s>
        for char in label:
            ret.append(self.char2idx[char])
        ret.append(ModelHyperParams.eos_idx)  #at the begin of label,insert <e>
        return copy.copy(ret[:-1]), copy.copy(ret[1:])


if __name__ == "__main__":
    args = cmdparser.parse_args()
    batchsize = TrainTaskConfig.batch_size
    print(args.train_filelist)
    vocab_fpath = args.trg_vocab_fpath
    train_imgshape = (3, 100, 30)
    train_dg = datagen(args.train_filelist, batchsize, vocab_fpath,
                       train_imgshape, args.imgs_dir)
    for data in train_dg:
        print(data[1:])
