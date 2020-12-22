# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import paddle
from paddle.nn import functional as F


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        support_character_type = ['ch', 'en', 'en_sensitive']
        assert character_type in support_character_type, "Only {} are supported now but get {}".format(
            support_character_type, self.character_str)

        self.beg_str = "sos"
        self.end_str = "eos"

        if character_type == "en":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == "ch":
            self.character_str = ""
            assert character_dict_path is not None, "character_dict_path should not be None when character_type is ch"
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif character_type == "en_sensitive":
            # same with ASTER setting (use 94 char).
            import string
            self.character_str = string.printable[:-6]
            dict_character = list(self.character_str)
        else:
            raise NotImplementedError
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=True):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        #print("ignored_tokens:", ignored_tokens)
        batch_size = len(text_index)
        char_list = []
        conf_list = []
        for batch_idx in range(batch_size):
            #char_list = []
            #conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    print("conf:",text_prob[batch_idx][idx].numpy())
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            #print("text:", text)
            result_list.append((text, conf_list))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             character_type, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        # out = self.decode_preds(preds)

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        #print("text:", text)
        #print("label:", label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def decode_preds(self, preds):
        probs_ind = np.argmax(preds, axis=2)

        B, N, _ = preds.shape
        l = np.ones(B).astype(np.int64) * N
        length = paddle.to_tensor(l)
        out = paddle.fluid.layers.ctc_greedy_decoder(preds, 0, length)
        batch_res = [
            x[:idx[0]] for x, idx in zip(out[0].numpy(), out[1].numpy())
        ]

        result_list = []
        for sample_idx, ind, prob in zip(batch_res, probs_ind, preds):
            char_list = [self.character[idx] for idx in sample_idx]
            valid_ind = np.where(ind != 0)[0]
            if len(valid_ind) == 0:
                continue
            conf_list = prob[valid_ind, ind[valid_ind]]
            result_list.append((''.join(char_list), conf_list))
        return result_list


class AttnLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(AttnLabelDecode, self).__init__(character_dict_path,
                                              character_type, use_space_char)
        self.beg_str = "sos"
        self.end_str = "eos"

    def add_special_char(self, dict_character):
        dict_character = [self.beg_str, self.end_str] + dict_character
        return dict_character

    def __call__(self, text):
        text = self.decode(text)
        return text

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx


class SRNLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self,
                 character_dict_path=None,
                 character_type='en',
                 use_space_char=False,
                 **kwargs):
        super(SRNLabelDecode, self).__init__(character_dict_path,
                                             character_type, use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        pred = preds['predict']
        if isinstance(preds, paddle.Tensor):
            pred = preds.numpy()
        pred = paddle.reshape(pred, shape=[-1,38])
        # out = self.decode_preds(preds)
        preds_idx = pred.argmax(axis=1)
        preds_prob = pred.max(axis=1)
        #end_pos = np.where(preds_idx.numpy() != 37)[0]
        #preds_idx = preds_idx[:int(end_pos[-1])+1]
        preds_idx = paddle.reshape(preds_idx, shape=[-1,25])
        preds_prob = paddle.reshape(preds_prob, shape=[-1,25])
        #print("before decode text:", preds_idx) 
        text = self.decode(preds_idx, preds_prob)
        #print("after decode text:", text)
        if label is None:
            return text
        #print("before decode:", label)
        label = self.decode(label, is_remove_duplicate=False)
        #print("after decoder:",label)
        return text, label

    def decode(self, text_index, text_prob=None, is_remove_duplicate=True):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        #print("ignored_tokens:", ignored_tokens)
        batch_size = len(text_index)
        #print("batch_size:", batch_size)
        #char_list = []
        #conf_list = []
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    #print("text prob:", text_prob)
                    conf_list.append(text_prob[batch_idx][idx].numpy())
                else:
                    conf_list.append(1)
            #if (batch_idx+1) % 25 == 0:
            #    text = ''.join(char_list)
            #    result_list.append((text, np.mean(conf_list)))
            #    char_list = []
            #    conf_list = []
            text = ''.join(char_list)
            #print("text:", text)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupport type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx
