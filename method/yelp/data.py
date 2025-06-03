from typing import List
import numpy as np
import os
import random
import torch
from torch.autograd import Variable
from nltk.translate.bleu_score import SmoothingFunction
import nltk
import json

use_cuda = torch.cuda.is_available()


def calc_bleu(reference, hypothesis):
    weights = (0.25, 0.25, 0.25, 0.25)
    return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weights,
                                                   smoothing_function=SmoothingFunction().method1)


def load_human_answer(data_path):
    ans = []
    file_list = [
        data_path + 'reference.0',
        data_path + 'reference.1',
    ]
    for file in file_list:
        with open(file) as f:
            for line in f:
                line = line.strip()
                line = line.split('\t')[1].split()
                parse_line = [int(x) for x in line]
                ans.append(parse_line)
    return ans


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def id2text_sentence(sen_id, id_to_word):
    sen_text = []
    max_i = len(id_to_word)
    for i in sen_id:
        if i == 3 or i == 0:  # id_eos
            break
        if i >= max_i:
            i = 1  # UNK
        sen_text.append(id_to_word[i])
    return ' '.join(sen_text)


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def get_cuda(tensor):
    if not torch.cuda.is_available():
        return tensor.to('cpu')
    return tensor.cuda()


def load_word_dict_info(word_dict_file, max_num):
    id_to_word = []
    with open(word_dict_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip()
            item_list = item.split('\t')
            word = item_list[0]
            if len(item_list) > 1:
                num = int(item_list[1])
                if num < max_num:
                    break
            id_to_word.append(word)
    print("Load word-dict with %d size and %d max_num." %
          (len(id_to_word), max_num))
    return id_to_word, len(id_to_word)


def load_data1(file1):
    token_stream = []
    with open(file1, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            token_stream.append(parse_line)
    return token_stream


def prepare_data(data_path, max_num, task_type):
    print("prepare data ...")
    id_to_word, vocab_size = load_word_dict_info(
        data_path + 'word_to_id.txt', max_num)

    # define train / test file
    train_file_list = []
    train_label_list = []
    if task_type == 'yelp' or task_type == 'amazon':
        train_file_list = [
            data_path + 'sentiment.train.0', data_path + 'sentiment.train.1',
            data_path + 'sentiment.dev.0', data_path + 'sentiment.dev.1',
        ]
        train_label_list = [
            [0],
            [1],
            [0],
            [1],
        ]

    return id_to_word, vocab_size, train_file_list, train_label_list


def pad_batch_seuqences(origin_seq, bos_id, eos_id, unk_id, max_seq_length, vocab_size, not_for_max=None):
    '''padding with 0, mask id_num > vocab_size with unk_id.
       not_for_max is the max for all ref 
    '''
    max_l = 0
    for i in origin_seq:
        max_l = max(max_l, len(i))
    if not_for_max is not None:
        max_l = max(max_l, not_for_max)
    max_l = min(max_seq_length, max_l + 1)
    # max_l = max_seq_length

    encoder_input_seq = np.zeros((len(origin_seq), max_l-1), dtype=int)
    decoder_input_seq = np.zeros((len(origin_seq), max_l), dtype=int)
    decoder_target_seq = np.zeros((len(origin_seq), max_l), dtype=int)
    encoder_input_seq_length = np.zeros((len(origin_seq)), dtype=int)
    decoder_input_seq_length = np.zeros((len(origin_seq)), dtype=int)

    for i in range(len(origin_seq)):
        decoder_input_seq[i][0] = bos_id
        for j in range(min(max_l-1, len(origin_seq[i]))):
            this_id = origin_seq[i][j]

            if this_id >= vocab_size:
                this_id = unk_id

            encoder_input_seq[i][j] = this_id
            decoder_input_seq[i][j + 1] = this_id
            decoder_target_seq[i][j] = this_id

        encoder_input_seq_length[i] = min(max_l-1, len(origin_seq[i]))
        decoder_input_seq_length[i] = min(max_l, len(origin_seq[i]) + 1)
        decoder_target_seq[i][decoder_input_seq_length[i]-1] = eos_id
    return encoder_input_seq, decoder_input_seq, decoder_target_seq, encoder_input_seq_length, decoder_input_seq_length


class non_pair_data_loader():
    def __init__(self, batch_size: int, id_bos: int, id_eos: int, id_unk: int, max_sequence_length: int, vocab_size: int):
        self.sentences_batches = []
        self.labels_batches = []

        self.src_batches = []
        self.src_mask_batches = []
        self.tgt_batches = []
        self.tgt_y_batches = []
        self.tgt_mask_batches = []
        self.ntokens_batches = []

        self.num_batch = 0
        self.batch_size = batch_size
        self.pointer = 0
        self.id_bos = id_bos
        self.id_eos = id_eos
        self.id_unk = id_unk
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size

    def create_batches(self, train_file_list, train_label_list, if_shuffle=True):
        self.data_label_pairs = []      # list of [sent, label] pairs
        for _index in range(len(train_file_list)):
            with open(train_file_list[_index]) as fin:
                for line in fin:
                    line = line.strip()
                    line = line.split()
                    parse_line = [int(x) for x in line]
                    self.data_label_pairs.append(
                        [parse_line, train_label_list[_index]])

        if if_shuffle:
            random.shuffle(self.data_label_pairs)

        # Split batches
        if self.batch_size == None:
            self.batch_size = len(self.data_label_pairs)
        self.num_batch = int(len(self.data_label_pairs) / self.batch_size)
        for _index in range(self.num_batch):
            item_data_label_pairs = self.data_label_pairs[_index*self.batch_size:(
                _index+1)*self.batch_size]
            item_sentences = [_i[0] for _i in item_data_label_pairs]
            item_labels = [_i[1] for _i in item_data_label_pairs]

            batch_encoder_input, batch_decoder_input, batch_decoder_target, \
                batch_encoder_length, batch_decoder_length = pad_batch_seuqences(
                    item_sentences, self.id_bos, self.id_eos, self.id_unk, self.max_sequence_length, self.vocab_size,)

            src = get_cuda(torch.tensor(batch_encoder_input, dtype=torch.long))
            tgt = get_cuda(torch.tensor(batch_decoder_input, dtype=torch.long))
            tgt_y = get_cuda(torch.tensor(
                batch_decoder_target, dtype=torch.long))

            src_mask = (src != 0).unsqueeze(-2)
            tgt_mask = self.make_std_mask(tgt, 0)
            ntokens = (tgt_y != 0).data.sum().float()

            self.sentences_batches.append(item_sentences)
            self.labels_batches.append(
                get_cuda(torch.tensor(item_labels, dtype=torch.float)))
            self.src_batches.append(src)
            self.tgt_batches.append(tgt)
            self.tgt_y_batches.append(tgt_y)
            self.src_mask_batches.append(src_mask)
            self.tgt_mask_batches.append(tgt_mask)
            self.ntokens_batches.append(ntokens)

        self.pointer = 0

    def make_std_mask(self, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    def next_batch(self):
        """take next batch by self.pointer"""
        this_batch_sentences = self.sentences_batches[self.pointer]
        this_batch_labels = self.labels_batches[self.pointer]

        this_src = self.src_batches[self.pointer]
        this_src_mask = self.src_mask_batches[self.pointer]
        this_tgt = self.tgt_batches[self.pointer]
        this_tgt_y = self.tgt_y_batches[self.pointer]
        this_tgt_mask = self.tgt_mask_batches[self.pointer]
        this_ntokens = self.ntokens_batches[self.pointer]

        self.pointer = (self.pointer + 1) % self.num_batch
        return this_batch_sentences, this_batch_labels, \
            this_src, this_src_mask, this_tgt, this_tgt_y, \
            this_tgt_mask, this_ntokens

    def reset_pointer(self):
        self.pointer = 0


if __name__ == '__main__':

    class Batch:
        "Object for holding a batch of data with mask during training."

        def __init__(self, src, trg=None, pad=0):
            self.src = src
            self.src_mask = (src != pad).unsqueeze(-2)
            if trg is not None:
                self.trg = trg[:, :-1]
                self.trg_y = trg[:, 1:]
                self.trg_mask = \
                    self.make_std_mask(self.trg, pad)
                self.ntokens = (self.trg_y != pad).data.sum()

        @staticmethod
        def make_std_mask(tgt, pad):
            "Create a mask to hide padding and future words."
            tgt_mask = (tgt != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & Variable(
                subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
            return tgt_mask

    def data_gen(V, batch, nbatches):
        "Generate random data for a src-tgt copy task."
        for i in range(nbatches):
            data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
            data[:, 0] = 1
            src = Variable(data, requires_grad=False)
            tgt = Variable(data, requires_grad=False)
            yield Batch(src, tgt, 0)

    for i in range(100):
        print("%d ----- " % i)
        data_iter = data_gen(10, 3, 2)
        for j, batch in enumerate(data_iter):
            print("%d:", j)
            print(batch.src)
            print(batch.src_mask)
            print(batch.trg)
            print(batch.trg_y)
            print(batch.trg_mask)
            input("=====")


class AttackDataLoader():
    def __init__(self, batch_size: int, id_bos: int, id_eos: int, id_unk: int, max_sequence_length: int, vocab_size: int):
        self.sentences_batches = []
        self.labels_batches = []

        self.src_batches = []
        self.src_mask_batches = []
        self.tgt_batches = []
        self.tgt_y_batches = []
        self.tgt_mask_batches = []
        self.ntokens_batches = []

        self.num_batch = 0
        self.batch_size = batch_size
        self.pointer = 0
        self.id_bos = id_bos
        self.id_eos = id_eos
        self.id_unk = id_unk
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size

    def create_batches(self, train_file_list: List, if_shuffle=False, origin_label='need', pos_num=None, neg_num=None):
        """
            origin_label==need --> need orgin ; 
            origin_label==only --> just need origin ;
            origin_label==not --> not need origin(make pos or neg data) ;
            origin_label==like --> sentences with the same style with origin  ;
        """
        self.sent_num = 0
        self.ref_num = 0
        self.sim_num = 0
        d = None
        data = []

        if origin_label == 'like' and pos_num == 0:
            return
        if origin_label == 'not'and neg_num == 0:
            return

        for train_file in train_file_list:
            with open(train_file, 'r') as fin:
                d = json.load(fin)
            # d['sent_num'] = min(d['sent_num'], 32)  #  for cpu easy test
            self.sent_num += d['sent_num']
            self.ref_num = d['ref_num'] if neg_num == None else neg_num
            self.sim_num = d['sim_num'] if pos_num == None else pos_num
            # if if_shuffle:
            #     random.shuffle(d['data'])
            data += d['data']

        self.num_batch = self.sent_num
        
        if(origin_label == 'need'):
            for i in range(self.sent_num):
                tmp_line = data[i]['origin'].strip().split()
                lines = [[int(x) for x in tmp_line]]
                for j in range(self.ref_num):
                    tmp_line = data[i]['ref'+str(j)].strip().split()
                    tmp_line = [int(x) for x in tmp_line]
                    lines.append(tmp_line)


                src, tgt, _, _, _ = pad_batch_seuqences(
                    lines, self.id_bos, self.id_eos, self.id_unk, self.max_sequence_length, self.vocab_size)

                # src = get_cuda(torch.tensor(src, dtype=torch.long))
                # tgt = get_cuda(torch.tensor(tgt, dtype=torch.long))
                src = torch.tensor(src, dtype=torch.long)
                tgt = torch.tensor(tgt, dtype=torch.long)

                src_mask = (src != 0).unsqueeze(-2)
                tgt_mask = self.make_std_mask(tgt, 0)

                self.src_batches.append(get_cuda(src))
                self.tgt_batches.append(get_cuda(tgt))
                self.src_mask_batches.append(get_cuda(src_mask))
                self.tgt_mask_batches.append(get_cuda(tgt_mask))
        else:
            # Split batches
            if self.batch_size == None:
                self.batch_size = self.sent_num
            self.num_batch = int(self.sent_num / self.batch_size)

            all_lines = []
            for i in range(self.sent_num):
                lines = []
                if(origin_label == 'only'):
                    tmp_line = data[i]['origin'].strip().split()
                    lines = [[int(x) for x in tmp_line]]
                if(origin_label == 'not'):
                    for j in range(self.ref_num):
                        tmp_line = data[i]['ref'+str(j)].strip().split()
                        tmp_line = [int(x) for x in tmp_line]
                        lines.append(tmp_line)
                if origin_label == 'like':
                    for j in range(self.sim_num):
                        tmp_line = data[i]['sim'+str(j)].strip().split()
                        tmp_line = [int(x) for x in tmp_line]
                        lines.append(tmp_line)
                    pass
                if(origin_label in ['not', 'like']):
                    all_lines.append(lines)
                else:
                    all_lines += lines

            if if_shuffle:
                random.shuffle(all_lines)

            for _index in range(self.num_batch):
                lines = all_lines[_index*self.batch_size:(
                    _index+1)*self.batch_size]
                if origin_label == 'not':
                    src, tgt, tgt_y = None, None, None
                    not_for_max = 0
                    for i in range(self.ref_num):
                        for j in range(len(lines)):
                            not_for_max = max(not_for_max, len(lines[j][i]))
                    for i in range(self.ref_num):
                        line_ref = []
                        for j in range(len(lines)):
                            line_ref.append(lines[j][i])
                        src_ref, tgt_ref, tgt_y_ref, _, _ = pad_batch_seuqences(
                            line_ref, self.id_bos, self.id_eos, self.id_unk, self.max_sequence_length, self.vocab_size, not_for_max=not_for_max)
                        if(src is None):
                            src, tgt, tgt_y = np.array(src_ref)[:, np.newaxis, :], np.array(tgt_ref)[:, np.newaxis, :], np.array(tgt_y_ref)[:, np.newaxis, :]
                        else:
                            src, tgt, tgt_y = np.concatenate((src, np.array(src_ref)[:, np.newaxis, :]), 1), np.concatenate((tgt, np.array(tgt_ref)[:, np.newaxis, :]), 1), \
                                              np.concatenate((tgt_y, np.array(tgt_y_ref)[:, np.newaxis, :]), 1)
                elif origin_label == 'like':
                    src, tgt, tgt_y = None, None, None
                    not_for_max = 0
                    for i in range(self.sim_num):
                        for j in range(len(lines)):
                            not_for_max = max(not_for_max, len(lines[j][i]))
                    for i in range(self.sim_num):
                        line_ref = []
                        for j in range(len(lines)):
                            line_ref.append(lines[j][i])
                        src_ref, tgt_ref, tgt_y_ref, _, _ = pad_batch_seuqences(
                            line_ref, self.id_bos, self.id_eos, self.id_unk, self.max_sequence_length, self.vocab_size, not_for_max=not_for_max)
                        if(src is None):
                            src, tgt, tgt_y = np.array(src_ref)[:, np.newaxis, :], np.array(tgt_ref)[:, np.newaxis, :], np.array(tgt_y_ref)[:, np.newaxis, :]
                        else:
                            src, tgt, tgt_y = np.concatenate((src, np.array(src_ref)[:, np.newaxis, :]), 1), np.concatenate((tgt, np.array(tgt_ref)[:, np.newaxis, :]), 1), \
                                              np.concatenate((tgt_y, np.array(tgt_y_ref)[:, np.newaxis, :]), 1)
                else:
                    src, tgt, tgt_y, _, _ = pad_batch_seuqences(
                        lines, self.id_bos, self.id_eos, self.id_unk, self.max_sequence_length, self.vocab_size)
                src = torch.tensor(src, dtype=torch.long)
                tgt = torch.tensor(tgt, dtype=torch.long)
                tgt_y = torch.tensor(tgt_y, dtype=torch.long)

                src_mask = (src != 0).unsqueeze(-2)
                tgt_mask = self.make_std_mask(tgt, 0)
                ntokens = (tgt_y != 0).data.sum().float()

                if(origin_label in ['not', 'like']):
                    src = src.permute(1, 0, 2)
                    tgt = tgt.permute(1, 0, 2)
                    tgt_y = tgt_y.permute(1, 0, 2)
                    src_mask = src_mask.permute(1, 0, 2, 3)
                    tgt_mask = tgt_mask.permute(1, 0, 2, 3)

                # self.src_batches.append(get_cuda(src))
                # self.tgt_batches.append(get_cuda(tgt))
                # self.tgt_y_batches.append(get_cuda(tgt_y))
                # self.src_mask_batches.append(get_cuda(src_mask))
                # self.tgt_mask_batches.append(get_cuda(tgt_mask))
                # self.ntokens_batches.append(get_cuda(ntokens))

                self.src_batches.append(src.cpu())
                self.tgt_batches.append(tgt.cpu())
                self.tgt_y_batches.append(tgt_y.cpu())
                self.src_mask_batches.append(src_mask.cpu())
                self.tgt_mask_batches.append(tgt_mask.cpu())
                self.ntokens_batches.append(ntokens.cpu())

        self.pointer = 0

    def make_std_mask(self, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    def next_batch(self, type=None):
        """
            type=None --> for just contrastive
            type='ae'--> for train ae_model
        """
        if self.src_batches == []:
            return None, None, None, None

        this_src = self.src_batches[self.pointer]
        this_tgt = self.tgt_batches[self.pointer]
        this_tgt_y = self.tgt_y_batches[self.pointer]
        this_src_mask = self.src_mask_batches[self.pointer]
        this_tgt_mask = self.tgt_mask_batches[self.pointer]
        this_ntokens = self.ntokens_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch

        if type is None:
            # return this_src, this_tgt, this_src_mask, this_tgt_mask
            return get_cuda(this_src), get_cuda(this_tgt), get_cuda(this_src_mask), get_cuda(this_tgt_mask)
        elif type == 'ae':
            # return this_src, this_tgt, this_src_mask, this_tgt_mask, this_tgt_y, this_ntokens
            return get_cuda(this_src), get_cuda(this_tgt), get_cuda(this_src_mask), get_cuda(this_tgt_mask), get_cuda(this_tgt_y), get_cuda(this_ntokens)

def expand_array(array, target: int):
    while 2 * len(array) < target:
        array = array + array

    residue = target - len(array)
    array += array[:residue]

    return array


def create_data_loader(train_file_list, train_label_list, batch_size, id_bos, id_eos, id_unk, max_sequence_length, vocab_size, if_shuffle=True):
    A_sent_label_pairs = []
    B_sent_label_pairs = []

    with open(train_file_list[0], 'r') as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            A_sent_label_pairs.append(
                [[int(x) for x in line], train_label_list[0]])

    with open(train_file_list[1], 'r') as fin:
        for line in fin:
            line = line.strip()
            line = line.split()
            B_sent_label_pairs.append(
                [[int(x) for x in line], train_label_list[1]])

    len_A_pair = len(A_sent_label_pairs)
    len_B_pair = len(B_sent_label_pairs)

    if if_shuffle:
        random.shuffle(A_sent_label_pairs)
        random.shuffle(B_sent_label_pairs)

    if len_A_pair < len_B_pair:
        A_sent_label_pairs = expand_array(A_sent_label_pairs, len_B_pair)
    if len_B_pair < len_A_pair:
        B_sent_label_pairs = expand_array(B_sent_label_pairs, len_A_pair)

    A_data = create_arr(A_sent_label_pairs, batch_size, id_bos,
                        id_eos, id_unk, max_sequence_length, vocab_size)
    B_data = create_arr(B_sent_label_pairs, batch_size, id_bos,
                        id_eos, id_unk, max_sequence_length, vocab_size)
    return A_data + B_data


def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask


def create_arr(data_label_pairs, batch_size, id_bos, id_eos, id_unk, max_sequence_length, vocab_size):
    arr_sentences, arr_id, arr_labels, \
        arr_src, arr_src_mask, arr_tgt, arr_tgt_y, \
        arr_tgt_mask, arr_ntokens, arr_src_len = [], [], [], [], [], [], [], [], [], []
    # logger.info(str(len(data_label_pairs)))
    print('Sentence-Label Pair: ', len(data_label_pairs))
    num_batch = int(len(data_label_pairs) / batch_size)
    data_set = []
    nowi = -1
    print('Batch Size: ', batch_size)

    # max_len = 0
    # for sent, _ in data_label_pairs:
    #     max_len = max(max_len, len(sent))
    # max_len = min(max_len, max_sequence_length)

    for _index in range(num_batch):
        item_data_label_pairs = data_label_pairs[_index *
                                                 batch_size:(_index + 1) * batch_size]
        item_sentences = [_i[0] for _i in item_data_label_pairs]
        item_labels = [_i[1] for _i in item_data_label_pairs]
        #print(item_labels, "haha")
        encoder_input, decoder_input, decoder_target, \
            encoder_length, decoder_length = pad_batch_seuqences(
                item_sentences, id_bos, id_eos, id_unk, max_sequence_length, vocab_size)
        src = get_cuda(torch.tensor(encoder_input, dtype=torch.long))
        tgt = get_cuda(torch.tensor(decoder_input, dtype=torch.long))
        tgt_y = get_cuda(torch.tensor(decoder_target, dtype=torch.long))

        src_mask = (src != 0).unsqueeze(-2)
        src_len = (src != 0).sum(dim=-1).squeeze()
        tgt_mask = make_std_mask(tgt, 0)
        #ntokens = (tgt_y != 0).data.sum().float()
        ntokens = (tgt_y != 0).data.sum(axis=1).float()
        """arr_sentences.append(item_sentences)
        arr_labels.append(get_cuda(torch.tensor(item_labels, dtype=torch.float)))
        arr_src.append(src)
        arr_tgt.append(tgt)
        arr_tgt_y.append(tgt_y)
        arr_src_mask.append(src_mask)
        arr_tgt_mask.append(tgt_mask)
        arr_ntokens.append(ntokens)"""
        arr_id += [i for i in range(_index * batch_size,
                                    (_index + 1) * batch_size)]
        arr_sentences += item_sentences
        arr_labels += [i for i in item_labels]
        arr_src += [i.data.cpu().numpy().tolist() for i in src]
        arr_tgt += [i.data.cpu().numpy().tolist() for i in tgt]
        arr_tgt_y += [i.data.cpu().numpy().tolist() for i in tgt_y]
        arr_src_len += [i.data.cpu().numpy().tolist() for i in src_len]
        arr_src_mask += [i.data.cpu().numpy().tolist() for i in src_mask]
        arr_tgt_mask += [i.data.cpu().numpy().tolist() for i in tgt_mask]
        # arr_ntokens += [i.data.cpu().numpy().tolist() for i in ntokens]

    print("ok")
    return [get_cuda(torch.tensor(arr_labels, dtype=torch.long)),
            get_cuda(torch.tensor(arr_src, dtype=torch.long)),
            get_cuda(torch.tensor(arr_tgt, dtype=torch.long)),
            get_cuda(torch.tensor(arr_src_mask, dtype=torch.long)),
            get_cuda(torch.tensor(arr_tgt_mask, dtype=torch.long))]
