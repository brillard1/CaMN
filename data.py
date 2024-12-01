import os
import torch
import numpy as np
from imageio import imread
from PIL import Image
import glob
import math

from termcolor import colored, cprint

from preprocess import clean_text

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision.utils as vutils
from torchvision import datasets

from transformers import BertTokenizer, XLNetTokenizer, ElectraTokenizer, AlbertTokenizer
from preprocess import clean_text

from base_dataset import BaseDataset
from base_dataset import expand2square

from paths import dataroot

from utils import move_to_cuda, reformat_text_data

task_dict = {
    'task1': 'informative',
    'task2': 'humanitarian',
    'task2_merged': 'humanitarian',
    'task3' : 'damage'
}

labels_task1 = {
    'informative': 1,
    'not_informative': 0
}

labels_task2 = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 6,
    'missing_or_found_people': 7,
}

labels_task2_merged = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 5,
    'missing_or_found_people': 5,
}

labels_task3 = {
    'little_or_no_damage': 0,
    'mild_damage': 1,
    'severe_damage' : 2
}

""" Crisis MMD """

import random
import torch
import pickle as pkl
import six
import ijson
import tensorflow as tf
import numpy as np

import json

PAD, UNK = '<PAD>', '<UNK>'
CLS = '<CLS>'
STR, END = '<STR>', '<END>'
SEL, rCLS, TL = '<SEL>', '<rCLS>', '<TL>'

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def ListsToTensor(xs, vocab=None, local_vocabs=None, unk_rate=0.):
    pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs): # change x value to lower(x)
        y = toIdx(x, i) + [pad] * (max_len - len(x))
        ys.append(y)
    data = torch.LongTensor(ys).t_().contiguous()
    return data


def ListsofStringToTensor(xs, vocab, max_string_len=30):  # 다 토큰들의 아이디로 들어간다.
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs: # change x value to lower(x)
        y = x + [PAD] * (max_len - len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([STR] + z + [END]) + [vocab.padding_idx] * (max_string_len - len(z)))
        ys.append(zs)

    data = torch.LongTensor(ys).transpose(0, 1).contiguous()
    return data


def ArraysToTensor(xs):
    x = np.array([list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis=0))
    data = np.zeros(shape, dtype=int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i + 1)] + [slice(0, x) for x in slicing_shape])
        data[slices] = x
        tensor = torch.from_numpy(data).long()
    return tensor


class InputExample(object):
  """A Single Input Sample."""

  def __init__(
      self,
      amr_id,
      text,
      text_tokens,
      img,
      label_text,
      label_img,
      label,
      attribute,
  ):
    """Construct an instance."""

    self.amr_id = amr_id
    self.text = text
    self.text_tokens = text_tokens
    self.img = img
    self.label_text = label_text
    self.label_img = label_img
    self.label = label
    self.attribute = attribute

class LexicalMap(object):

    def __init__(self):
        pass

    def get(self, concept, vocab=None):
        cp_seq = []
        for conc in concept:
            cp_seq.append(conc)

        if vocab is None:
            return cp_seq
        new_tokens = set(cp for cp in cp_seq if vocab.token2idx(cp) == vocab.unk_idx)
        token2idx, idx2token = dict(), dict()
        nxt = vocab.size
        for x in new_tokens:
            token2idx[x] = nxt
            idx2token[nxt] = x
            nxt +=1

        return cp_seq, token2idx, idx2token

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials=None):

        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        self._priority = dict()
        num_tot_tokens = 0
        num_vocab_tokens = 0
        for line in open(filename).readlines():
            try:
                token, cnt = line.strip().split('\t')
                cnt = int(cnt)
                num_tot_tokens += cnt
            except:
                print(line)

            if cnt >= min_occur_cnt:
                idx2token.append(token)
                num_vocab_tokens += cnt
            self._priority[token] = int(cnt)
        self.coverage = num_vocab_tokens / num_tot_tokens
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    def priority(self, x):
        return self._priority.get(x, 0)

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]

        return self._token2idx.get(x, self.unk_idx)

def relation_encoder(vocabs, data, train_flag=True):
    if train_flag:
        all_relations = dict()
        cls_idx = vocabs['relation'].token2idx(CLS)
        rcls_idx = vocabs['relation'].token2idx(rCLS)
        self_idx = vocabs['relation'].token2idx(SEL)

        all_relations[tuple([cls_idx])] = 0
        all_relations[tuple([rcls_idx])] = 1
        all_relations[tuple([self_idx])] = 2

        _relation_type = []

        for bidx, x in enumerate(data):
            n = len(x.attribute['concept'])
            brs = [[2] + [0] * (n)]
            for i in range(n):
                rs = [1]
                for j in range(n):
                    all_path = x.attribute['relation'][str(i)][(str(j))]
                    path = random.choice(all_path)['edge']
                    if len(path) == 0:  # self loop
                        path = [SEL]
                    if len(path) > 8:  # too long distance
                        path = [TL]

                    path = tuple(vocabs['relation'].token2idx(path))
                    rtype = all_relations.get(path, len(all_relations))
                    if rtype == len(all_relations):
                        all_relations[path] = len(all_relations)
                    rs.append(rtype)

                rs = np.array(rs, dtype=int)
                brs.append(rs)

            brs = np.stack(brs)
            _relation_type.append(brs)
        _relation_type = ArraysToTensor(_relation_type).transpose_(0, 2)  # 얘가 나중에 batch로 반환되는 relation애들이다.

        B = len(all_relations)

        _relation_bank = dict()
        _relation_length = dict()

        for k, v in all_relations.items():
            # relation
            _relation_bank[v] = np.array(k, dtype=int)
            _relation_length[v] = len(k)

        _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
        _relation_length = [_relation_length[i] for i in range(len(all_relations))]
        _relation_bank = ArraysToTensor(_relation_bank).t_()
        _relation_length = torch.LongTensor(_relation_length)


    else:
        all_relations = dict()
        cls_idx = vocabs['relation'].token2idx(CLS)
        rcls_idx = vocabs['relation'].token2idx(rCLS)
        self_idx = vocabs['relation'].token2idx(SEL)
        pad_idx = vocabs['relation'].token2idx(PAD)

        all_relations[tuple([pad_idx])] = 0
        all_relations[tuple([cls_idx])] = 1
        all_relations[tuple([rcls_idx])] = 2
        all_relations[tuple([self_idx])] = 3

        _relation_type = []
        record = []
        bsz, num_concepts, num_paths = 0, 0, 0

        for bidx, x in enumerate(data):
            n = len(x.attribute['concept'])
            num_concepts = max(n + 1, num_concepts)
            brs = [[[3]] + [[1]] * (n)]
            for i in range(n):
                rs = [[2]]
                for j in range(n):
                    all_r = []
                    all_path = x.attribute['relation'][str(i)][str(j)]
                    path0 = all_path[0]['edge']
                    if len(path0) == 0 or len(path0) > 8:
                        all_path = all_path[:1]
                    for path in all_path:
                        path = path['edge']
                        if len(path) == 0:  # self loop
                            path = [SEL]
                        if len(path) > 8:  # too long distance
                            path = [TL]
                        path = tuple(vocabs['relation'].token2idx(path))
                        rtype = all_relations.get(path, len(all_relations))
                        if rtype == len(all_relations):
                            all_relations[path] = len(all_relations)
                        all_r.append(rtype)

                    record.append(len(all_r))
                    num_paths = max(len(all_r), num_paths)
                    rs.append(all_r)
                brs.append(rs)
            _relation_type.append(brs)
        bsz = len(_relation_type)
        _relation_matrix = np.zeros((bsz, num_concepts, num_concepts, num_paths))

        for b, x in enumerate(_relation_type):
            for i, y in enumerate(x):
                for j, z in enumerate(y):
                    for k, r in enumerate(z):
                        _relation_matrix[b, i, j, k] = r

        _relation_type = torch.from_numpy(_relation_matrix).transpose_(0, 2).long()

        B = len(all_relations)
        _relation_bank = dict()
        _relation_length = dict()

        for k, v in all_relations.items():
            _relation_bank[v] = np.array(k, dtype=int)
            _relation_length[v] = len(k)
            # print(k, v)

        _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
        _relation_length = [_relation_length[i] for i in range(len(all_relations))]
        _relation_bank = ArraysToTensor(_relation_bank).t_()
        _relation_length = torch.LongTensor(_relation_length)

    return _relation_type ,_relation_bank, _relation_length


def cn_relation_encoder(args, vocabs, data, train_flag=True):

    data = [x.attribute for x in data]

    if train_flag:
        _cn_relation_bank_total = []
        _cn_relation_length_total = []
        _cn_relation_type_total = []

        for bidx, x, in enumerate(data):
            all_relations = dict()
            cls_idx = vocabs['relation'].token2idx(CLS)
            rcls_idx = vocabs['relation'].token2idx(rCLS)
            self_idx = vocabs['relation'].token2idx(SEL)

            all_relations[tuple([cls_idx])] = 0
            all_relations[tuple([rcls_idx])] = 1
            all_relations[tuple([self_idx])] = 2
            _relation_type = []

            for oidx in range(args.n_answers):
                n = len(x['cn_concept'][oidx])
                brs = [[2] + [0] * (n)]
                for i in range(n):
                    rs = [1]
                    for j in range(n):
                        all_path = x['cn_relation'][oidx][str(i)][(str(j))]
                        path = random.choice(all_path)['edge']
                        if len(path) == 0:  # self loop
                            path = [SEL]
                        if len(path) > 8:  # too long distance
                            path = [TL]
                        # change path to lower(path)
                        path = tuple(vocabs['relation'].token2idx(path))
                        rtype = all_relations.get(path, len(all_relations))
                        if rtype == len(all_relations):
                            all_relations[path] = len(all_relations)
                        rs.append(rtype)

                    rs = np.array(rs, dtype=int)
                    brs.append(rs)

                brs = np.stack(brs)
                _relation_type.append(brs)
            _relation_type = ArraysToTensor(_relation_type).transpose_(0, 2)

            B = len(all_relations)
            _relation_bank = dict()
            _relation_length = dict()

            for k, v in all_relations.items():

                # relation
                _relation_bank[v] = np.array(k, dtype=int)
                _relation_length[v] = len(k)

            _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
            _relation_length = [_relation_length[i] for i in range(len(all_relations))]
            _relation_bank = ArraysToTensor(_relation_bank).t_()
            _relation_length = torch.LongTensor(_relation_length)

            _cn_relation_bank_total.append(_relation_bank)
            _cn_relation_length_total.append(_relation_length)
            _cn_relation_type_total.append(_relation_type)


    else:
        _cn_relation_bank_total = []
        _cn_relation_length_total = []
        _cn_relation_type_total = []

        for bidx, x in enumerate(data):
            all_relations = dict()
            all_nodes = dict()
            cls_idx = vocabs['relation'].token2idx(CLS)
            rcls_idx = vocabs['relation'].token2idx(rCLS)
            self_idx = vocabs['relation'].token2idx(SEL)
            pad_idx = vocabs['relation'].token2idx(PAD)

            all_relations[tuple([pad_idx])] = 0
            all_relations[tuple([cls_idx])] = 1
            all_relations[tuple([rcls_idx])] = 2
            all_relations[tuple([self_idx])] = 3

            all_nodes[tuple([pad_idx])] = 0
            all_nodes[tuple([cls_idx])] = 1
            all_nodes[tuple([rcls_idx])] = 2
            all_nodes[tuple([self_idx])] = 3

            _relation_type = []
            record = []
            bsz, num_concepts, num_paths = 0, 0, 0

            for oidx in range(args.n_answers):
                n = len(x['cn_concept'][oidx])
                num_concepts = max(n + 1, num_concepts)
                brs = [[[3]] + [[1]] * (n)]
                for i in range(n):
                    rs = [[2]]
                    for j in range(n):
                        all_r = []
                        all_path = x['cn_relation'][oidx][str(i)][str(j)]

                        path0 = all_path[0]['edge']

                        if len(path0) == 0 or len(path0) > 8:
                            all_path = all_path[:1]

                        for path in all_path:
                            node = path['node']
                            path = path['edge']

                            if len(path) == 0:  # self loop
                                path = [SEL]
                            if len(path) > 8:  # too long distance
                                path = [TL]
                            # change path to lower(path)
                            path = tuple(vocabs['relation'].token2idx(path))
                            rtype = all_relations.get(path, len(all_relations))

                            if rtype == len(all_relations):
                                all_relations[path] = len(all_relations)
                                all_nodes[path] = node
                            all_r.append(rtype)

                        record.append(len(all_r))
                        num_paths = max(len(all_r), num_paths)
                        rs.append(all_r)
                    brs.append(rs)
                _relation_type.append(brs)
            bsz = len(_relation_type)
            _relation_matrix = np.zeros((bsz, num_concepts, num_concepts, num_paths))

            for b, x in enumerate(_relation_type):
                for i, y in enumerate(x):
                    for j, z in enumerate(y):
                        for k, r in enumerate(z):
                            _relation_matrix[b, i, j, k] = r

            _relation_type = torch.from_numpy(_relation_matrix).transpose_(0, 2).long()

            B = len(all_relations)
            _relation_bank = dict()
            _relation_length = dict()

            for k, v in all_relations.items():
                _relation_bank[v] = np.array(k, dtype=int)
                _relation_length[v] = len(k)

            _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]

            _relation_length = [_relation_length[i] for i in range(len(all_relations))]
            _relation_bank = ArraysToTensor(_relation_bank).t_()
            _relation_length = torch.LongTensor(_relation_length)

            _cn_relation_bank_total.append(_relation_bank)
            _cn_relation_length_total.append(_relation_length)
            _cn_relation_type_total.append(_relation_type)
    return _cn_relation_type_total,_cn_relation_bank_total, _cn_relation_length_total

def batchify(args, data, vocabs, unk_rate=0., train=True):

    ####################### Question ############################
    _conc = ListsToTensor([[CLS] + x.attribute['concept'] for x in data], vocabs['concept'],
                          unk_rate=unk_rate)

    _conc_char = ListsofStringToTensor([[CLS] + x.attribute['concept'] for x in data], vocabs['concept_char'])


    _depth = ListsToTensor([[0] + x.attribute['depth'] for x in data])

    # _argsions = [x.attribute['argsion'] for x in data]
    # Preparing data and label

    _text_tokens = [x.text_tokens for x in data]
    _text_tokens = reformat_text_data(_text_tokens)
    _img = torch.tensor([x.img for x in data])

    _label_text = torch.tensor([x.label_text for x in data])
    _label_img = torch.tensor([x.label_img for x in data])
    _label = torch.tensor([x.label for x in data])

    #print('data types:', _text_tokens, _img, _label)

    _id = [x.amr_id for x in data]
    _token_data = [x.text.split(' ') for x in data]
    if args.omcs:
        _evidence = [x.attribute['omcs'] for x in data]
    else:
        _evidence = ''

    local_token2idx = [x.attribute['token2idx'] for x in data]
    local_idx2token = [x.attribute['idx2token'] for x in data]

    augmented_token = [[STR] + x.attribute['token'] + [END] for x in data]

    _token_in = ListsToTensor(augmented_token, vocabs['token'], unk_rate=unk_rate)[:-1]
    _token_char_in = ListsofStringToTensor(augmented_token, vocabs['token_char'])[:-1]

    _token_out = ListsToTensor(augmented_token, vocabs['token'], local_token2idx)[1:]
    _cp_seq = ListsToTensor([x.attribute['cp_seq'] for x in data], vocabs['token'], local_token2idx)

    abstract = [x.attribute['abstract'] for x in data]

    _relation_type, _relation_bank, _relation_length = relation_encoder(vocabs, data, train_flag=train)

    ####################### argsions ############################
    _cn_concs = []
    _cn_conc_chars = []
    _cn_depths = []
    _cn_relation_types = []
    _cn_relation_banks = []
    _cn_relation_lengths = []


    for b in data:

        _cn_conc = ListsToTensor([[CLS] + x for x in b.attribute['cn_concept']], vocabs['concept'],
                                 unk_rate=unk_rate)
        _cn_conc_char = ListsofStringToTensor([[CLS] + x for x in b.attribute['cn_concept']], vocabs['concept_char'])
        _cn_depth = ListsToTensor([[0] + x for x in b.attribute['cn_depth']])

        _cn_concs.append(_cn_conc)
        _cn_conc_chars.append(_cn_conc_char)
        _cn_depths.append(_cn_depth)

    _cn_relation_type, _cn_relation_bank, _cn_relation_length = cn_relation_encoder(args, vocabs, data, train_flag=train)

    ret = {
        'concept': _conc,
        'concept_char': _conc_char,
        'concept_depth': _depth,
        'relation': _relation_type,
        'relation_bank': _relation_bank,
        'relation_length': _relation_length,
        'cn_concept': _cn_concs,
        'cn_concept_char':_cn_conc_chars,
        'cn_concept_depth': _cn_depths,
        'cn_relation': _cn_relation_type,
        'cn_relation_bank':_cn_relation_bank,
        'cn_relation_length':_cn_relation_length,
        'local_idx2token': local_idx2token,
        'local_token2idx': local_token2idx,
        'token_in': _token_in,
        'token_char_in': _token_char_in,
        'token_out': _token_out,
        'cp_seq': _cp_seq,
        'abstract': abstract,
        'token_data': _token_data,
        'label_text': _label_text,
        'label_image': _label_img,
        'label': _label,
        'id': _id,
        'evidence': _evidence, 
        # other features
        'text_tokens': _text_tokens, 
        'image': _img
    }

    return ret


def read_jsonl(input_file):
    with tf.gfile.Open(input_file, 'r') as f:

        return [json.loads(ln) for ln in f]


def read_large_file(objects, block_size=50):
    block = []
    for i, line in enumerate(objects):
        block.append(line)
        if len(block) == block_size:
            yield block
            block = []

    # don't forget to yield the last block
    if block:
        yield block

class DataLoader(object):
    def __init__(self, args, vocabs, lex_map, filename, batch_size, flag, 
                phase='train', cat='all', task='task2', shuffle=False, consistent_only=False     # For CRISIS MMD
            ):

        self.lex_map = lex_map
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.flag = flag
        self.unk_rate = 0.
        self.record_flag = False
        self.args = args
        self.filename = filename

        """CRISISMMD Dataset"""
        self.args = args
        self.shuffle = shuffle
        self.consistent_only = consistent_only
        self.dataset_root = f'{dataroot}/CrisisMMD_v2.0' if args.debug else f'{dataroot}/CrisisMMD_v2.0'
        self.image_root = f'{self.dataset_root}/data_image'
        self.label_map = None
        self.task = task
        if self.task == 'task1':
            self.label_map = labels_task1
        elif self.task == 'task2':
            self.label_map = labels_task2
        elif self.task == 'task2_merged':
            self.label_map = labels_task2_merged
        elif self.task == 'task3':
            self.label_map = labels_task3

        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
        ann_file = '%s/crisismmd_datasplit_all/AMR_main/task_%s_text_img_%s.tsv' % (
            self.dataset_root, task_dict[task], phase
        ) # defined in main folder

        # Append list of data to self.data_list
        self.read_data(ann_file)

        if self.shuffle:
            np.random.default_rng(seed=0).shuffle(self.data_list)
        self.data_list = self.data_list[:self.args.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.data_list)), 'yellow')

        self.N = len(self.data_list)

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda img: expand2square(img)),
            transforms.Resize((args.load_size, args.load_size)),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomCrop((args.crop_size, args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        """ CRISIS MMD """

    def read_data(self, ann_file):
        print(f'reading annotions from {ann_file}')
        with open(ann_file, encoding='utf-8') as f:
            self.info = f.readlines()[1:]

        self.data_list = []

        for l in self.info:
            l = l.rstrip('\n')
            if self.task == 'task3':
                event_name, tweet_id, image_id, tweet_text , image, label, _, _  = l.split(
                '\t')
            else:
                event_name, tweet_id, image_id, tweet_text , image, label, label_text, label_image, label_text_image, _  = l.split(
                    '\t')
            
            if self.consistent_only and label_text != label_image:
                continue
            #print(trigger_words)
            if self.task == 'task3':
                self.data_list.append(
                    {
                        'path_image': '%s/%s' % (self.dataset_root, image),

                        'text': tweet_text,
                        'text_tokens': self.tokenize(tweet_text),

                        'label_str': label,
                        'label': self.label_map[label],

                        'image_id': image_id,
                        'label_image_str': label,
                        'label_image': self.label_map[label],

                        'tweet_id': tweet_id,
                        'label_text_str': label,
                        'label_text': self.label_map[label]
                    }
                )
                
            else:
                self.data_list.append(
                    {
                        'path_image': '%s/%s' % (self.dataset_root, image),

                        'text': tweet_text,
                        'text_tokens': self.tokenize(tweet_text),

                        'label_str': label,
                        'label': self.label_map[label],

                        'image_id': image_id,
                        'label_image_str': label_image,
                        'label_image': self.label_map[label_image],

                        'tweet_id': tweet_id,
                        'label_text_str': label_text,
                        'label_text': self.label_map[label_text]
                    }
                )

        self.data_dict = {}
        for i in range(len(self.data_list)):
            id = self.data_list[i]['tweet_id']
            self.data_dict[id] = self.data_list[i]

    def tokenize(self, sentence):
        ids = self.tokenizer(clean_text(
            sentence), padding='max_length', max_length=40, truncation=True).items()
        return {k: torch.tensor(v) for k, v in ids}
    
    def transit_same_class(self, curr_class, curr_idx):
        while True:
            target_idx = np.random.choice(self.class_dict[curr_class])
            if target_idx != curr_idx:
                return self.data_list[target_idx]

    def get_transit_data(self, curr_class, curr_idx):
        should_keep_same_class = self.should_do(self.p_img_conn)
        if should_keep_same_class:
            # Transit to the same class, but not to itself
            target_data = self.transit_same_class(curr_class, curr_idx)
        else:
            # Transit to another class
            rand_idx = np.random.choice(len(self.transition_probs[curr_class]), p=[
                p[1] for p in self.transition_probs[curr_class]])
            target_class = self.transition_probs[curr_class][rand_idx][0]
            target_data_idx = np.random.choice(self.class_dict[target_class])
            target_data = self.data_list[target_data_idx]
        return target_data
    
    def set_unk_rate(self, x):
        self.unk_rate = x

    def record(self):
        self.record_flag = True

    def __len__(self):
        return math.ceil(len(self.data_list) / self.batch_size)

    def __iter__(self):
        with open(self.filename, 'rb') as input_file:
            objects = ijson.items(input_file, 'item')

            for block in read_large_file(objects, 16):
                fin_examples = []
                
                for i, d in enumerate(block):
                    cp_seq, token2idx, idx2token = self.lex_map.get(d['concept'], self.vocabs['concept'])
                    d['cp_seq'] = cp_seq
                    d['token2idx'] = token2idx
                    d['idx2token'] = idx2token

                    text = d['sentences']
                    amr_id = d['id']

                    amr_id = str(amr_id)

                    # check data_list for amr_id
                    cont_flag = False
                    cont_flag = self.data_dict.get(amr_id, None)

                    if not cont_flag:
                        continue

                    img_path = self.data_dict[amr_id]['path_image']
                    with Image.open(img_path).convert('RGB') as img:
                        image = self.transforms(img)

                    # Convert to list for batch processing
                    img = image.tolist()

                    text_tokens = self.data_dict[amr_id]['text_tokens']

                    # Getting text corresponding to AMR ID
                    label_text = self.data_dict[amr_id]['label_text']
                    label_img = self.data_dict[amr_id]['label_image']
                    label = self.data_dict[amr_id]['label']

                    # label = d['answer_key']
                    if self.args.omcs:
                        ex = [ex for ex in self.examples if ex['id'] == amr_id][0]
                        omcss = [
                            choice['omcs'][0] for choice in ex['question']['choices']]

                        d['omcs'] = omcss

                    line_ex = InputExample(amr_id, text, text_tokens, img, label_text, label_img, label, d)
                    fin_examples.append(line_ex)

                idx = list(range(len(fin_examples)))
        
                batches = []
                num_tokens, data = 0, []
                for i in idx:
                    data.append(fin_examples[i])

                    if i % self.batch_size == (self.batch_size - 1):
                        batches.append(data)
                        num_tokens, data = 0, []
                
                # if self.batch_size > 15:
                #     print(f'Reduce batch_size, current batch_size is {self.batch_size}')

                for batch in batches:
                    if len(batch) != self.batch_size:
                        self.batch_size = len(batch[0])
                   
                    if not self.record_flag:
                        yield batchify(self.args, batch, self.vocabs, self.unk_rate, self.flag)
                    else:
                        yield batchify(self.args, batch, self.vocabs, self.unk_rate, self.flag), batch
