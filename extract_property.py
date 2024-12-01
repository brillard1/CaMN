import itertools
import json
from smatch import AMR
from AMRGraph import AMRGraph
from AMRGraph import  _is_abs_form
from AMR_CN_Graph import AMRCNGraph
from collections import Counter
from multiprocessing import Pool
from pathFinder import WikiTag
from tqdm.auto import tqdm

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

class AMRIO:
    def __init__(self):
        pass

    @staticmethod
    def read(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()

                if line.startswith('# ::id '):
                    amr_id = line[len('# ::id '):]
                # elif line.startswith('# ::snt '):
                    # sentence = line[len('# ::snt '):]
                elif line.startswith('# ::tokens '):
                    tokens = json.loads(line[len('# ::tokens '):])
                    tokens = [ to if _is_abs_form(to) else to.lower() for to in tokens]
                elif line.startswith('# ::lemmas '):
                    lemmas = json.loads(line[len('# ::lemmas '):])
                    lemmas = [le if _is_abs_form(le) else le.lower() for le in lemmas]
                elif line.startswith('# ::pos_tags '):
                    pos_tags = json.loads(line[len('# ::pos_tags '):])
                elif line.startswith('# ::ner_tags '):
                    ner_tags = json.loads(line[len('# ::ner_tags '):])
                elif line.startswith('# ::abstract_map '):
                    abstract_map = json.loads(line[len('# ::abstract_map '):])
                elif line.startswith('# ::option '):
                    option = line[len('# ::option '):]
                    option = ast.literal_eval(option)
                elif line.startswith('# ::answer_key '):
                    answer_key = line[len('# ::answer_key '):]
                # elif line.startswith('# ::answer '):
                #     answer = line[len('# ::answer '):]
                elif line.startswith('# ::snt '):
                    sentence = line[len('# ::snt '):]
                    graph_line = AMR.get_amr_line(f)
                    raw_amr = AMR.parse_AMR_line(graph_line)
                    if raw_amr != None:
                        myamr = AMRGraph(raw_amr)
                    else:
                        continue

                    yield tokens, lemmas, abstract_map, myamr, answer_key, option, amr_id, sentence, raw_amr

def read_file(filename):
    # read preprocessed amr file
    token, lemma, abstract, amrs, answer_keys, options, amr_ids, sentences, raw_amrs = [], [], [], [], [], [], [], [], []

    for _tok, _lem, _abstract, _myamr, _answer_key, _option, _amr_id, _sentence, _raw_amr in AMRIO.read(filename):
        token.append(_tok)
        lemma.append(_lem)
        abstract.append(_abstract)
        amrs.append(_myamr)
        answer_keys.append(_answer_key)
        options.append(_option)
        amr_ids.append(_amr_id)
        sentences.append(_sentence)
        raw_amrs.append(_raw_amr)

    print ('read from %s, %d amrs'%(filename, len(token)))
    return amrs, token, lemma, abstract, answer_keys, options, amr_ids, sentences, raw_amrs

def make_vocab(batch_seq, char_level=False):
    cnt = Counter()
    for seq in batch_seq:
        cnt.update(seq)
    if not char_level:
        return cnt
    char_cnt = Counter()
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt


def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n'%(x,y))



import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--amr_files', type=str, nargs='+')
    parser.add_argument('--nprocessors', type=int, default=4)
    parser.add_argument('--concept_seed', type=str)
    parser.add_argument('--tagme_tokenID', type=int)
    parser.add_argument('--elf', type=float)
    parser.add_argument('--clf', type=float)

    return parser.parse_args()

import ast
if __name__ == "__main__":
    # collect concepts and relations
    args = parse_config()
    cn = dict()

    lexical_map = LexicalMap()

    wikiNetP = WikiTag([], 1, 0, 0)
    entity_dict, id_dict, relation_dict = wikiNetP.read_entities_relations()
    wiki_concept = wikiNetP.read_concepts()

    def work(data):
        amr, lem, tok = data
        concept, depth, relation, ok, ARGs = amr.collect_concepts_and_relations()
        assert ok, "not connected"

        return concept, depth, relation, ARGs

    pool = Pool(args.nprocessors)

    for file in args.amr_files:
        cnt = 0
        with open(
                 file[0:file.index('.txt')] + '_cn_extended_final.json',
                 'w', encoding='utf-8', ) as json_result:

            json_result.write('[')
            my_data = []
            amrs, token, lemma, abstract, answer_keys, options, amr_ids, sentences, raw_amrs = read_file(file)
            res = pool.map(work, zip(amrs, lemma, token), len(amrs) // args.nprocessors)

            file_tag = file[file.index('.txt') - 9 : file.index('.txt')]
            if (file_tag[-2] == '_'):
                if (file_tag[:2] == 'in'):
                    file_tag = 'train_' + file_tag[-1] + '.txt'
                elif (file_tag[:2] == 'st'):
                    file_tag = 'test_' + file_tag[-1] + '.txt'
                elif (file_tag[:2] == 'ev'):
                    file_tag = 'dev_' + file_tag[-1] + '.txt'
            elif (file_tag[-2] != '_'):
                if (file_tag[:1] == 'n'):
                    file_tag = 'train_' + file_tag[-2:] + '.txt'
                elif (file_tag[:1] == 't'):
                    file_tag = 'test_' + file_tag[-2:] + '.txt'
                elif (file_tag[:1] == 'v'):
                    file_tag = 'dev_' + file_tag[-2:] + '.txt'

            with open('wikinet/vocab/'+file_tag, 'w') as f:
                f.write('')
            f.close()

            for gr, to, le, ab, answer_key, option, amr_id, sent, raw_amr in tqdm(zip(res, token, lemma, abstract,
                                                                                 answer_keys,
                                                                                 options, amr_ids, sentences, raw_amrs), total = len(sentences)):

                concept, depth, relation, ARGs = gr

                ## ConceptNet
                cnt += 1
                cn_concepts = []
                cn_depths = []
                cn_relations = []

                print(f'\nAMR: {cnt} ID: {amr_id}')
                entities = list(set(concept)-set(ARGs))
                if args.concept_seed == 'AMR_CN_PRUNE':
                    cn_seed = list(set([con for con in ARGs if con in cn]))

                    wikiNet = WikiTag(entities, hops = 5, entity_threshold = args.elf, path_threshold = args.clf, token_index=args.tagme_tokenID, file_tag = file_tag)
                    cn_seed, concept_graph = wikiNet.output_graph(entity_dict, id_dict, relation_dict, wiki_concept)
                    print(f'\nFinal concept graph: {concept_graph} \n')

                    _cn_concepts, _cn_depths, _cn_relations, _is_connected, g = [], [], [], [], []

                else:
                    cn_seed = []
                    _cn_concepts, _cn_depths, _cn_relations, _is_connected, g = [], [], [], [], []

                cn_concepts.append(_cn_concepts)
                cn_depths.append(_cn_depths)
                cn_relations.append(_cn_relations)

                cn_concepts2 = [cn_concepts[0] for i in range(5)]
                cn_depths2 = [cn_depths[0] for i in range(5)]
                cn_relations2 = [cn_relations[0] for i in range(5)]

                item = {
                    'concept': concept,
                    'depth': depth,
                    'relation': relation,
                    'cn_concept': cn_concepts2,
                    'cn_depth': cn_depths2,
                    'cn_relation': cn_relations2,
                    'token': to,
                    'lemma': le,
                    'abstract': ab,
                    'answer_key': answer_key,
                    'option': option,
                    'id': amr_id,
                    'sentences': sent}

                json.dump(item, json_result)
                json_result.write(',')

            json_result.write(']')
