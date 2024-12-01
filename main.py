from os import path as osp
import os
import logging
from PIL import Image
from torch.serialization import save
from args import get_args
from trainer import Trainer
from mm_models import DenseNetBertMMModel, ImageOnlyModel, TextOnlyModel, VIT, MAE
import os
import numpy as np
import torch
from torch.nn.modules import activation
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import math
from optimization import AdamW, WarmupCosineSchedule , WarmupLinearSchedule
import time
from utils import move_to_cuda

from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

from data import Vocab, LexicalMap, DataLoader, STR, END, CLS, SEL, TL, rCLS

import yaml
import nltk

def _prepare_data(args):

        vocabs = dict()
        vocabs['concept'] = Vocab(args.concept_vocab, 5, [CLS])
        vocabs['token'] = Vocab(args.token_vocab, 5, [STR, END])
        vocabs['token_char'] = Vocab(args.token_char_vocab, 100, [STR, END])
        vocabs['concept_char'] = Vocab(args.concept_char_vocab, 100, [STR, END])
        vocabs['relation'] = Vocab(args.relation_vocab, 5, [CLS, rCLS, SEL, TL])
        lexical_mapping = LexicalMap()

        return vocabs, lexical_mapping

def get_params(path='./params.yaml'):
    with open(path) as file:
        params = yaml.safe_load(file)
    return params

if __name__ == '__main__':
    opt = get_args()
    params = get_params()
    opt.__dict__.update(params)

    model_to_load = opt.model_to_load
    image_model_to_load = opt.image_model_to_load
    text_model_to_load = opt.text_model_to_load

    device = opt.device

    num_workers = opt.num_workers

    EVAL = opt.eval
    USE_TENSORBOARD = opt.use_tensorboard
    SAVE_DIR = opt.save_dir
    MODEL_NAME = opt.model_name if opt.model_name else str(int(time.time()))

    MODE = opt.mode
    TASK = opt.task
    MAX_ITER = opt.max_iter
    OUTPUT_SIZE = None 
    if TASK == 'task1':
        OUTPUT_SIZE = 2
    elif TASK == 'task2':
        OUTPUT_SIZE = 8
    elif TASK == 'task2_merged':
        OUTPUT_SIZE = 6
    elif TASK == 'task3':
        OUTPUT_SIZE = 3
    else:
        raise NotImplemented

    WITH_SSE = opt.with_sse
    pv = opt.pv # How many times more likely do we transit to the same class
    pt = opt.pt 
    pv0 = opt.pv0  # Probability of not doing a transition
    pt0 = opt.pt0

    # General hyper parameters
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size

    # Create folder for saving
    save_dir = osp.join(SAVE_DIR, MODEL_NAME)
    if not osp.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)


    # set logger
    logging.basicConfig(filename=osp.join(save_dir, 'output_{}.log'.format(int(time.time()))), level=logging.INFO)

    """
    Use vocab for dataloader too
    """

    vocabs, lexical_mapping = _prepare_data(opt)

    train_loader, dev_loader = None, None
    train_load,  dev_load, test_load = [], [], []

    if not EVAL:
        train_loader = DataLoader(opt, vocabs, lexical_mapping, opt.train_data_amr, batch_size, flag=True, 
                phase='train', cat='all', task=TASK, shuffle=False, consistent_only=False)
        for data in tqdm(train_loader, total=len(train_loader)):
               train_load.append(move_to_cuda(data, device=device))
    
    dev_loader = DataLoader(opt, vocabs, lexical_mapping, opt.dev_data_amr, batch_size, flag=False, 
                phase='dev', cat='all', task=TASK, shuffle=False, consistent_only=False)
    for data in tqdm(dev_loader, total=len(dev_loader)):
           dev_load.append(move_to_cuda(data, device=device))

    test_loader = DataLoader(opt, vocabs, lexical_mapping, opt.test_data_amr, batch_size, flag=False, 
                phase='test', cat='all', task=TASK, shuffle=False, consistent_only=False)
    for data in tqdm(test_loader, total=len(test_loader)):
           test_load.append(move_to_cuda(data, device=device))

    # Converting opt namespace to vars
    opt_vars = vars(opt)
    if opt.mae:
        dim_visual_repr = 1280*65
    else:
        dim_visual_repr = 1000

    loss_fn = nn.CrossEntropyLoss()
    cre_loss = nn.CosineEmbeddingLoss()
    if MODE == 'text_only':
        model = TextOnlyModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    elif MODE == 'image_only':
        if opt.mae:
            model = MAE(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
        elif opt.vit:
             model = VIT().to(device)
        else:
            model = ImageOnlyModel(num_class=OUTPUT_SIZE, save_dir=save_dir).to(device)
    elif MODE == 'both':
        model = DenseNetBertMMModel(loss_fn=cre_loss, num_class=OUTPUT_SIZE, vocabs=vocabs, dim_visual_repr= dim_visual_repr, **opt_vars)#.to(device)
    else:
        raise NotImplemented

    t_total = math.ceil((len(train_loader))/(40)) * 50 if train_loader != None else 0

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=1e-4)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, cooldown=0, verbose=True)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=t_total*0.1, t_total=t_total)
    #scheduler = OneCycleLR(optimizer, total_steps=t_total, max_lr = learning_rate,epochs = 50)

    trainer = Trainer(train_load, dev_load, test_load,
                      model, loss_fn, optimizer, scheduler, eval=EVAL, device=device, tensorboard=USE_TENSORBOARD, mode=MODE, opt=opt, save_dir=save_dir)

    if model_to_load:
        ckpt = torch.load(model_to_load, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        logging.info("\n***********************")
        logging.info("Model Loaded!")
        logging.info("***********************\n")
    if text_model_to_load:
        model.load(text_model_to_load)
    if image_model_to_load: # only for image_only task can be done in mm_models
        ckpt = torch.load(image_model_to_load, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)

    if not EVAL:
        logging.info("\n================Training Summary=================")
        logging.info("Training Summary: ")
        logging.info("Learning rate {}".format(learning_rate))
        logging.info("Batch size {}".format(batch_size))
        logging.info(trainer.model)
        logging.info("\n=================================================")

        trainer.train(MAX_ITER)

    else:
        logging.info("\n================Evaluating Model=================")
        logging.info(trainer.model)

        # trainer.validate()
        trainer.predict()
