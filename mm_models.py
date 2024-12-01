import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from transformers import ElectraConfig, ElectraModel
import os
import torch.nn.functional as F

""" AMR """

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pickle
import math
nltk.download('stopwords')

from encoder import RelationEncoder, TokenEncoder
from transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask
from graph_transformer import GraphTransformer
from utils import *

from functools import partial
from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from models_mae import MaskedAutoencoderViT
from models_vit import VisionTransformer

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from joblib import Parallel, delayed

""" AMR """

class BaseModel(nn.Module):
    def __init__(self, save_dir):
        super(BaseModel, self).__init__()
        self.save_dir = save_dir

    def save(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(self.save_dir, filename + '.pt'))

    def load(self, filepath):
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict, strict=False)


class MMModel(BaseModel):
    def __init__(self, imageEncoder, textEncoder, save_dir):
        super(MMModel, self).__init__(save_dir=save_dir)
        self.imageEncoder = imageEncoder
        self.textEncoder = textEncoder

    def forward(self, x, train):
        raise NotImplemented


class TextOnlyModel(BaseModel):
    def __init__(self, save_dir, dim_text_repr=768, num_class=2):
        super(TextOnlyModel, self).__init__(save_dir)

        self.dropout = nn.Dropout()
        config = ElectraConfig()
        self.textEncoder = ElectraModel(config).from_pretrained('google/electra-base-discriminator')
        self.linear = nn.Linear(dim_text_repr, num_class)

    def forward(self, x, train):
        text, _, _ = x

        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        e_i = self.dropout(hidden_states[0][:,0,:])  # N, dim_text_repr

        return self.linear(e_i)

class ImageOnlyModel(BaseModel):
    def __init__(self, save_dir, dim_visual_repr=1000, num_class=2):
        super(ImageOnlyModel, self).__init__(save_dir=save_dir)

        self.imageEncoder = torch.hub.load(
            'pytorch/vision:v0.8.0', 'densenet201', pretrained=True)
        self.flatten_vis = nn.Flatten()
        self.linear = nn.Linear(dim_visual_repr, num_class)
        self.dropout = nn.Dropout()

    def forward(self, x, train):
        _, image, _ = x

        f_i = self.dropout(self.flatten_vis(self.imageEncoder(image)))

        return self.linear(f_i)

class MAE(MaskedAutoencoderViT):
    def __init__(self, save_dir, num_class=2):
        super(MAE, self).__init__(patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.save_dir = save_dir
        C = 65
        DIM = int(1280*C)
        self.gelu = nn.GELU()

        self.linear = nn.Linear(DIM, num_class)
        # self.dropout = nn.Dropout()

    def save(self, filename):
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(self.save_dir, filename + '.pt'))

    def load(self, filepath):
        state_dict = torch.load(filepath)
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x, mask_ratio=0.75):
        _, imgs, _ = x
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)

        f_i = latent.reshape(latent.shape[0], -1) # maybe use dropout

        return loss, self.linear(f_i), f_i

class VIT(VisionTransformer):
    def __init__(self):
        super(VIT, self).__init__(patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=2, drop_path_rate=0.1, global_pool=True)

    def forward(self, x):
        _, imgs, _ = x
        return self.forward_features(imgs)

class DenseNetBertMMModel(MMModel):

    def apply_self_attention(self, input_tensor):
        # Calculate the dot product similarity scores
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))

        # Normalize the scores with softmax
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of input features using the attention weights
        attn_output = torch.matmul(attn_weights, input_tensor)

        return attn_output

    def __init__(self, save_dir, vocabs,
                concept_char_dim, concept_dim,
                cnn_filters, char2concept_dim,
                rel_dim, rnn_hidden_size, rnn_num_layers,
                embed_dim, ff_embed_dim, num_heads, dropout_amr,
                snt_layer, graph_layers,
                pretrained_file, loss_fn, device, batch_size, n_answers, 
                dim_visual_repr=1280*65, dim_text_repr=768, dim_amr_repr=256, dim_proj=100, num_class=2, **opt):
        self.save_dir = save_dir
        self.vocabs = vocabs
        self.opt = opt

        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr
       
        if opt['mae']:
            #imageDenoiser = MAE(save_dir=self.save_dir)
            #image_model_to_load = "./output/image_only_mae/best.pth"
            imageEncoder = MAE(num_class=num_class, save_dir=self.save_dir)
            # ckpt = torch.load(opt['image_model_to_load'], map_location='cpu')
            # imageEncoder.load_state_dict(ckpt['model'], strict=False)
        elif opt['vit']:
            image_model_to_load = "./checkpoints/mae_pretrain_vit_huge.pth"
            imageEncoder = VIT()
            checkpoint = torch.load(image_model_to_load, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % image_model_to_load)
            checkpoint_model = checkpoint['model']
            
            state_dict = imageEncoder.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(imageEncoder, checkpoint_model)

            # load pre-trained model
            msg = imageEncoder.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            # manually initialize fc layer
            trunc_normal_(imageEncoder.head.weight, std=2e-5)
        else:
            imageEncoder = torch.hub.load(
                'pytorch/vision:v0.8.0', 'densenet201', pretrained=True)
       
        config = ElectraConfig()
        textEncoder = ElectraModel(config).from_pretrained('google/electra-base-discriminator')

        super(DenseNetBertMMModel, self).__init__(imageEncoder, textEncoder, save_dir)
        self.dropout = Dropout()

        # Flatten image features to 1D array
        self.flatten_vis = torch.nn.Flatten()

        self.gelu = nn.GELU()

        # Linear layers used to project embeddings to fixed dimension (eqn. 3)
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)
        self.proj_amr = nn.Linear(dim_amr_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)
        self.proj_amr_bn = nn.BatchNorm1d(dim_proj)

        # Linear layers to produce attention masks (eqn. 4)
        self.layer_attn_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(dim_text_repr, dim_proj)
        self.layer_attn_amr = nn.Linear(dim_amr_repr, dim_proj)

        # An extra fully-connected layer for classification
        # The authors wrote "we add self-attention in the fully-connected networks"
        # Here it is assumed that they mean 'we added a fully-connected layer as self-attention'.
        # For 3 modalities (6C2)
        p=6
        self.fc_as_self_attn = nn.Linear(p*dim_proj, p*dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(p*dim_proj)

        # Classification layer
        self.cls_layer = nn.Linear(p*dim_proj, num_class)

        """ AMR """
        cnn_filters = list(zip(cnn_filters[:-1:2], cnn_filters[1::2]))

        self.vocabs = vocabs
        self.embed_scale = math.sqrt(embed_dim)
        self.concept_encoder = TokenEncoder(vocabs['concept'], vocabs['concept_char'],
                                            concept_char_dim, concept_dim, embed_dim,
                                            cnn_filters, char2concept_dim, dropout_amr, pretrained_file)
        self.relation_encoder = RelationEncoder(vocabs['relation'], rel_dim, embed_dim, rnn_hidden_size, rnn_num_layers,
                                                dropout_amr)
        self.graph_encoder = GraphTransformer(graph_layers, embed_dim, ff_embed_dim, num_heads, dropout_amr)
        self.transformer = Transformer(snt_layer, embed_dim, ff_embed_dim, num_heads, dropout_amr, with_external=True)
        self.c_transformer = Transformer(snt_layer, embed_dim, ff_embed_dim, num_heads, dropout_amr, with_external=True)

        self.pretrained_file = pretrained_file
        self.embed_dim = embed_dim
        self.concept_dim = concept_dim
        self.embed_scale = math.sqrt(embed_dim)

        self.token_position = SinusoidalPositionalEmbedding(embed_dim, device)
        self.concept_depth = nn.Embedding(32, embed_dim)
        self.token_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.concept_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_mask = SelfAttentionMask(device)

        self.answer_len = n_answers
        self.device = device
        self.batch_size = batch_size
        self.cre_loss = loss_fn

        self.reset_parameters()

        """ AMR """

    def reset_parameters(self):
        nn.init.constant_(self.concept_depth.weight, 0.)


    def encode_cn_step(self, inp, i, train=True):
        
        cn_concept_input = inp['cn_concept'][i][:,0].unsqueeze(1)
        cn_concept_char_input = inp['cn_concept_char'][i][:,0].unsqueeze(1)
        cn_concept_depth_input = inp['cn_concept_depth'][i][:,0].unsqueeze(1)
        cn_relation_bank_input = inp['cn_relation_bank'][i]
        cn_relation_length_input = inp['cn_relation_length'][i]
        cn_relation_input = inp['cn_relation'][i][:,:,0].unsqueeze(2)

        concept_repr = self.embed_scale * self.concept_encoder(cn_concept_input,
                                                               cn_concept_char_input) + self.concept_depth(
            cn_concept_depth_input)
        concept_repr = self.concept_embed_layer_norm(concept_repr)
        concept_mask = torch.eq(cn_concept_input, self.vocabs['concept'].padding_idx)
        relation = self.relation_encoder(cn_relation_bank_input, cn_relation_length_input)
        
        if str(train)=='True':
            relation = relation.index_select(0, cn_relation_input.reshape(-1)).view(*cn_relation_input.size(), -1)

        else:
            relation[0, :] = 0. # cn_relation_length x dim
            relation = relation[cn_relation_input]  # i x j x bsz x num x dim

            sum_relation = relation.sum(dim=3)  # i x j x bsz x dim
            num_valid_paths = cn_relation_input.ne(0).sum(dim=3).clamp_(min=1)  # i x j x bsz

            divisor = (num_valid_paths).unsqueeze(-1).type_as(sum_relation) # i x j x bsz x 1
            relation = sum_relation / divisor # i x j x bsz dim
        concept_repr = self.graph_encoder(concept_repr, relation, self_padding_mask=concept_mask)

        return concept_repr

    def prepare_graph_state(self, graph_state, ans_len, concept_dim):
        tot_initial = torch.tensor(1).to(self.device)

        j = 0
        while j < (1*ans_len)-1:
            initial = graph_state[0][j].view(1, -1).to(self.device)
            for i in graph_state[1:]:  # i = [5 x 512] x 7
                com_tensor = i[j + 1].view(1, -1).to(self.device)

                initial = torch.cat([initial, com_tensor], dim=0)
            if j == 0:
                tot_initial = initial.view(1, -1, concept_dim)
            j += 1
            initial = initial.view(1, -1, concept_dim)
            tot_initial = torch.cat([tot_initial, initial], dim=0)
        return tot_initial

    def forward(self, x, train):
        text, image, amr = x

        """ AMR """

        tot_concept_reprs = []
        for i in range(self.batch_size):
            ## AMR-GTOS
            concept_repr = self.encode_cn_step(amr, i, train=train)  # concept_seq_len x 1 x concept_embed_size # Decide true or false based on argument
            concept_repr = self.transformer(concept_repr, kv=None)  # res = concept_seq_len x bsz x concept_embed_size

            if concept_repr.size()[1] == 1:
                concept_repr = concept_repr.squeeze().unsqueeze(0).mean(1).unsqueeze(1)

            else:
                concept_repr = self.prepare_graph_state(concept_repr, concept_repr.size()[1], self.embed_dim).mean(
                    1).unsqueeze(1)  # re = bsz x 1 x concept_embed_size

            #concept_repr = concept_repr.repeat(1, answer_len, 1)  # re = 1 x 5 x concept_embed_size

            tot_concept_repr = self.c_transformer(concept_repr, kv=None)
            tot_concept_repr = tot_concept_repr.squeeze(1)
            tot_concept_reprs.append(tot_concept_repr)

        tot_concept_reprs = torch.squeeze(torch.stack(tot_concept_reprs), 1)

        """ AMR """

        if self.opt['mae']:
            f_i = self.imageEncoder(x)[2]
        else:
            f_i = self.dropout(self.flatten_vis(self.imageEncoder(image)))

        # Getting sentence representation (eqn. 2)
        hidden_states = self.textEncoder(**text)  # N, T, dim_text_repr
        # The authors used embedding associated with [CLS] to represent the whole sentence
        e_i = self.dropout(hidden_states[0][:,0,:])  # N, dim_text_repr

        # Applying self-attention to e_i and f_i and a_i
        e_i_self_attn = self.apply_self_attention(e_i)  # N, dim_proj
        f_i_self_attn = self.apply_self_attention(f_i)  # N, dim_proj
        a_i_self_attn = tot_concept_reprs  # N, dim_proj

        # Getting linear projections (eqn. 3)
        f_i_tilde = self.gelu(self.proj_visual_bn(
            self.proj_visual(f_i_self_attn)))  # N, dim_proj
        e_i_tilde = self.gelu(self.proj_text_bn(
            self.proj_text(e_i_self_attn)))  # N, dim_proj
        a_i_tilde = self.gelu(self.proj_amr_bn(
            self.proj_amr(a_i_self_attn)))  # N, dim_proj

        # cross attention masking for all modals
        alpha_e_i = torch.sigmoid(self.layer_attn_text(e_i_self_attn))  # N, dim_proj
        alpha_f_i = torch.sigmoid(self.layer_attn_visual(f_i_self_attn))  # N, dim_proj
        alpha_a_i = torch.sigmoid(self.layer_attn_amr(a_i_self_attn))  # N, dim_proj 
    
        #print("values",f_i_tilde.shape,alpha_v_i.shape)
        # The authors concatenated masked embeddings to get a joint representation
        masked_fe_i = torch.multiply(alpha_e_i, f_i_tilde)
        masked_ef_i = torch.multiply(alpha_f_i, e_i_tilde)

        masked_ae_i = torch.multiply(alpha_e_i, a_i_tilde)
        masked_ea_i = torch.multiply(alpha_a_i, e_i_tilde)

        masked_af_i = torch.multiply(alpha_f_i, a_i_tilde)
        masked_fa_i = torch.multiply(alpha_a_i, f_i_tilde)

        # Cross-embedding loss
        target = torch.ones(self.batch_size).to(self.device)
        cre_loss = (
            self.cre_loss(alpha_e_i, f_i_tilde, target) + self.cre_loss(alpha_f_i, e_i_tilde, target)
            + self.cre_loss(alpha_e_i, a_i_tilde, target) + self.cre_loss(alpha_a_i, e_i_tilde, target)
            + self.cre_loss(alpha_f_i, a_i_tilde, target) + self.cre_loss(alpha_a_i, f_i_tilde, target)
        )
        cre_loss /= p
        joint_repr = torch.cat((masked_fe_i, masked_ef_i, masked_ae_i, masked_ea_i, masked_af_i, masked_fa_i),
                               dim=1)  # N, 2*dim_proj

        # for attention weights
        # tensor_cpu = joint_repr.to('cpu')
        # torch.save(tensor_cpu, 'assets/Attention_Weights.pkl')

        # Get class label prediction logits with final fully-connected layers
        return cre_loss, self.cls_layer(self.dropout(self.gelu(self.self_attn_bn(self.fc_as_self_attn(joint_repr)))))
