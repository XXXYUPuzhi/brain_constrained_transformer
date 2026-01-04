# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 13:12:56 2023

@author: lzq19
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timm.models.vision_transformer import DropPath, Mlp
from timm.models.layers import to_2tuple
from pytorch_model_summary import summary
import einops
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import pickle
import shelve
import os
CodePath = os.path.dirname(os.path.abspath(__file__))
os.chdir(CodePath)
import seaborn as sns
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
import sys
sys.path.append('../..')

from utils import *

# In[]
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4,causal=False, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        # number of heads should be dvidable by dim, the input dimension
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.causal = causal
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients
        
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map.clone()

    def save_unnormed_attn_map(self,attn_map):
        self.unnormed_attn_map = attn_map.clone()
    
    def get_attention_map(self):
        return self.attention_map
    
    def get_unnormed_attn_map(self):
        return self.unnormed_attn_map
    
    def forward(self, x, q_in=None,k_in=None,v_in=None, register_hook=False,attn_in=None,attn_v_in=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   
        if q_in is not None:
            q=q_in
        if k_in is not None:
            k=k_in
        if v_in is not None:
            v=v_in
        # with torch.cuda.amp.autocast(enabled=False): #?
        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        if self.causal==True:
            causal_mask = torch.triu(torch.ones((N, N), device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(causal_mask, -1e9)
        self.save_unnormed_attn_map(attn)
        attn = attn.softmax(dim=-1)
        if attn_in is not None:
            attn = attn_in
        # attn = self.attn_drop(attn)
        self.save_attention_map(attn)
        if register_hook:
            attn.register_hook(self.save_attn_gradients)
        
        attn_v = attn @ v
        if attn_v_in is not None:
            attn_v = attn_v_in
        x = attn_v.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, q, k, v


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.,causal=False, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layerID=None, analizemode=False, save_path=None, skipnorm=[],
                 linearized_norm=[],device='cuda'):
        super().__init__()
        self.device=device
        self.savepath=save_path
        self.layerID = layerID
        self.norm1 = norm_layer(dim)
        self.analizemode = analizemode
        # dim = the input dimension
        self.attn = Attention(
            dim, num_heads=num_heads, causal=causal, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skipnorm = skipnorm
        self.linearized_norm = linearized_norm
        if len(linearized_norm)>0:
            self.LNparas = nn.ParameterDict()
            for (i, paras) in linearized_norm:
                self.LNparas[str(i)] = nn.ParameterList()
                for p in paras:
                    # import pdb;pdb.set_trace()
                    self.LNparas[str(i)].append(nn.Parameter(torch.Tensor(p).to(self.device)))


    def forward(self, x_and_ID, q_in=None, k_in=None, v_in=None,attn_in=None,attn_v_in=None,register_hook=None,pre_norm=True):
        if register_hook is None:
            if self.analizemode is True:
                # x = x_and_ID[0]
                x = x_and_ID
                # storageID = x_and_ID[1]
                register_hook=True
            elif self.analizemode is False:
                x = x_and_ID
                register_hook=False
        else:
            x = x_and_ID
    # def forward(self, x, q_in=None, k_in=None, v_in=None):
        if pre_norm:
            if 0 in self.skipnorm:
                x1 = x
            elif 0 in [x[0] for x in self.linearized_norm]:
                x1 = (x-self.LNparas['0'][0])/self.LNparas['0'][1]*self.LNparas['0'][2]+self.LNparas['0'][3]
            else:
                x1 = self.norm1(x)
            y, attn, q, k, v = self.attn(x1,q_in=q_in, k_in=k_in, v_in=v_in, 
                                register_hook=register_hook, attn_in=attn_in,attn_v_in=attn_v_in)
            x = x + self.drop_path(y)
            if 1 in self.skipnorm:
                x2 = x
            elif 1 in [x[0] for x in self.linearized_norm]:
                x1 = (x-self.LNparas['1'][0])/self.LNparas['1'][1]*self.LNparas['1'][2]+self.LNparas['1'][3]
            else:
                x2 = self.norm2(x)
            x = x + self.drop_path(self.mlp(x2))
            
            # if self.layerID is not None:
            #     if self.analizemode is True:
            #         with open(self.savepath+self.layerID+'-'+storageID+'.pkl','wb') as f:
            #             pickle.dump((attn,q,k,v,),f)
            if self.analizemode is True:
                return x
            elif self.analizemode is False:
                return x
        elif pre_norm is False:
            y, attn, q, k, v = self.attn(x,q_in=q_in, k_in=k_in, v_in=v_in, 
                                register_hook=register_hook, attn_in=attn_in,attn_v_in=attn_v_in)
            x = x + self.drop_path(y)
            if 0 not in self.skipnorm:
                x1 = self.norm1(x)
            x = x1 + self.drop_path(self.mlp(x1))
            if 1 not in self.skipnorm:
                x = self.norm2(x)

            # if self.layerID is not None:
            #     if self.analizemode is True:
            #         with open(self.savepath+self.layerID+'-'+storageID+'.pkl','wb') as f:
            #             pickle.dump((attn,q,k,v,),f)
            if self.analizemode is True:
                return x
            elif self.analizemode is False:
                return x
# In[]    
class visual2action_3(nn.Module):
    def __init__(self,configs):
        super(visual2action_3,self).__init__()
        self.skip_norm=[]
        self.pre_norm = True
        for k,v in configs.items():
            setattr(self,k,v)      
            
        # self.emb = nn.Conv2d(17*configs['t'], configs['patch_emb_dim'], kernel_size=configs['ker_size'], stride=configs['step_size'])
        if 'EmbLN' in configs and configs['EmbLN']==False:
            self.to_patch_embedding = nn.Sequential(
                Rearrange("b (h p1) (w p2) c-> b (h w) (p1 p2 c)", p1 = configs['ker_size'][0], p2 = configs['ker_size'][1]),
                nn.Linear(configs['ker_size'][0]*configs['ker_size'][1]*17*configs['t'], configs['patch_emb_dim']),
            )
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange("b (h p1) (w p2) c-> b (h w) (p1 p2 c)", p1 = configs['ker_size'][0], p2 = configs['ker_size'][1]),
                nn.LayerNorm(configs['ker_size'][0]*configs['ker_size'][1]*17*configs['t']),
                nn.Linear(configs['ker_size'][0]*configs['ker_size'][1]*17*configs['t'], configs['patch_emb_dim']),
                nn.LayerNorm( configs['patch_emb_dim']),
            )
        
        if 'num_heads' in configs:
            n_head = configs['num_heads']
        else:
            n_head=4
        self.encode = nn.ModuleList([Block(configs['patch_emb_dim'], n_head, 4, False) for _ in range(configs['n_layers'])])
        
        for _, (nth_layer, nth_norm) in enumerate(self.skip_norm):
            self.encode[nth_layer] = Block(configs['patch_emb_dim'], n_head, mlp_ratio=4.,causal=False, skipnorm=nth_norm)

        self.input_pos_emb = posemb_sincos_2d(h=configs['H']/configs['ker_size'][0], 
                                              w=configs['W']/configs['ker_size'][1], 
                                              dim=configs['patch_emb_dim']).to(configs['device'])
        self.act_token = nn.Parameter(torch.zeros(1, 1, configs['patch_emb_dim']))
        # self.action_lossfun = nn.CrossEntropyLoss(reduction='mean')
        self.action_lossfun = FocalLoss(gamma=2.0, alpha=0.25)
        self.predhead = nn.Linear(configs['patch_emb_dim'],4)
        
    def forward(self, sample,register_hook=False):
            
        x, tar = sample
        
        B,H,W,D = x.shape
        if self.input_scaling==True:
            x = (x-0.5)*2
        

        # x = einops.rearrange(x, 'b h w c ->b c h w',b=B,h=H,w=W)
        # x = self.emb(x)
        # import ipdb;ipdb.set_trace()
        x = self.to_patch_embedding(x)
        
        # x = einops.rearrange(x, 'b c h w ->b (h w) c')
        x = x + self.input_pos_emb.expand(B, -1, -1)
        act_token = self.act_token.expand(B, -1, -1)
        x = torch.cat((act_token, x), dim=1)
        for i,module in enumerate(self.encode):
            x = self.encode[i](x, register_hook=register_hook,pre_norm=self.pre_norm)
        # z = self.encode(x)
        z=x
        y = self.predhead(z[:,0,:])
        y = F.softmax(y,dim=-1) # for focal loss
        loss = self.action_lossfun(y,tar)

        return y, loss    
    

class visual2action_3_linearLN(nn.Module):
    def __init__(self,configs):
        super(visual2action_3_linearLN,self).__init__()
        self.skipnorm = []
        self.linearized_norm = []
        self.pre_norm = True
        for k,v in configs.items():
            setattr(self,k,v)      
            
        # self.emb = nn.Conv2d(17*configs['t'], configs['patch_emb_dim'], kernel_size=configs['ker_size'], stride=configs['step_size'])
        if 'EmbLN' in configs and configs['EmbLN']==False:
            self.to_patch_embedding = nn.Sequential(
                Rearrange("b (h p1) (w p2) c-> b (h w) (p1 p2 c)", p1 = configs['ker_size'][0], p2 = configs['ker_size'][1]),
                nn.Linear(configs['ker_size'][0]*configs['ker_size'][1]*17*configs['t'], configs['patch_emb_dim']),
            )
        else:
            self.to_patch_embedding = nn.Sequential(
                Rearrange("b (h p1) (w p2) c-> b (h w) (p1 p2 c)", p1 = configs['ker_size'][0], p2 = configs['ker_size'][1]),
                nn.LayerNorm(configs['ker_size'][0]*configs['ker_size'][1]*17*configs['t']),
                nn.Linear(configs['ker_size'][0]*configs['ker_size'][1]*17*configs['t'], configs['patch_emb_dim']),
                nn.LayerNorm( configs['patch_emb_dim']),
            )
        
        if 'num_heads' in configs:
            n_head = configs['num_heads']
        else:
            n_head=4
        self.encode = nn.ModuleList([Block(configs['patch_emb_dim'], n_head, 4, False) for _ in range(configs['n_layers'])])
        if len(self.linearized_norm)>0:
            # import pdb;pdb.set_trace()
            for _, (norm_id, norm_paras) in enumerate(configs['linearized_norm']):
                self.encode[norm_id[0]] = Block(configs['patch_emb_dim'], n_head, mlp_ratio=4.,causal=False, linearized_norm=[(norm_id[1], norm_paras),], device=self.device)
        # self.encode = nn.Sequential(*[Block(configs['patch_emb_dim'], n_head, 4, False) for _ in range(configs['n_layers'])])
        # self.emb = nn.Linear(configs['input_dim'],configs['emb_dim'])
        # self.encode = nn.Sequential(*[Block(configs['emb_dim'], 4, 4, False) for _ in range(configs['n_layers'])])
        # self.input_pos_emb = nn.Parameter(torch.randn(1, 31*28+1, configs['emb_dim'])) 
        self.input_pos_emb = posemb_sincos_2d(h=configs['H']/configs['ker_size'][0], 
                                              w=configs['W']/configs['ker_size'][1], 
                                              dim=configs['patch_emb_dim']).to(configs['device'])
        self.act_token = nn.Parameter(torch.zeros(1, 1, configs['patch_emb_dim']))
        # self.action_lossfun = nn.CrossEntropyLoss(reduction='mean')
        self.action_lossfun = FocalLoss(gamma=2.0, alpha=0.25)
        self.predhead = nn.Linear(configs['patch_emb_dim'],4)
        
    def forward(self, sample,register_hook=False):
            
        x, tar = sample
        
        B,H,W,D = x.shape
        if self.input_scaling==True:
            x = (x-0.5)*2
        

        # x = einops.rearrange(x, 'b h w c ->b c h w',b=B,h=H,w=W)
        # x = self.emb(x)
        # import ipdb;ipdb.set_trace()
        x = self.to_patch_embedding(x)
        
        # x = einops.rearrange(x, 'b c h w ->b (h w) c')
        x = x + self.input_pos_emb.expand(B, -1, -1)
        act_token = self.act_token.expand(B, -1, -1)
        x = torch.cat((act_token, x), dim=1)
        for i,module in enumerate(self.encode):
            x = self.encode[i](x, register_hook=register_hook,pre_norm=self.pre_norm)
        # z = self.encode(x)
        z=x
        y = self.predhead(z[:,0,:])
        y = F.softmax(y,dim=-1) # for focal loss
        loss = self.action_lossfun(y,tar)

        return y, loss    


# In[]
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        FL = []
        for i in range(inputs.shape[1]):
            # import ipdb;ipdb.set_trace()
            BCE_loss = F.binary_cross_entropy_with_logits(inputs[:,i], (targets==i).float())
            pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
            # import ipdb;ipdb.set_trace()
            FL.append(self.alpha * (1-pt)**self.gamma * BCE_loss)
            F_loss = torch.stack(FL)
        
        if self.reduction == 'sum':
            return torch.sum(F_loss)
        elif self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss
        