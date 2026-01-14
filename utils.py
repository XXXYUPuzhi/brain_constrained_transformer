# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 13:15:14 2023

@author: lzq19
"""
# In[]
import pandas as pd
import numpy as np
import torch
import einops
import shelve
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.textpath import TextToPath
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
from scipy.interpolate import interp1d
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import OneHotEncoder
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

CodePath = os.path.dirname(os.path.abspath(__file__))
FP = FontProperties(fname=os.path.join(CodePath,'Font Awesome 6 Free-Solid-900.otf'))
iconpath=os.path.join(CodePath,'pacmanicons')

# In[]
def downsample(group,interval=6):
    n_samples = len(group)
    n_downsampled = n_samples //interval
    remainder = n_samples % interval
    if n_downsampled == 0 and remainder > 0:
        return group.iloc[-1:]
    downsampled = group.iloc[interval-1 : interval * n_downsampled][::interval]
    if remainder > 0:
        last_sample = group.iloc[-1:]
        downsampled = pd.concat([downsampled, last_sample])
    return downsampled

def map2wall(inp):
    return np.array([x in ['-','|'] for x in inp]).reshape(36,28)
def map2bean(inp):
    return np.array([x=='.' for x in inp]).reshape(36,28)
def map2energizer(inp):
    return np.array([x=='o' for x in inp]).reshape(36,28)    
def map2fruit(inp):
    return np.array([[1,2,3,4,5][['A','C','M','O','S'].index(x)] if x in ['A','C','M','O','S'] else 0 for x in inp]).reshape(36,28)

def df2ghost(pos,scared):
    # chase=1, scare=4, flash=5, eaten=3, scatter=2
    # -> chase & scatter=1, eaten=2, scare=3, flash=4
    # ghost = np.zeros((36,28,len(pos)))   
    ghost_=[]
    for i,(j1,j2) in enumerate(zip(pos,scared)):
        ghost = np.zeros((36,28))  
        if j1[0] in range(1,29):
            if j2 in [2,1]:
                ghost[j1[1]-1,j1[0]-1]=1 # ghost=1 if it appears normal
            else:
                ghost[j1[1]-1,j1[0]-1]=j2-1 # ghost = mode num-1 (range from 2 to 4) if it its not normal
        ghost_.append(np.eye(5)[ghost.astype(int)][:,:,1:])
    return np.concatenate(ghost_,axis=2)

def df2pacman(pos):
    pos = eval(pos)
    pacman = np.zeros((36,28)) 
    if pos[0] in range(1,29):
        pacman[pos[1]-1,pos[0]-1]=1
    return pacman
    
def convert_data(df):
    walls = np.array(df.Map.apply(map2wall).tolist())
    beans = np.array(df.Map.apply(map2bean).tolist())
    energizer = np.array(df.Map.apply(map2energizer).tolist())
    fruit = np.array(df.Map.apply(map2fruit).tolist())
    fruit_ = np.eye(6)[fruit][:,:,:,1:]
    pacman = np.array(df.pacmanPos.apply(df2pacman).tolist())
    
    g_pos = [(eval(x),eval(y)) for (x,y) in zip(df.ghost1Pos,df.ghost2Pos)]
    g_scared = [(x,y) for (x,y) in zip(df.ifscared1,df.ifscared2)]
    ghost=[]
    for i in range(len(g_pos)):
        ghost.append(df2ghost(g_pos[i],g_scared[i]))
    ghost = np.array(ghost)   
    r_feats = np.stack((beans,energizer,fruit),axis=1)
    r1 = ((r_feats[1:,:,:,:].sum(-1).sum(-1)-r_feats[:-1,:,:,:].sum(-1).sum(-1)))
    r = -r1*(r1<0)
    r = r[:,0]+r[:,1]*2+(r[:,2]+2)*(r[:,2]>0)
    n_eaten = (ghost[:,:,:,range(1,ghost.shape[-1],4)]).sum(-1).sum(-1).sum(-1)
    r2 = (n_eaten[:-1]-n_eaten[1:]>0)*1
    R = np.stack((r,r2),axis=1)
    reward = np.insert(R,0,0,axis=0)
    
    action = np.array([[1,2,3,4][['left','right','up','down'].index(x)] 
              if x in ['left','right','up','down'] else 0 for x in df.JoyStick])
    
    
    board_feature = np.concatenate((walls[...,np.newaxis],beans[...,np.newaxis],
                                    energizer[...,np.newaxis],fruit_,ghost,pacman[...,np.newaxis]),axis=3)
    # board_feature = einops.rearrange(board_feature,'c b h w -> b c h w')
    # 0 wall, 1 bean, 2 energizer, 3 fruit1, 4 fruit2, 5 fruit3, 6 fruit4, 7 fruit5, 
    # 8 g1 normal, 9 g1 eaten, 10 g1 scared, 11 g1 flash, 12 g2 normal, 13 g2 eaten, 14 g2 scared, 15 g2 flash,
    # 16 pacman
    return board_feature, reward, action

def get_batch(board_feature,reward,action,indices,device,seq_length=20,mode='voc'):

    observation = np.stack([board_feature[i:i+seq_length,:,:,:] for i in indices], axis=0)
    if mode=='label':
        observation = format_processor(observation,device).int()
    elif mode=='voc':
        observation = format_processor(label2voc(observation),device)
    
    rew = np.stack([reward[i:i+seq_length,:] for i in indices], axis=0)
    rew = format_processor(rew,device).int()
    act = np.stack([action[i:i+seq_length] for i in indices], axis=0)
    act = format_processor(act,device).int()
    return observation, rew, act

        
def emb2label(x):
    inp = x.detach().numpy()
    props_l = np.argmax(inp[...,:props_class],axis=-1)
    chara_l = np.argmax(inp[...,props_class:],axis=-1)
    label = np.stack((props_l,chara_l),axis=-1)
    return label

def label2voc(x,tensor=False,device='cpu'):
    if tensor==False:
        return (np.concatenate((voc_props[x[...,0]],voc_chara[x[...,1]]),axis=-1))
    else:
        return (torch.cat((torch.Tensor(voc_props).to(device)[x[...,0].long()],torch.Tensor(voc_chara).to(device)[x[...,1].long()]),axis=-1))

def emb2voc(x):
    label = emb2label(x)
    return label2voc(label)

def format_processor(data,device):
    return(torch.tensor(data).float().to(device))

def reshape4loss(batch,mode):
    if mode=='pred':
        return einops.rearrange(batch,'B T H W c -> (B T H W) c')
    elif mode=='tar':
        return einops.rearrange(batch,'B T H W -> (B T H W)')

    
class kl_weight():
    def __init__(self,k=1,b=1):
        self.k=k
        self.b=b
    def __call__(self,epoch):
        # kld annealing
        return 1/(1+np.exp(-self.k*epoch+self.b))

def interleave_alongaxis1(a,b):
    c = np.empty((a.shape[0],a.shape[1]*2), dtype=a.dtype)
    c[:,0::2] = a
    c[:,1::2] = b            
    return c
    
def sincos_pe(i,dim,seq_length=0):
    i_ = np.array(range(i,i+seq_length)).reshape(-1,1)/10000**(np.array(range(dim//2))/dim).reshape(1,-1)
    return interleave_alongaxis1(np.sin(i_),np.cos(i_))

def get_FR_batch(FR, indices, device, seq_length=20):
    FR = np.stack([FR[i:i+seq_length,:] for i in indices], axis=0)
    FR = format_processor(FR,device).float()
    return FR

def get_FR_batch_fromtensor(FR, indices, device, seq_length=20):
    FR = torch.stack([FR[i:i+seq_length,:] for i in indices], axis=0)
    FR = FR.to(device).float()
    return FR


def generate_test_batch(game_info,reward,action,FR,indices,device,seq_length,mode='alldata',order='default',tensor=False,return_index=False,batch_size=0):
    idx = indices[:-seq_length].copy()
    if order=='shuffle':
        np.random.shuffle(idx)

    rng = range(0,len(idx),batch_size)
    
    if return_index==False:
        for i in rng:
            ii = list(range(i,min(i+batch_size,len(idx))))
            i_range = idx[ii]
            if tensor==False:
                if mode=='alldata':  
                    FR_b = get_FR_batch(FR, i_range, device, seq_length=seq_length)
                    observation,rew,act = get_batch(game_info,reward,action,i_range,device,seq_length=seq_length,mode='voc')   
                    yield FR_b, observation,rew,act, i_range
                elif mode=='FRonly':
                    FR_b = get_FR_batch(FR, i_range, device, seq_length=seq_length)
                    yield FR_b, i_range
                elif mode=='boardonly':
                    observation,rew,act = get_batch(game_info,reward,action,i_range,device,seq_length=seq_length,mode='voc')  
                    yield observation,rew,act, i_range
            elif tensor==True:
                if mode=='alldata':  
                    FR_b = get_FR_batch_fromtensor(FR, i_range, device, seq_length=seq_length)
                    observation,rew,act = get_batch_from_torchtensor(game_info,reward,action,i_range,device,seq_length=seq_length,mode='voc')   
                    yield FR_b, observation,rew,act, i_range
                elif mode=='FRonly':
                    FR_b = get_FR_batch_fromtensor(FR, i_range, device, seq_length=seq_length)
                    yield FR_b, i_range
                elif mode=='boardonly':
                    observation,rew,act = get_batch_from_torchtensor(game_info,reward,action,i_range,device,seq_length=seq_length,mode='voc')  
                    yield observation,rew,act, i_range
    elif return_index==True:
        for i in rng:
            ii = list(range(i,min(i+batch_size,len(idx))))
            i_range = idx[ii]
            ind = get_batch_from_torchtensor(game_info,reward,action,i_range,device,seq_length=seq_length,mode='voc',return_index=True)  
            yield ind

def load_trained_tssm_statedict_and_configurations(model_path):
    if '.pkl' in model_path:
        with open(model_path, 'rb') as f:
            x = pickle.load(f)
            state_dict = x[0]
            tssm_configs = x[-1]
    elif '.shelve' in model_path:
        with shelve.open(model_path) as db:
            state_dict = db['state_dict']
            tssm_configs = db['model_configs']
    return state_dict, tssm_configs
            
class checkpoint_and_stoptraining_tssm():
    def __init__(self,configs,path=[],load_state=False):
        self.cache_loss = 1e100
        self.test_error = []
        self.train_error = []
        self.check_count = 0
        self.path = path
        self.configs=configs
        self.cache_point=0
        if load_state==True:
            self.load_state()
    
    def load_state(self):
        with shelve.open(self.path) as db:
            self.state_dict = db['state_dict']
            (self.train_error,self.test_error,self.cache_point) = db['training_records']
            self.cache_loss = self.test_error[self.cache_point]
            self.check_count=len(self.test_error)
            self.configs = db['configs']
    
    def checkpoint(self,trainerror, testerror, model):
        self.test_error.append(testerror)
        self.train_error.append(trainerror)
        
        if self.test_error[-1]<self.cache_loss:
            self.cache_loss = self.test_error[-1]
    
            with shelve.open(self.path) as db:
                # db["model"] = model
                db['configs'] = self.configs
                db['state_dict'] = model.state_dict()
                self.state_dict = model.state_dict()
                
                training_records = (self.train_error,self.test_error,self.check_count)
                self.cache_point = self.check_count
                db['training_records'] = training_records
                print('checkpoint saved')
        else:
            with shelve.open(self.path) as db:
                db["training_records"] = (self.train_error,self.test_error,self.cache_point)   
        self.check_count+=1
        
        # if self.check_count-self.cache_point>200:
        #     raise StopTrainingException
            
class StopTrainingException(Exception):
    pass     

def save_model(path,configs,state_dict):
    p = os.path.split(path)[0]
    if not os.path.exists(p):
        os.makedirs(p)
    with open(path, 'wb') as f:
        pickle.dump([state_dict,configs],f)   
# In[]
import itertools
voc_staticitems = list(range(9))
voc_ghost1 = list(range(5))
voc_ghost2 = list(range(5))
voc_pacman = list(range(2))
combinations = list(itertools.product(voc_staticitems, voc_ghost1, voc_ghost2,voc_pacman))
combinations = [x for x in combinations if not (  (x[0]==1) & (((x[1] == 4 )|(x[2] == 4)) | (x[3]==1))  )]
#                                                   wall         ghost1 & ghost2 flashing    pacman        
com = np.array(combinations)

voc =np.concatenate( (np.eye(9)[com[:,0]][:,1:],np.eye(5)[com[:,1]][:,1:],np.eye(5)[com[:,2]][:,1:],np.eye(2)[com[:,3]][:,1:]), axis=1 )
vocabulary_ = torch.tensor(voc)

voc_props = np.unique(voc[:,:8],axis=0)
voc_chara = np.unique(voc[:,8:],axis=0)
props_class = voc_props.shape[0]
chara_class = voc_chara.shape[0]
v1 = torch.tensor(voc_props)
v2 = torch.tensor(voc_chara)
# In[]
def loaddata(i,DataPath, boarddata_path):
    p = os.path.join(DataPath, boarddata_path)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Expected directory not found: {p}")
    files = os.listdir(p)
    with open(p+'/'+files[i], 'rb') as f:
        tiles, actions, ind = pickle.load(f)  
    tiles_cat = np.array([x[0] for x in tiles])
    # import ipdb;ipdb.set_trace()
    B,T,H,W,C = tiles_cat.shape
    x_flat = torch.tensor(tiles_cat.reshape(-1,tiles_cat.shape[-1]))
    props_indices = torch.where(torch.all(v1 == x_flat[:, None, :8], dim=2))[1]
    chara_indices = torch.where(torch.all(v2 == x_flat[:, None, 8:], dim=2))[1]
    del x_flat
    game_info = np.stack((props_indices.reshape(B,T,H,W),chara_indices.reshape(B,T,H,W)),axis=-1)
    
    return game_info, actions, ind

# In[]
def find_unique_rows(input_2darray):
    unique_vectors, indices = np.unique(input_2darray, axis=0, return_inverse=True)
    return unique_vectors, indices

# def one_hot_encode_tile_states(game_info):
#     s,t,h,w,d = game_info.shape
#     g = einops.rearrange(game_info,'s t h w d->(s h w) (t d)') 
#     v,i = find_unique_rows(g)
#     encoder = OneHotEncoder(categories='auto')
#     encoder.fit(i.reshape(-1, 1))
#     one_hot_encoded = encoder.transform(i.reshape(-1, 1)).toarray()
#     one_hot_encoded = einops.rearrange(one_hot_encoded,'(s h w) c->s h w c',s=s,h=h,w=w) 
#     return one_hot_encoded, encoder

def fit_one_hot_encoder(unique_states):
    encoder = OneHotEncoder(categories='auto')
    v,i = find_unique_rows(unique_states)
    encoder.fit(i.reshape(-1, 1))
    return encoder
    
def turn_states_into_indices(g, unique_states):
    indices = []
    row_map = {tuple(row): idx for idx, row in enumerate(unique_states)}
    # Iterate over each row in B
    for row in g:
        # Convert the row to a tuple and get the index from the row_map
        row_idx = row_map.get(tuple(row))
        indices.append(row_idx)
    indices_array = np.array(indices)
    return indices_array
    
def indices_encode_tile_states(game_info,unique_states):
    s,t,h,w,d = game_info.shape
    g = einops.rearrange(game_info,'s t h w d->(s h w) (t d)') 
    i = turn_states_into_indices(g, unique_states)
    i = einops.rearrange(i,'(s h w)->s h w',s=s,h=h,w=w) 
    return i

def one_hot_encode_tile_states(game_info,encoder,unique_states):
    s,t,h,w,d = game_info.shape
    g = einops.rearrange(game_info,'s t h w d->(s h w) (t d)') 
    i = turn_states_into_indices(g, unique_states)
    one_hot_encoded = encoder.transform(i.reshape(-1, 1)).toarray()
    one_hot_encoded = einops.rearrange(one_hot_encoded,'(s h w) c->s h w c',s=s,h=h,w=w) 
    return one_hot_encoded

def extract_all_states(configs):
    u = []
    for i_dataset in configs['train_files']:
        print(i_dataset)
        game_info,action,inds = loaddata(i_dataset,configs['DataPath'],configs['boarddata_path'])
        s,t,h,w,d = game_info.shape
        g = einops.rearrange(game_info,'s t h w d->(s h w) (t d)') 
        v,i = find_unique_rows(g)
        u.append(v)
    unique_states, _ = find_unique_rows(np.vstack(u))
    return unique_states

class logger():
    def __init__(self,log_items):
        for i in log_items:
            setattr(self,i,[])
            
    def update(self,keys,values):
        for k,v in zip(keys,values):
            l = getattr(self,k)
            l.append(v)
            setattr(self,k,l)
# In[]
# num_epochs = 200
SYMBOLS = dict(
    ghost="\uf6e2",
    circle="\uf111",
    cat="\uf6be",
    eye="\uf06e",
    monkey="\uf6fb",
    fix="\uf648",
    apple="\uf5d1",
    moon="\uf186",
    pacman="\ue125",
    quote_left="\uf10d",
    infinity="\uf534",
    stroopwafel='\uf551',
    pizza_slice='\uf818',
    diagram_project='\uf542',
    bowling_ball="\uf436",
    basketball="\uf434",
    circle_chevron_left="\uf137",
    left="\uf060",
    right="\uf061",
    down="\uf063",
    up="\uf062"
    )

def get_marker(symbol):
    v, codes = TextToPath().get_text_path(FP, symbol)
    v = np.array(v)
    mean = np.mean([np.max(v, axis=0), np.min(v, axis=0)], axis=0)
    return Path(v - mean, codes, closed=False)  

def gm(symbol):
    tp = TextToPath()
    v, codes = tp.get_text_path(FP, symbol)
    v = np.array(v)
    mean = np.mean([np.max(v, axis=0), np.min(v, axis=0)], axis=0)
    return Path(v - mean, codes, closed=False)  

def mat2scatters(mat,ax,marker,s=250,color=[0,0,0],edgecolor=None,continuous=False,alpha=1,max_value=None):
    if continuous==False:
        y, x = np.where(mat==1)
        ax.scatter(x+0.5,y+0.5,marker=marker,color=color,s=s,edgecolor=edgecolor,alpha=alpha)
    elif continuous==True:
        if np.any(mat!=0):
            value = mat[mat!=0]/max_value
            y, x = np.where(mat!=0)
            ax.scatter(x+0.5,y+0.5,marker=marker,color=color,s=s,edgecolor=edgecolor,alpha=[np.array(alpha)*x for x in value])
    
def render_board_old(src,title=None,hold_on=False,ax=None,f=None, intensity=1,plot_wall=True,continuous=False,mapalpha=1,
                 bean_color=[1,1,1],max_value=None,wallcolor=None,tilecolor=None,markersize=1,linecolor=[1,1,1]):
    ms = markersize
    if ax is None:
        f, ax = plt.subplots(figsize=(src.shape[1]*0.3,src.shape[0]*0.3))
    if plot_wall:
        if wallcolor is not None and tilecolor is not None:
            sns.heatmap(1-src[...,0], ax=ax,linewidth=1,linecolor=linecolor, annot=False, cbar=False,cmap=[wallcolor,tilecolor],alpha=mapalpha)
        else:
            sns.heatmap(1-src[...,0], ax=ax,linewidth=1, annot=False, cbar=False,cmap="YlGnBu",alpha=mapalpha)
    # import ipdb;ipdb.set_trace()
    if max_value is None:
        max_value = src[...,1:].max()
    mat2scatters(src[...,1], ax, get_marker(SYMBOLS['circle']),s=20*ms,color=bean_color,alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,2], ax, get_marker(SYMBOLS['circle']),s=90*ms,color=bean_color,alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,3], ax, get_marker(SYMBOLS['diagram_project']),color=[1,.3,0],s=180*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,4], ax, get_marker(SYMBOLS['apple']),color=[1,.3,0],s=180*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,5], ax, get_marker(SYMBOLS['pizza_slice']),color=[1,.3,0],s=180*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,6], ax, get_marker(SYMBOLS['basketball']),color=[0.4,.7,0],s=180*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,7], ax, get_marker(SYMBOLS['bowling_ball']),color=[1,.6,0],s=180*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    
    mat2scatters(src[...,8], ax, get_marker(SYMBOLS['ghost']),color=[0.8,0.0,0],s=300*ms,edgecolor=[1,1,1],alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,9], ax, get_marker(SYMBOLS['quote_left']),color=[0.,0.5,0.8],s=70*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,10], ax, get_marker(SYMBOLS['ghost']),color=[0.,0.5,0.8],s=300*ms,edgecolor=[1,1,1],alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,11], ax, get_marker(SYMBOLS['ghost']),color=[1,1,1],edgecolor=[0.,0.5,0.8],s=300*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    
    mat2scatters(src[...,12], ax, get_marker(SYMBOLS['ghost']),color=[0.9,0.7,0],s=300*ms,edgecolor=[1,1,1],alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,13], ax, get_marker(SYMBOLS['quote_left']),color=[0.,0.5,0.8],s=70*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,14], ax, get_marker(SYMBOLS['ghost']),color=[0.,0.5,0.8],s=300*ms,edgecolor=[1,1,1],alpha=intensity,continuous=continuous,max_value=max_value)
    mat2scatters(src[...,15], ax, get_marker(SYMBOLS['ghost']),color=[1,1,1],edgecolor=[0.,0.5,0.8],s=300*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    
    mat2scatters(src[...,16], ax, get_marker(SYMBOLS['moon']),s=250*ms,color=[0.8,0.75,0],alpha=intensity,continuous=continuous,max_value=max_value)
    if title is not None:
        plt.title(title)
    if hold_on==False:
        plt.show()
    return f,ax

def layicon(mat,ax,name,s=0.01,color=[0,0,0],edgecolor=None,continuous=False,alpha=1,max_value=None):
    image = mpimg.imread(iconpath+ name +'.png')

    if continuous==False:
        y, x = np.where(mat==1)
        image[...,3] = image[...,3]*alpha
        for yi,xi in zip(y,x):
            imagebox = OffsetImage(image, zoom=s)
            ab = AnnotationBbox(imagebox, (xi+.5, yi+.5), frameon=False)
            ax.add_artist(ab)
    elif continuous==True:
        if np.any(mat!=0):
            value = mat[mat!=0]/max_value

            y, x = np.where(mat!=0)
            for yi,xi,v in zip(y,x,value):
                img = image.copy()
                img[...,3] = img[...,3]*alpha*v
                imagebox = OffsetImage(img, zoom=s)
                ab = AnnotationBbox(imagebox, (xi+.5, yi+.5), frameon=False)
                ax.add_artist(ab)


def render_board(src,title=None,hold_on=False,ax=None,f=None, intensity=1,plot_wall=True,continuous=False,mapalpha=1,
                 max_value=None,tilecolor=[.4,.6,.7],wallcolor=[.2,.2,.3],markersize=1,linecolor=[0,0,0]):
    ms = markersize
    # ax=None 
    if ax is None:
        f, ax = plt.subplots(figsize=(src.shape[1]*0.3,src.shape[0]*0.3))
    if plot_wall:
        if wallcolor is not None and tilecolor is not None:
            sns.heatmap(1-src[...,0], ax=ax,linewidth=1,linecolor=linecolor, annot=False, cbar=False,cmap=[wallcolor,tilecolor],alpha=mapalpha)
        else:
            sns.heatmap(1-src[...,0], ax=ax,linewidth=1, annot=False, cbar=False,cmap="YlGnBu",alpha=mapalpha)
    # import ipdb;ipdb.set_trace()
    if max_value is None:
        max_value = src[...,1:].max()
    
    # for i in range(1,8):
    #     for j in [8,10,11,12,14,15]:
    #         src[src[...,j]==1,i]=0
    
    layicon(src[...,1], ax, 'dot',s=0.0085*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,2], ax, 'energizer',s=0.025*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,3], ax, 'apple',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,4], ax, 'cherry',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,5], ax, 'melon',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,6], ax, 'orange',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,7], ax, 'strawberry',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,8], ax, 'ghost1',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,9], ax, 'eyes1',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,10], ax, 'scaredghost1',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,11], ax, 'flashghost1',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,12], ax, 'ghost2',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,13], ax, 'eyes2',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,14], ax, 'scaredghost2',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    layicon(src[...,15], ax, 'flashghost2',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)
    
    layicon(src[...,16], ax, 'pacman',s=0.006*ms,alpha=intensity,continuous=continuous,max_value=max_value)

    if title is not None:
        plt.title(title)
    if hold_on==False:
        plt.show()
    return f,ax
    
# def render_board(src,title=None,hold_on=False,ax=None,f=None, intensity=1,plot_wall=True):
#     if ax is None:
#         f, ax = plt.subplots(figsize=(src.shape[1]*0.3,src.shape[0]*0.3))
#     if plot_wall:
#         sns.heatmap(1-src[...,0], ax=ax,linewidth=1, annot=False, cbar=False,cmap="YlGnBu")
#     mat2scatters(src[...,1], ax, get_marker(SYMBOLS['circle']),s=20,color=[1,1,1],alpha=intensity)
#     mat2scatters(src[...,2], ax, get_marker(SYMBOLS['circle']),s=90,color=[1,1,1],alpha=intensity)
#     mat2scatters(src[...,3], ax, get_marker(SYMBOLS['diagram_project']),color=[1,.3,0],s=180,alpha=intensity)
#     mat2scatters(src[...,4], ax, get_marker(SYMBOLS['apple']),color=[1,.3,0],s=180,alpha=intensity)
#     mat2scatters(src[...,5], ax, get_marker(SYMBOLS['pizza_slice']),color=[1,.3,0],s=180,alpha=intensity)
#     mat2scatters(src[...,6], ax, get_marker(SYMBOLS['basketball']),color=[0.4,.7,0],s=180,alpha=intensity)
#     mat2scatters(src[...,7], ax, get_marker(SYMBOLS['bowling_ball']),color=[1,.6,0],s=180,alpha=intensity)
    
#     mat2scatters(src[...,8], ax, get_marker(SYMBOLS['ghost']),color=[0.8,0.0,0],s=300,edgecolor=[1,1,1],alpha=intensity)
#     mat2scatters(src[...,9], ax, get_marker(SYMBOLS['quote_left']),color=[0.,0.5,0.8],s=70,alpha=intensity)
#     mat2scatters(src[...,10], ax, get_marker(SYMBOLS['ghost']),color=[0.,0.5,0.8],s=300,edgecolor=[1,1,1],alpha=intensity)
#     mat2scatters(src[...,11], ax, get_marker(SYMBOLS['ghost']),color=[1,1,1],edgecolor=[0.,0.5,0.8],s=300,alpha=intensity)
    
#     mat2scatters(src[...,12], ax, get_marker(SYMBOLS['ghost']),color=[0.9,0.7,0],s=300,edgecolor=[1,1,1],alpha=intensity)
#     mat2scatters(src[...,13], ax, get_marker(SYMBOLS['quote_left']),color=[0.,0.5,0.8],s=70,alpha=intensity)
#     mat2scatters(src[...,14], ax, get_marker(SYMBOLS['ghost']),color=[0.,0.5,0.8],s=300,edgecolor=[1,1,1],alpha=intensity)
#     mat2scatters(src[...,15], ax, get_marker(SYMBOLS['ghost']),color=[1,1,1],edgecolor=[0.,0.5,0.8],s=300,alpha=intensity)
    
#     mat2scatters(src[...,16], ax, get_marker(SYMBOLS['moon']),s=250,color=[0.8,0.75,0],alpha=intensity)
#     if title is not None:
#         plt.title(title)
#     if hold_on==False:
#         plt.show()
#     return f,ax

def render_2frame_seq(src_seq,plot_wall=True,title=None,hold_on=False,ax=None,f=None,mapalpha=1,
                      wallcolor=None,tilecolor=None,markersize=1,r=0.3,continuous=False,linecolor=[0,0,0]):
    if ax is None:
        f, ax = plt.subplots(figsize=(src_seq.shape[1]*r,src_seq.shape[0]*r))
    
    src_seq = src_seq.copy()
    for i in np.concatenate((range(1,8),range(18,25))):
        for j in [8,10,11,12,14,15]:
            src_seq[src_seq[...,j]>0,i]=0
                
    src = src_seq[...,:17]
    f, ax = render_board(src,title=title,hold_on=True,ax=ax,f=f,intensity=0.5,plot_wall=plot_wall,
                 mapalpha=mapalpha,wallcolor=wallcolor,tilecolor=tilecolor,markersize=markersize,continuous=continuous,linecolor=linecolor)
    src = src_seq[...,17:]
    f, ax = render_board(src,title=title,hold_on=hold_on,ax=ax,f=f,intensity=1,plot_wall=False,
                 mapalpha=mapalpha,wallcolor=wallcolor,tilecolor=tilecolor,markersize=markersize,continuous=continuous,linecolor=linecolor)
    return f,ax

def overlay_attn_old(src,a,title=None,hold_on=False,markersize=1):
    ms = markersize
    f, ax = plt.subplots(figsize=(src.shape[1]*0.3,src.shape[0]*0.3))
    
    sns.heatmap(1-src[...,0], ax=ax,linewidth=1, annot=False, cbar=False,cmap="YlGnBu")
    mat2scatters(src[...,1], ax, get_marker(SYMBOLS['circle']),s=20*ms,color=[1,1,1])
    mat2scatters(src[...,2], ax, get_marker(SYMBOLS['circle']),s=90*ms,color=[1,1,1])
    mat2scatters(src[...,3], ax, get_marker(SYMBOLS['diagram_project']),color=[1,.3,0],s=180*ms)
    mat2scatters(src[...,4], ax, get_marker(SYMBOLS['apple']),color=[1,.3,0],s=180*ms)
    mat2scatters(src[...,5], ax, get_marker(SYMBOLS['pizza_slice']),color=[1,.3,0],s=180*ms)
    mat2scatters(src[...,6], ax, get_marker(SYMBOLS['basketball']),color=[0.4,.7,0],s=180*ms)
    mat2scatters(src[...,7], ax, get_marker(SYMBOLS['bowling_ball']),color=[1,.6,0],s=180*ms)
    
    mat2scatters(src[...,8], ax, get_marker(SYMBOLS['ghost']),color=[0.8,0.0,0],s=300*ms,edgecolor=[1,1,1])
    mat2scatters(src[...,9], ax, get_marker(SYMBOLS['quote_left']),color=[0.,0.5,0.8],s=70*ms)
    mat2scatters(src[...,10], ax, get_marker(SYMBOLS['ghost']),color=[0.,0.5,0.8],s=300*ms,edgecolor=[1,1,1])
    mat2scatters(src[...,11], ax, get_marker(SYMBOLS['ghost']),color=[1,1,1],edgecolor=[0.,0.5,0.8],s=300*ms)
    
    mat2scatters(src[...,12], ax, get_marker(SYMBOLS['ghost']),color=[0.9,0.7,0],s=300*ms,edgecolor=[1,1,1])
    mat2scatters(src[...,13], ax, get_marker(SYMBOLS['quote_left']),color=[0.,0.5,0.8],s=70*ms)
    mat2scatters(src[...,14], ax, get_marker(SYMBOLS['ghost']),color=[0.,0.5,0.8],s=300*ms,edgecolor=[1,1,1])
    mat2scatters(src[...,15], ax, get_marker(SYMBOLS['ghost']),color=[1,1,1],edgecolor=[0.,0.5,0.8],s=300*ms)
    
    mat2scatters(src[...,16], ax, get_marker(SYMBOLS['moon']),s=250*ms,color=[0.8,0.75,0])
    if title is not None:
        plt.title(title)
        
    # normed_a = (a - a.min()) / (a.max() - a.min())
    alpha_colormap = LinearSegmentedColormap.from_list(
    name='custom_alpha_cmap',
    colors=[(0, 0, 0, 1-alpha*0.1) for alpha in range(0,11)],
    N=a.size)
    sns.heatmap(a, cmap=alpha_colormap, cbar=False)
    if hold_on==False:
        plt.show()

def overlay_attn(src,a,title=None,hold_on=False,markersize=1):
    ms = markersize
    f, ax = plt.subplots(figsize=(src.shape[1]*0.3,src.shape[0]*0.3))
    sns.heatmap(1-src[...,0], ax=ax,linewidth=1, annot=False, cbar=False,cmap="YlGnBu")

    layicon(src[...,1], ax, 'dot',s=0.008*ms)
    layicon(src[...,2], ax, 'energizer',s=0.022*ms)
    layicon(src[...,3], ax, 'apple',s=0.005*ms)
    layicon(src[...,4], ax, 'cherry',s=0.005*ms)
    layicon(src[...,5], ax, 'melon',s=0.005*ms)
    layicon(src[...,6], ax, 'orange',s=0.005*ms)
    layicon(src[...,7], ax, 'strawberry',s=0.005*ms)
    layicon(src[...,8], ax, 'ghost1',s=0.005*ms)
    layicon(src[...,9], ax, 'eyes1',s=0.005*ms)
    layicon(src[...,10], ax, 'scaredghost1',s=0.005*ms)
    layicon(src[...,11], ax, 'flashghost1',s=0.005*ms)
    layicon(src[...,12], ax, 'ghost2',s=0.005*ms)
    layicon(src[...,13], ax, 'eyes2',s=0.005*ms)
    layicon(src[...,14], ax, 'scaredghost2',s=0.005*ms)
    layicon(src[...,15], ax, 'flashghost2',s=0.005*ms)
    if title is not None:
        plt.title(title)
        
    # normed_a = (a - a.min()) / (a.max() - a.min())
    alpha_colormap = LinearSegmentedColormap.from_list(
    name='custom_alpha_cmap',
    colors=[(0, 0, 0, 1-alpha*0.1) for alpha in range(0,11)],
    N=a.size)
    sns.heatmap(a, cmap=alpha_colormap, cbar=False)
    if hold_on==False:
        plt.show()

def overlay_attn_on_seq_old(src_seq,a,title=None,hold_on=False,markersize=1,tilecolor=[.4,.6,.7],wallcolor=[.2,.2,.3]):
    render_2frame_seq(src_seq,title=title,hold_on=True,markersize=markersize,tilecolor=tilecolor,wallcolor=wallcolor)
    # normed_a = (a - a.min()) / (a.max() - a.min())
    alpha_colormap = LinearSegmentedColormap.from_list(
    name='custom_alpha_cmap',
    colors=[(0, 0, 0, 1-alpha*0.1) for alpha in range(0,11)],
    N=a.size)
    sns.heatmap(a, cmap=alpha_colormap, cbar=False)
    if hold_on==False:
        plt.show()
    
def overlay_attn_on_seq(src_seq,a,title=None,hold_on=False,markersize=1,tilecolor=[.4,.6,.7],wallcolor=[.2,.2,.3]):
    render_2frame_seq(src_seq,title=title,hold_on=True,markersize=markersize,tilecolor=tilecolor,wallcolor=wallcolor)
    # normed_a = (a - a.min()) / (a.max() - a.min())
    alpha_colormap = LinearSegmentedColormap.from_list(
    name='custom_alpha_cmap',
    colors=[(0, 0, 0, 1-alpha*0.1) for alpha in range(0,11)],
    N=a.size)
    sns.heatmap(a, cmap=alpha_colormap, cbar=False)
    if hold_on==False:
        plt.show()
        
def visualize(src,pred):
    render_board(src)
    render_board(pred)

def deformat(x):
    if type(x)==torch.Tensor:
        return x.cpu().detach().numpy()
    elif type(x)==tuple or type(x)==list:
        return [deformat(x_) for x_ in x]
    else:
        return x

def test(model,test_idx,test_data,configs):
    idx = test_idx
    np.random.shuffle(idx)
    obs_test,rew_test,act_test = get_batch(test_data[0],test_data[1],test_data[2],idx[:128],configs['device'],seq_length=configs['seq_length'],mode='voc')   
    loss_, recon_obs_, pred_obs_,action_loss_,recon_loss_,latent_kl_,_ = model(obs_test, act_test, rew_test,start_pos=idx[:128])
    return (loss_, recon_obs_, pred_obs_,action_loss_,recon_loss_,latent_kl_), (obs_test,rew_test,act_test)

# In[] spk

class cache_anything():
    def __init__(self, path=[],name=None):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.count=0
        self.name = name or 'cache_data'
    
    def cache(self,cache):
        with open(self.path+self.name+'_'+str(self.count)+'.pkl','wb') as f:
            pickle.dump(cache,f)
        self.count+=1

def go_through_dataset(model,game_info,game_info_gpu,reward_gpu,action_gpu,FR_gpu,device,seq_length,batch_size,returnFR=False):
    y_train=[]
    losses_train=[]
    y_test=[]
    losses_test=[]
    FR_train=[]
    FR_test=[]
    train_idx, test_idx = split_train_test_dataset(game_info.shape[0],mode='middle',seq_length=seq_length)
    for i_train, (FR_, observation, rew, act, idx) in enumerate(
            generate_test_batch(game_info_gpu, reward_gpu, action_gpu, 
                                FR_gpu, train_idx, device, seq_length,batch_size=batch_size, tensor=True)):
        y, losses = model(observation, act, rew, FR_, start_pos=idx)
        y_train.append(deformat(y))
        losses_train.append(deformat(losses))
        FR_train.append(deformat(FR_))
    for i_test, (FR_, observation, rew, act, idx) in enumerate(
            generate_test_batch(game_info_gpu, reward_gpu, action_gpu, 
                                FR_gpu, test_idx, device, seq_length,batch_size=batch_size, tensor=True)):
        y, losses = model(observation, act, rew, FR_, start_pos=idx)
        y_test.append(deformat(y))
        losses_test.append(deformat(losses))
        FR_test.append(deformat(FR_))
    if returnFR==False:
        return y_train, losses_train, y_test, losses_test
    else:
        return y_train, losses_train, y_test, losses_test,FR_train, FR_test

def split_train_test_dataset(n_sample, mode='last',ratio=.1, seq_length=0,exist_history=False):
    if exist_history==True:
        start = seq_length
    else:
        start = 0
    if mode=='last':
        i_ = round(n_sample*(1-ratio))
        train_idx = np.array(range(start,i_-seq_length))
        test_idx = np.array(range(i_, n_sample-seq_length))
    if mode=='middle':
        i1 = round(n_sample*(1-ratio)/2)
        i2 = round(n_sample*(1+ratio)/2)
        train_idx = np.concatenate((range(start,i1-seq_length),range(i2,n_sample-seq_length)))
        test_idx = np.array(range(i1, i2-seq_length))

    return train_idx, test_idx


def visualizeFR(FR, y, sample=0,nrows=5, ncols=5,title=None):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    for j, ax in enumerate(axes.flat):
        ax.plot(deformat(FR)[sample,:,j])
        ax.plot(deformat(y)[sample,:,j])
        ax.set_title(f'Neuron {j}')
        ax.axis('off')
    if title is not None:
        fig.text(0.45,0.05,title)
    plt.show()

def normalize_data(data,normalization,device):
    if normalization is None:
        FR_gpu = format_processor(data, device)
    elif normalization == 'zscore' and normalization!='stdnorm_timeregress':
        temp = data
        FR_mean = temp.mean(0)[np.newaxis,:]
        FR_std = temp.std(0)[np.newaxis,:]
        FR_zscored = (temp-FR_mean)/FR_std
        FR_gpu = format_processor(FR_zscored, device)
    elif normalization == 'maxnorm':
        temp = data
        FR_max = temp.max(0)[np.newaxis,:]
        FR_maxnormed = temp/FR_max
        FR_gpu = format_processor(FR_maxnormed, device)
    elif 'stdnorm' in normalization:
        temp = data
        FR_std = temp.std(0)[np.newaxis,:]
        FR_stdnormed = temp/FR_std*10
        if 'timeregress' in normalization:
            FR_stdnormed = regress_time_out(FR_stdnormed,order=3) 
        FR_gpu = format_processor(FR_stdnormed, device)
        
    elif normalization == 'invcdf':
        temp = np.zeros_like(data)
        for i in range(temp.shape[1]):
            sorted_data = np.sort(temp[:,i])
            empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            interp_cdf = interp1d( sorted_data, empirical_cdf, kind='linear', bounds_error=False, fill_value=(sorted_data[0], sorted_data[-1]))
            temp[:,i] = interp_cdf(temp[:,i])*10
        FR_gpu = format_processor(temp, device)
    return FR_gpu
    
def regress_time_out(X,order=3):
    t = np.array(list(range(X.shape[0]))).reshape(-1,1)
    t=t-t.mean()
    tx= np.hstack([t**i for i in range(1,order+1)])
    
    reg=[]
    res = np.zeros(X.shape)
    for i in range(X.shape[1]):
        reg.append(LinearRegression().fit(tx,X[:,i]))
        res[:,i] = X[:,i]-reg[i].predict(tx)        
    return res

def interleave_alongaxis1(a,b):
    c = np.empty((a.shape[0],a.shape[1]*2), dtype=a.dtype)
    c[:,0::2] = a
    c[:,1::2] = b            
    return c

class checkpoint_and_stoptraining_spk():
    def __init__(self, configs, path, load_state=False):
        self.cache_loss = 1e100
        self.test_error = []
        self.train_error = []
        self.check_count = 0
        self.path = path
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0])
        self.cache_point=0
        self.records=[]
        self.model_configs = configs
        with shelve.open(self.path) as db:
            db['model_configs'] = configs    
        if load_state==True:
            self.load_state()
    
    def load_state(self):
        with shelve.open(self.path) as db:
            self.state_dict = db['state_dict']
            self.model_configs = db['model_configs']
            (self.train_error,self.test_error,self.cache_point) = db['training_records']
    
    def checkpoint(self,trainerror, testerror, model, records, criter=None):
        self.check_count+=1
        self.test_error.append(testerror)
        self.train_error.append(trainerror)
        self.records.append(records)
        
        if criter is None:
            if self.test_error[-1]<self.cache_loss:
                self.cache_loss = self.test_error[-1]
        
                with shelve.open(self.path) as db:
                    # db["model"] = model
                    db['state_dict'] = model.state_dict()
                    self.state_dict = model.state_dict()
                    
                    training_records = (self.train_error,self.test_error,self.check_count)
                    self.cache_point = self.check_count
                    db['training_records'] = training_records
                    db['extra_records'] = self.records
                    print('checkpoint saved')
            else:
                with shelve.open(self.path) as db:
                    db["training_records"] = (self.train_error,self.test_error,self.cache_point)    
                    db['extra_records'] = self.records
        
        else:
            if criter<self.cache_loss:
                self.cache_loss = criter
                with shelve.open(self.path) as db:
                    # db["model"] = model
                    db['state_dict'] = model.state_dict()
                    self.state_dict = model.state_dict()
                    
                    training_records = (self.train_error,self.test_error,self.check_count)
                    self.cache_point = self.check_count
                    db['training_records'] = training_records
                    db['extra_records'] = self.records
                    print('checkpoint saved')
            else:
                with shelve.open(self.path) as db:
                    db["training_records"] = (self.train_error,self.test_error,self.cache_point)    
                    db['extra_records'] = self.records        

        if self.check_count-self.cache_point>200e10:
            raise StopTrainingException
            
def perturb_FR(FR_batch, ratio=.2, scale=1.):
    n = round(FR_batch.shape[2]*ratio)
    i_neuron = np.random.choice(FR_batch.shape[2],n)
    noise = torch.randn(FR_batch.shape[0],FR_batch.shape[1],n, device=FR_batch.device) * scale
    FR_batch[...,i_neuron] = FR_batch[...,i_neuron].add_(noise)
    return FR_batch

def compute_similarity(data1,data2,slice_axis=0, norm=False):
    data1 = np.moveaxis(deformat(data1),slice_axis,0).reshape(data1.shape[0],-1)
    data2 = np.moveaxis(deformat(data2),slice_axis,0).reshape(data2.shape[0],-1)
    cosdist = np.diag(cosine_distances(data1,data2))
    if norm==True:
        cosdist = (cosdist-cosdist.min())/(cosdist.max()-cosdist.min())
    return cosdist

def get_contrastive_batch(game_info_gpu, reward_gpu, action_gpu, FR, all_idx, i_starts, 
                          device, seq_length, batch_size, norm=False,single_batch=False,
                          returnFR=False, dist_type='cosine_dist'):
    
    dat1 = get_batch_from_torchtensor(game_info_gpu,reward_gpu,action_gpu,i_starts,device,seq_length=seq_length,mode='voc')   
    FR1 = get_FR_batch(FR, i_starts, 'cpu', seq_length=seq_length)
    if single_batch==False:
        ref_idx = np.random.choice(all_idx,size=(batch_size,))
        dat2 = get_batch_from_torchtensor(game_info_gpu,reward_gpu,action_gpu,ref_idx,device,seq_length=seq_length,mode='voc')   
        FR2 = get_FR_batch(FR, ref_idx, 'cpu', seq_length=seq_length)
        
        dissimilarity = torch.Tensor(compute_similarity(FR1, FR2, norm=norm).copy()).float().to(device)
        if returnFR==False:
            return dat1, dat2, dissimilarity, i_starts, ref_idx
        elif returnFR==True:
            return dat1, dat2, dissimilarity, i_starts, ref_idx, FR1.to(device),FR2.to(device)
        
    elif single_batch==True:
        fr = einops.rearrange(FR1,'b t n->b (t n)')
        if dist_type=='cosine_dist':
            cos_dists = torch.Tensor(cosine_distances(deformat(fr)))
            mask = torch.triu(torch.ones_like(cos_dists),diagonal=1).bool()
            dissimilarity = cos_dists[mask].float().to(device)
        elif dist_type=='euclidean':
            dist = torch.sqrt(((fr.T[:,None,:] - fr.T[:,:,None]) ** 2).sum(dim=0))
            mask = torch.triu(torch.ones_like(dist),diagonal=1).bool()
            dissimilarity = dist[mask].float().to(device)
            
        if returnFR==False:
            return dat1, dissimilarity, i_starts
        elif returnFR==True:
            return dat1, dissimilarity, i_starts, FR1.to(device)

def visualize_similarity(tar,pred,title=None):
    plt.figure(figsize=[10,2])
    plt.plot(deformat(tar))
    plt.plot(deformat(pred))
    plt.title(title)
    plt.show()
    
def balanced_sampling(allidx, i, batch_size, search_range=32):
    b0 = np.array(allidx[i:i+batch_size])
    b1 = b0 + np.random.choice(search_range*2,size=b0.shape)-search_range
    while np.any([x not in allidx for x in b1]):
        b1 = b0 + np.random.choice(search_range*2,size=b0.shape)-search_range
    return [*b0,*b1]

def sort_pred_and_tar(pred, tar):
    out2, idx = torch.sort(tar)
    out1 = pred[idx]
    return out1, out2

def generalized_simloss(tar, pred,mode='vanilla',temperature=1):
    if mode=='vanilla':
        P = F.softmax(-tar/temperature, dim=0)
        Q = F.softmax(-pred/temperature, dim=0)
        loss = torch.sum(P*torch.log(P/Q))
    elif mode=='balanced_kld':
        P = F.softmax(-tar/temperature, dim=0)
        Q = F.softmax(-pred/temperature, dim=0)
        loss = 1/2*(torch.sum(P*torch.log(P/Q)) + torch.sum(Q*torch.log(Q/P)))
    elif mode=='balanced_p':
        # if the mode is balanced_p, the input of target & predicted distance should be matrices instead of triu elements
        diag_mask = torch.eye(tar.shape[0]).bool().to(tar.device)
        pred[diag_mask] = 1e9
        tar[diag_mask] = 1e9
        Q = torch.exp(-pred/temperature)/torch.exp(-pred/temperature).sum()*2
        P = (torch.exp(-tar/temperature)/torch.exp(-tar/temperature).sum(0)+torch.exp(-tar/temperature)/torch.exp(-tar/temperature).sum(1))/2/tar.shape[0]*2 
        mask = torch.triu(torch.ones_like(Q),diagonal=1).bool()
        loss = torch.sum(P[mask]*torch.log(P[mask]/Q[mask]))
    elif mode=='balanced_p_t':
        # if the mode is balanced_p, the input of target & predicted distance should be matrices instead of triu elements
        diag_mask = torch.eye(tar.shape[0]).bool().to(tar.device)
        pred[diag_mask] = 1e9
        tar[diag_mask] = 1e9
        Q = 1/(1+pred)/(1/(1+pred)).sum()*2
        P = (torch.exp(-tar/temperature)/torch.exp(-tar/temperature).sum(0)+torch.exp(-tar/temperature)/torch.exp(-tar/temperature).sum(1))/2/tar.shape[0]*2 
        mask = torch.triu(torch.ones_like(Q),diagonal=1).bool()
        loss = torch.sum(P[mask]*torch.log(P[mask]/Q[mask]))
        
    return loss