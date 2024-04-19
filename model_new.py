import torch
import torch.nn as nn 
# from Layers import EncoderLayer, DecoderLayer
# from Embed import Embedder, PositionalEncoder
import random
import copy
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from layers_new import  FDModel, MLP,GAT 


def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
def setEmbedingModel(d_list,d_out):
    
    return nn.ModuleList([Mlp(d,d,d_out)for d in d_list])

class Mlp(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)

        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout1:
            out = self.dropout2(out)
        return out

class Net(nn.Module):
    def __init__(self, d_list,num_classes,beta,in_layers,class_emb,adj,rand_seed=0):
        super(Net, self).__init__()
        # self.configs = configs
        self.rand_seed = rand_seed
        
        # Label semantic encoding module
        self.label_embedding = nn.Parameter(torch.eye(num_classes),
                                            requires_grad=True)

        self.label_adj = nn.Parameter(torch.eye(num_classes),
                                      requires_grad=True)
        self.adj = adj

        self.GAT_encoder = GAT(num_classes, class_emb)
        
        # Semantic-guided feature-disentangling module
        self.FD_model = FDModel(d_list,class_emb,
                                512, 512,beta, in_layers, 1,
                                False, 'leaky_relu', 0.1)
        # Classifier
        self.cls_conv = nn.Conv1d(num_classes, num_classes,
                                  512, groups=num_classes)
        


        hidden_list = [512] * (1-1)

        self.NN3 = MLP(512, 512, hidden_list,
                       False, 'leaky_relu', 0.1)

        self.maxP = nn.MaxPool2d((len(d_list),1))
        self.cuda()
        self.reset_parameters()
    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        nn.init.normal_(self.label_embedding)
        nn.init.normal_(self.label_adj)

        self.FD_model.reset_parameters()
        self.cls_conv.reset_parameters()
        
    def get_config_optim(self):
        return [{'params': self.FD_model.parameters()},
                {'params': self.cls_conv.parameters()}]

    def forward(self, input,mask):
        # Generating semantic label embeddings via label semantic encoding module

        label_embedding = self.GAT_encoder(self.label_embedding, self.adj)

        Z,confi = self.FD_model(input, label_embedding,mask)  #Z[i]=[128, 260, 512] b c d_e
        p_list = []
        
        for z_i in Z:
            p_ii = self.cls_conv(z_i).squeeze(2)

            p_list.append(p_ii)
        p_pre = p_list

        p = torch.stack(p_list,dim=1)        # b*m*c

        

        mask_confi = (1-mask).mul(confi.t())+mask # b m

        mask_confi = mask_confi/(mask_confi.sum(dim=1,keepdim=True)+1e-9)   # b*m
        
        p = p.mul(mask_confi.unsqueeze(dim=-1))
        p = p.sum(dim = 1)
        

        p = torch.sigmoid(p)
        
        return p, label_embedding, p_pre

def get_model(d_list,num_classes,beta,in_layers,class_emb,adj,rand_seed=0):
    
    model = Net(d_list,num_classes=num_classes,beta=beta,in_layers=in_layers,class_emb=class_emb,adj=adj,rand_seed=rand_seed)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() 
                                    else 'cpu'))
    return model
    
if __name__=="__main__":
    # input=torch.ones([2,10,768])
    from MLdataset import getIncDataloader
    dataloder,dataset = getIncDataloader('/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view.mat','/disk/MATLAB-NOUPLOAD/MyMVML-data/corel5k/corel5k_six_view_MaskRatios_0_LabelMaskRatio_0_TraindataRatio_0.7.mat',training_ratio=0.7,mode='train',batch_size=3,num_workers=2)
    input = next(iter(dataloder))[0]
    model=get_model(num_classes=260,beta=0.2,in_features=1,class_emb=260,rand_seed=0)
    print(model)
    input = [v_data.to('cuda:0') for v_data in input]
    # print(input[0])
    pred,_,_=model(input)
    print(pred.shape)