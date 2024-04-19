import torch
import torch.nn as nn
import torch.nn.functional as F

def cosdis(x1,x2):
    return (1-torch.cosine_similarity(x1,x2,dim=-1))/2
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def emb_loss(self, emb, adj):
        '''
        Parameters
        ----------
        emb : Tensor
            An MxE tensor, the embedding of the ith node is stored in emb[i,:].
        adj : Tensor
            An MxM tensor, adjacent matrix of the graph.
        
        Returns
        -------
        loss : float
            The link prediction loss.
        '''
        # print()
        emb_norm = emb.norm(dim=1, keepdim=True)
        emb_norm = emb / (emb_norm.cuda() + 1e-6)
        adj_pred = torch.matmul(emb_norm, emb_norm.t())
        adj_pred = (torch.matmul(emb_norm, emb_norm.t())+1)/2
        adj_pred = torch.clamp(adj_pred,min=0.,max=1.)
        loss = torch.mean(-adj.mul((adj_pred+1e-9).log())-(1-adj).mul((1-adj_pred+1e-5).log()))
        # print(torch.isnan((1-adj_pred+1e-5).log()).sum())
        # loss = torch.mean(torch.pow(adj - adj_pred, 2))
        
        return loss
    def rec_loss(self,recx,orix,mask):
        mask_mn1 = mask.t().unsqueeze(-1)
        loss = torch.pow((recx - orix), 2).mul(mask_mn1).mean()
        return loss



    def weighted_wmse_loss(self,input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        ret = torch.mean(ret)
        return ret

    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        
        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res
                            
    def BCE_loss(self,target_pre,sub_target):
        return torch.mean(torch.abs((sub_target.mul(torch.log(target_pre + 1e-10)) \
                                        + (1-sub_target).mul(torch.log(1 - target_pre + 1e-10)))))
    
    