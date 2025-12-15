'''
To optimize the target scores for images in an image set
'''
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
import random
import torch
from torch import nn
from fast_soft_sort.pytorch_ops import soft_rank
import pandas as pd
import argparse

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    np.random.seed(seed)
    random.seed(seed)

def corrcoef(target, pred):
    pred_n = pred - pred.mean()
    target_n = target - target.mean()
    pred_n = pred_n / pred_n.norm()
    target_n = target_n / target_n.norm()
    return (pred_n * target_n).sum()

class Spearman(nn.Module):
    def __init__(self,regularization_strength=None):
        super().__init__()
        self.regularization="l2"
        if regularization_strength is None:
            self.regularization_strength=1.0
        else:
            self.regularization_strength = regularization_strength
    def forward(self,target,pred):
        pred = soft_rank(
            pred.unsqueeze(0),
            regularization=self.regularization,
            regularization_strength=self.regularization_strength,
        )[0,:]
        target = soft_rank(
            target.unsqueeze(0),
            regularization=self.regularization,
            regularization_strength=self.regularization_strength,
        )[0,:]
        return corrcoef(target, pred / pred.shape[-1])

def numerai_spearman(target, pred):
    # spearman used for numerai CORR
    return np.corrcoef(target, pred.rank(pct=True, method="first"))[0, 1]
    
def save_scores(config,modelname,i,attacked_scores,moses):
    attacked_scores1 = attacked_scores.detach().numpy()
    moses1 = moses.detach().numpy()

    if config.type == "SROCC":
        savepath = './record/'+modelname+'_attack_scores_maxvar_maxmse_ep{}_lam{}_lammse{}_beta_{}_a{}.npy'.format(i, config.lamda, config.lamda_mse, config.beta, config.alpha)
    np.save(savepath,np.concatenate((moses1[np.newaxis,:],attacked_scores1[np.newaxis,:]),axis=0))
    print('Saved:',savepath)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--beta", "-b", type=float, default=1.0)
    parser.add_argument("--lamda", "-l", type=float, default=0.0001, help='weight of varianace in loss')
    parser.add_argument("--lamda_mse", "-lmse", type=float, default=0.0001, help='weight of MSE in loss')
    parser.add_argument('--epochs', "-e",type=int, default=100000)
    parser.add_argument('--type', type=str, default="SROCC",help='Objective for optimizing correlation')
    parser.add_argument('--model', '-m', type=str, default="DBCNN", choices=['HyperIQA','DBCNN','MANIQA','LIQE'], help='attacked model')
    return parser.parse_args()

def main():
    fix_seed(919)
    config = parse()
    modelname = config.model

    if modelname == 'DBCNN':
        record = np.load('./DBCNN_test_pred_mos_normalized.npy')
        ori_scores = record[:,0]
        moses = record[:,1]
    else:
        raise NotImplementedError

    print('Attacked model:',modelname)

    ori_scores = torch.tensor(ori_scores)
    attacked_scores = ori_scores.clone()
    moses = torch.tensor(moses)
    print("Numerai CORR", numerai_spearman(
        pd.Series(attacked_scores.cpu().detach().numpy()),
        pd.Series(moses.detach().numpy()),
    ))

    # mae = torch.nn.L1Loss()
    mse = nn.MSELoss()
    spearman = Spearman(regularization_strength=config.beta)
    for i in range(config.epochs):
        attacked_scores.requires_grad = True
        if i == 0:
            init_noise = torch.randint(-1, 1, attacked_scores.size()).to(attacked_scores)
            init_noise = init_noise*0.01
            attacked_scores = attacked_scores.detach() + init_noise 
            attacked_scores.requires_grad = True
            optimizer = torch.optim.Adam([attacked_scores], lr=config.alpha)
        if config.type == 'SROCC':
            s = spearman(ori_scores, attacked_scores, ) # SROCC
        cost_var = -config.lamda * torch.var(attacked_scores)
        cost_mse = -config.lamda_mse * mse(attacked_scores,ori_scores)
        loss = s + cost_var + cost_mse
        optimizer.zero_grad()
        loss.backward()
        
        attacked_scores.grad.data[torch.isnan(attacked_scores.grad.data)] = 0
        attacked_scores.grad.data = attacked_scores.grad.data / (attacked_scores.grad.data.reshape(attacked_scores.grad.data.size(0), -1) + 1e-12).norm(dim=1)

        optimizer.step()
        attacked_scores.data.clamp_(min=0, max=100)
        
        if i%1000==0:
            num_corr = numerai_spearman(
                pd.Series(attacked_scores.detach().numpy()),
                pd.Series(moses.detach().numpy()),
            )
            diff = np.abs(attacked_scores.detach().numpy()-ori_scores.detach().numpy())
            diff = np.mean(np.abs(diff))
            print('Epoch {0}:Diff_CORR/Num_CORR/MAE:{1:.4f}\t{2:.4f}\t{3:.4f}'.format(i,s.item(),num_corr,diff))
            print('s/cost_var/cost_mse:',s.item(),cost_var.item(),cost_mse.item())

        if i in [50000,80000,100000]:
            save_scores(config,modelname,i,attacked_scores,moses)
    approx_corr = spearman(ori_scores, attacked_scores)
    real_corr = numerai_spearman(
        pd.Series(attacked_scores.cpu().detach().numpy()),
        pd.Series(ori_scores.cpu().detach().numpy()))


    attacked_scores = attacked_scores.detach().numpy()
    ori_scores = ori_scores.detach().numpy()
    moses = moses.detach().numpy()

    if config.type == "SROCC":
        savepath = './record/'+modelname+'_attack_scores_maxvar_maxmse_ep{}_lam{}_lammse{}_beta_{}_a{}.npy'.format(config.epochs, config.lamda, config.lamda_mse, config.beta, config.alpha)
    np.save(savepath,np.concatenate((moses[np.newaxis,:],attacked_scores[np.newaxis,:]),axis=0))
    print('Saved:',savepath)   

    diffs = np.abs(attacked_scores-ori_scores)
    cost = diffs.sum()
    print('Lambda:',config.lamda)
    print('Cost:',cost)
    print('diffs.max(),diffs.min()',diffs.max(),diffs.min())
    rho_s, _ = spearmanr(attacked_scores, moses)
    rho_p, _ = pearsonr(attacked_scores, moses)
    rho_k, _ = kendalltau(attacked_scores, moses)
    rmse = np.sqrt(np.mean(np.power((attacked_scores-moses),2)))
    mae =np.mean(np.abs(attacked_scores-moses))
    print('After Attack & MOS: SROCC/KROCC/PLCC/RMSE/MAE:\n{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\n'.format(rho_s,rho_k,rho_p,rmse,mae))
    print('Ori & After Attack: SROCC/KROCC/PLCC/RMSE/MAE:\n{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}'.format(rho_s,rho_k,rho_p,rmse,mae))

    rho_s, _ = spearmanr(ori_scores, attacked_scores)
    rho_p, _ = pearsonr(ori_scores, attacked_scores)
    rho_k, _ = kendalltau(ori_scores, attacked_scores)
    rmse = np.sqrt(np.mean(np.power((ori_scores-attacked_scores),2)))
    mae =np.mean(np.abs(ori_scores-attacked_scores))
    # print('{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}'.format(rho_s,rho_k,rho_p,rmse,mae))
    
    error = torch.abs(real_corr - approx_corr)
    print('s/error:{0:.4f} {1:.4f}'.format(approx_corr.item(),error.item()))
    

if __name__ == "__main__":
    main()