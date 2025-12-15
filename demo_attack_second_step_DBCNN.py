'''
To generate the adversarial example for each image, 
and demostrate the difference between score-based attacks and order-based attacks
'''
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import copy
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
import os
import argparse
import sys
sys.path.append('DBCNN')

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
])

transform_wo_norm = transforms.Compose([
    transforms.ToTensor()
])

def norm(x):
    mean = torch.ones((1,3,500,500)).cuda()
    std = torch.ones((1,3,500,500)).cuda()
    mean[:,0,:,:]=0.485
    mean[:,1,:,:]=0.456
    mean[:,2,:,:]=0.406
    std[:,0,:,:]=0.229
    std[:,1,:,:]=0.224
    std[:,2,:,:]=0.225 
    
    x = (x - mean) / std
    
    return x

def de_norm(x):
    mean = torch.ones((1,3,500,500)).cuda()
    std = torch.ones((1,3,500,500)).cuda()
    mean[:,0,:,:]=0.485
    mean[:,1,:,:]=0.456
    mean[:,2,:,:]=0.406
    std[:,0,:,:]=0.229
    std[:,1,:,:]=0.224
    std[:,2,:,:]=0.225 
    
    x = x * std + mean
    
    return x

# save images
def save(pert_image, path):
    pert_image = torch.round(pert_image * 255) / 255
    quantizer = transforms.ToPILImage()
    pert_image = quantizer(pert_image.squeeze())
    pert_image.save(path)

# model to be attacked
use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
import torch.nn as nn
from DBCNN.DBCNN_train import DBCNN
options = {'fc': True}
scnn_root = './DBCNN/pretrained_scnn/scnn.pkl'
load_path = './DBCNN/db_models/livec_net_params_best.pkl'
model = nn.DataParallel(DBCNN(scnn_root, options), device_ids=[0]).cuda()
checkpoint = torch.load(load_path)
print('Load from',load_path)
model.load_state_dict(checkpoint)
model.eval()

df = pd.read_csv('./splitfiles/livec-test.csv')
images_all = df['filename']
mos_all = df['mos']
mos_max = 92.43195
mos_min = 3.42
mos_all = (mos_all-mos_min)/(mos_max-mos_min)*100

mini_list = [i for i in range(len(mos_all))]
images_mini = [images_all[i] for i in mini_list]
mos_mini = [mos_all[i] for i in mini_list]
img_folder = './CLIVE/ChallengeDB_release/Images'
    
'''
FGSM--targeted attack
x: input image
y: target score
'''

def IFGSM_IQA_target(model, x, y, eps=0.01, alpha=0.001, iteration=10, x_val_min=0, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            tmp_adv = norm(x_adv)
            score_adv = model(tmp_adv)
            loss = torch.pow(score_adv - y, 2)
            model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            loss.backward(retain_graph=True)

            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = torch.where(x_adv > x+eps, x+eps, x_adv)
            x_adv = torch.where(x_adv < x-eps, x-eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        score_org = model(norm(x))
        score_adv = model(norm(x_adv))

        return x_adv, score_adv, score_org


def fix_seed(seed):
    torch.manual_seed(seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    np.random.seed(seed)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=0.005)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument('--epochs', "-e", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0.005)

    parser.add_argument("--lamda", "-l", type=float, default=0.0001)
    parser.add_argument("--lamda_mse", "-lmse", type=float, default=0.0001)
    parser.add_argument('--goal', type=str, default="maxvar_maxmse", help='maxvar, maxvar_maxmse')
    parser.add_argument("--beta", "-b", type=float, default=1.0)
    
    parser.add_argument('--type', type=str, default="SROCC", help='Objective for optimizing correlation') # SROCC PLCC SROCC_target
    return parser.parse_args()

def main():
    fix_seed(919)
    config = parse()
    moses = []
    pred_scores = []
    pred_scores_ori = []
    eps = config.eps
    iter = config.iter
    if config.type=="SROCC":
        if config.goal=="maxvar_maxmse":
            save_dir = './generated_adversarial_examples/DBCNN_{}_var_maxmse_epoch{}_lam{}_lammse{}_eps{}_iter{}'.format(config.type,config.epochs,config.lamda,config.lamda_mse,eps,iter) # var
        else:
            save_dir = './generated_adversarial_examples/DBCNN_{}_var_epoch{}_lam{}_eps{}_iter{}'.format(config.type,config.epochs,config.lamda,eps,iter) 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if config.type == "SROCC":
        record_path = './record/DBCNN_attack_scores_maxvar_maxmse_ep{}_lam{}_lammse{}_beta_1.0_a{}.npy'.format(config.epochs, config.lamda, config.lamda_mse, config.alpha)
    print('record_path',record_path)
    
    attack_record = np.load(record_path)
    atttack_goal = attack_record[1,:]
    atttack_goal_mini = atttack_goal
    
    for i in range(len(mini_list)):
        moses.append(mos_mini[i])
        img_path = os.path.join(img_folder, images_mini[i])
        image = pil_loader(img_path)
        image = transform_wo_norm(image)
        image = image.unsqueeze(0)
        image = image.cuda()
    
        org_score = model(norm(image))
        
        target_score = atttack_goal_mini[i]
        pert_image, pert_score, org_score = IFGSM_IQA_target(model, image, target_score, eps=eps, iteration=iter)
            
        save_name = images_mini[i][:-4] + '.bmp'
        save_path = os.path.join(save_dir, save_name)
        save(pert_image, save_path)
        org_score = org_score.detach().cpu().numpy()
        pert_score = pert_score.detach().cpu().numpy()
        pred_scores.append(pert_score)
        pred_scores_ori.append(org_score)

    pred_scores = np.array(pred_scores).squeeze()
    pred_scores_ori = np.array(pred_scores_ori).squeeze()
    moses = np.array(moses).squeeze()
    
    rho_s, _ = spearmanr(atttack_goal_mini, moses)
    rho_p, _ = pearsonr(atttack_goal_mini, moses)
    rho_k, _ = kendalltau(atttack_goal_mini, moses)
    rmse = np.sqrt(np.mean(np.power((atttack_goal_mini-moses),2)))
    mae = np.mean(np.abs(atttack_goal_mini-moses))
    print('Performance between MOS and target scores (SROCC/KROCC/PLCC/RMSE/MAE):\n{0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}'.format(rho_s,rho_k,rho_p,rmse,mae))
    
    rho_s, _ = spearmanr(pred_scores, moses)
    rho_p, _ = pearsonr(pred_scores, moses)
    rho_k, _ = kendalltau(pred_scores, moses)
    rmse = np.sqrt(np.mean(np.power((pred_scores-moses),2)))
    mae = np.mean(np.abs(pred_scores-moses))
    print('Performance between MOS and predicted score of adverarial examples (SROCC/KROCC/PLCC/RMSE/MAE):\n {0:.4f}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}'.format(rho_s,rho_k,rho_p,rmse,mae))
    

if __name__ == '__main__':
    main()