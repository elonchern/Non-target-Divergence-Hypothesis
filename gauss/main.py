import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from cmd import CMD
from crd import CRDLoss
from util import ConvReg
from distiller_zoo.FitNet import HintLoss
from distiller_zoo.RKD import RKDLoss
from distiller_zoo.PKT import PKT
from distiller_zoo.SP import Similarity
from distiller_zoo.DKD import dkd_loss

def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')

    # NCE distillation
    parser.add_argument('--feat_dim', default=8, type=int, help='feature dimension')
    parser.add_argument('--nce_k', default=2000, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--n_data', default=200, type=int, help='data number')
    parser.add_argument('--s_dim', default=2, type=int, help='student dim')
    parser.add_argument('--t_dim', default=2, type=int, help='teacher dim')
    opt = parser.parse_args()

    return opt



def pd_diff(pl,y_data,y_pred):
    
    max_value, max_index_t = torch.max(torch.softmax(pl, dim=1), dim=1)
    mask_tf = max_index_t - y_data
    mask_tf[mask_tf!=0] = True # Where the teacher's predictions were incorrect
    mask_tt = 1 - mask_tf
    
    max_index_s = torch.argmax(y_pred, dim=1)
    mask_sf = max_index_s - y_data
    mask_sf[mask_sf!=0] = True # Where the student's predictions were incorrect
    mask_st = 1 - mask_sf
     
    # Calculate cosine distance
    cosine_similarity = torch.cosine_similarity(pl,y_pred, dim=1)

    normalized_similarity = (cosine_similarity + 1) / 2
    
    cosine_mask_g = cosine_similarity.ge(0.996) # Keep values greater than 0.996, set values less than 0.996 to 0.
    
    mask = normalized_similarity*cosine_mask_g
    
    omega = mask_tf*mask + mask_tt
    
    return omega, mask_tf


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(LinearClassifier, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

def error_distribution(x1, x2, y2, model,teacher_model):
    model.eval()
    teacher_model.eval()
    with torch.no_grad():
        output = model(x2)
        pl = teacher_model(x1)
        
        _, mask = pd_diff(pl,y2,output)
        
        indices = torch.nonzero(y2 == 0).squeeze()
        # Select the mask for label=0.
        mask_0 = mask[indices]
        
        error_output = torch.softmax(output[indices][mask_0.bool()], dim=1)
        error_pl = torch.softmax(pl[indices][mask_0.bool()], dim=1)
        
        error_output_mean = torch.mean(error_output, dim=0)
        error_pl_mean = torch.mean(error_pl, dim=0)
        
    return error_output_mean,error_pl_mean
        
        
        
def evaluate(x, y, x_test, y_test, model):
    model.eval()
    with torch.no_grad():
        output = model(x)
        _, predicted = torch.max(output.data, 1)
        train_acc = (y == predicted).sum() / y.shape[0]

        output = model(x_test)
        _, predicted = torch.max(output.data, 1)
        test_acc = (y_test == predicted).sum() / y_test.shape[0]
    return train_acc.item(), test_acc.item()


def train(x, y, x_test, y_test, device, n_epoch=1000, eval_epoch=1, plot=False):
    model = LinearClassifier(input_dim=x.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    x, y, x_test, y_test, model = map(lambda x: x.to(device), (x, y, x_test, y_test, model))
    train_loss_curve = []
    train_acc_curve = []
    test_acc_curve = []
    for epoch in range(n_epoch):
        model.train()
        output = model(x)
        loss = criterion(output, y)
        train_loss_curve.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % eval_epoch == 0 or epoch == n_epoch - 1:
            train_acc, test_acc = evaluate(x, y, x_test, y_test, model)
            train_acc_curve.append(train_acc)
            test_acc_curve.append(test_acc)
            # print(f"epoch: {epoch}, loss: {loss:.3f} train acc {train_acc:.4f} test acc {test_acc:.4f}")

        if epoch == n_epoch - 1 and plot:
            plt.subplot(1, 2, 1)
            plt.plot(train_loss_curve)
            plt.subplot(1, 2, 2)
            plt.plot(train_acc_curve)
            plt.plot(test_acc_curve)
            plt.show()

    return model, test_acc

# cross-modal kd, teacher : x1 as input, student: x2 as input
def train_kd(x1, x2, y2, x2_test, y2_test, teacher_model, weight, device, mode, n_epoch, eval_epoch, plot,*args):
    opt = parse_option()
    teacher_model.eval()
    model = LinearClassifier(input_dim=x2.shape[1])
    criterion = torch.nn.CrossEntropyLoss()
    criterion_pl = torch.nn.KLDivLoss(reduction='batchmean')
    criterion_crd = CRDLoss(opt)
    criterion_hint = HintLoss()
    criterion_rkd = RKDLoss()
    criterion_pkt = PKT()
    criterion_sp = Similarity()
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    cmd = CMD()
    
    
    x1, x2, y2, x2_test, y2_test, teacher_model, model = map(lambda x: x.to(device),
                                                             (x1, x2, y2, x2_test, y2_test, teacher_model, model))
    train_loss_curve = []
    train_acc_curve = []
    test_acc_curve = []
   
    for epoch in range(n_epoch):
        model.train()
        
        pl = teacher_model(x1)
        output = model(x2)
        loss_gt = criterion(output, y2)
        omega,_ = pd_diff(pl,y2,output)
        
        if mode == 'KD_KL':
            loss_pl = criterion_pl(torch.log_softmax(output/0.7,dim=1), torch.softmax(pl/0.7, dim=1))
        elif mode == 'KD_KL_PSSM':
            loss_pl = criterion_pl(torch.log_softmax(output/0.7*omega.unsqueeze(1) ,dim=1), torch.softmax(pl/0.7*omega.unsqueeze(1), dim=1))
        elif mode == 'crd':
            index, sample_idx = map(lambda x: x.to(device),
                                    (args[0], args[1]))
            loss_pl = criterion_crd(output, pl, index, sample_idx) 
        elif mode == 'hint': # Fitnets: hints for thin deep nets
            regress_s = ConvReg(output.shape, pl.shape)
            f_s = regress_s(output)
            loss_pl = criterion_hint(f_s, pl)
            
        elif mode == 'rkd':
            loss_pl = criterion_rkd(output, pl)
            
        elif mode == 'pkt':
            loss_pl = criterion_pkt(output, pl) + criterion_pl(torch.log_softmax(output/0.7,dim=1), torch.softmax(pl/0.7, dim=1))
            
        elif mode == 'similarity':
            g_s = [output]
            g_t = [ pl]
            loss_group = criterion_sp(g_s, g_t)   
            loss_pl = sum(loss_group) + criterion_pl(torch.log_softmax(output/0.7,dim=1), torch.softmax(pl/0.7, dim=1)) 
            
        elif mode == 'dkd':
            omega = torch.ones_like(omega)
            loss_pl = dkd_loss(output, pl, y2, 1, 1,0.7, omega)
            
        elif mode == 'dkd_PSSM':
            loss_pl = dkd_loss(output, pl, y2, 1, 1,0.7,omega)                    
            
        else:
            raise NotImplementedError
        
        
        loss = weight[0] * loss_gt + weight[1] * loss_pl 
        train_loss_curve.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % eval_epoch == 0 or epoch == n_epoch - 1:
            
            train_acc, test_acc = evaluate(x2, y2, x2_test, y2_test, model)
            
            train_acc_curve.append(train_acc) 
            test_acc_curve.append(test_acc)
            
        if epoch == n_epoch -1:
            error_output_mean,error_pl_mean = error_distribution(x1, x2, y2, model,teacher_model)
            
            output_cmd = cmd(torch.softmax(output ,dim=1), torch.softmax(pl,dim=1),5)
            
        
        if epoch == n_epoch - 1 and plot:
            plt.subplot(1, 2, 1)
            plt.plot(train_loss_curve)
            plt.subplot(1, 2, 2)
            plt.plot(train_acc_curve)
            plt.plot(test_acc_curve)
            plt.show()
    
    return test_acc,error_output_mean,error_pl_mean,output_cmd


def gen_mm_data(a, n,  x1_dim=-1, x2_dim=-1, xs1_dim=-1, xs2_dim=-1, overlap_dim=-1, distill='kd'):
    """
    :param a: the separating hyperplane $\delta$ in Eq.(27) in the paper
    :param n: data number
    :param mode: a or b
    :param x1_dim: modality 1 feature dimension
    :param x2_dim: modality 2 feature dimension
    :param xs1_dim: modality 1 decisive feature dimension
    :param xs2_dim: modality 2 decisive feature dimension
    :param overlap_dim:
    :return:
    """
    opt = parse_option()
    xs = np.random.randn(n, xs1_dim + xs2_dim)     # decisive features
    a = a[0:xs1_dim + xs2_dim]                     # separating hyperplane
    y = (np.dot(xs, a) > 0).ravel()                # decisive features xs -> label y

    # x2, 0:xs2_dim-decisive features, other dim-gaussian noise
    x2 = np.random.randn(n, x2_dim)
    x2[:, 0:xs2_dim] = xs[:, 0:xs2_dim]

    # x1, among all x1_dim channels, xs1_dim channels are decisive and other are noise; among all xs1_dim decisive
    # channels, overlap_dim are shared between x1 and x2
    x1 = np.random.randn(n, x1_dim)
    x1[:, xs2_dim - overlap_dim:xs2_dim - overlap_dim + xs1_dim] = \
        xs[:, xs2_dim - overlap_dim:xs2_dim - overlap_dim + xs1_dim]


    if distill == 'crd':
        # 构造正负样本的index
        num_classes = 2
        label = np.array(y).astype(int)
        num_samples = len(x1)
        cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            cls_positive[label[i]].append(i)
            
        cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                cls_negative[i].extend(cls_positive[j])            
        cls_positive = [np.asarray(cls_positive[i]) for i in range(num_classes)]
        cls_negative = [np.asarray(cls_negative[i]) for i in range(num_classes)]
        cls_positive = np.asarray(cls_positive,dtype=object)
        cls_negative = np.asarray(cls_negative,dtype=object)
        
        idx = np.arange(num_samples)
        k = opt.nce_k
        
        sample_idx = [[] for i in range(num_samples)]
        for i in range(num_samples):
            replace = True if k > len(cls_negative[label[i]]) else False
            neg_idx = np.random.choice(cls_negative[label[i]], k, replace=replace)
            
            temp = np.hstack((np.asarray([idx[i]]), neg_idx))
           
            sample_idx[i].extend(temp)
            
        sample_idx = np.array(sample_idx)
        return torch.Tensor(x1), torch.Tensor(x2), torch.LongTensor(y), torch.LongTensor(idx),torch.LongTensor(sample_idx)
    
    elif  distill == 'kd':
          
        return torch.Tensor(x1), torch.Tensor(x2), torch.LongTensor(y)


def run_mm(seed_num, vari_dim, mode):
    seed(seed_num)
    
    n_train = 200
    n_test = 1000
    d = 500

    # ------------------------------ data generation ----------------------------- #
    a = np.random.randn(d)   # a random separating hyperplane

    if mode == 'crd':
        x1_train, x2_train, y_train, idx, sample_idx = gen_mm_data(a, n_train, x1_dim=25, x2_dim=25,
                                                    xs1_dim=10, xs2_dim=10, overlap_dim=vari_dim, distill=mode)
        seed(seed_num + 1)
        x1_test, x2_test, y_test, _, _ = gen_mm_data(a, n_test, x1_dim=25, x2_dim=25,
                                                xs1_dim=10, xs2_dim=10, overlap_dim=vari_dim, distill=mode)
    else:
    
        x1_train, x2_train, y_train = gen_mm_data(a, n_train, x1_dim=25, x2_dim=25,
                                                    xs1_dim=10, xs2_dim=10, overlap_dim=vari_dim, distill='kd')
        seed(seed_num + 1)
        x1_test, x2_test, y_test = gen_mm_data(a, n_test, x1_dim=25, x2_dim=25,
                                                xs1_dim=10, xs2_dim=10, overlap_dim=vari_dim, distill='kd')
    # ------------------------------------------------------------------------------ #
    cmd = CMD()
    cmd_distance = cmd(x1_train,x2_train,5)
   
             
    # train a unimodal teacher model that takes input from modality 1
    teacher_model, teacher_acc = train(x1_train, y_train, x1_test, y_test, device=device, n_epoch=1000)
    
    # cross-modal KD: the x1 teacher distills knowledge to a x2 student
    if mode == 'crd':
        kd_student_acc,error_output_mean,error_pl_mean,output_cmd = train_kd(x1_train, x2_train, y_train, x2_test, y_test, teacher_model, 
                                                                            [1.0, 0.02], device, mode,1000, 1, False, idx, sample_idx)
    else:
        kd_student_acc,error_output_mean,error_pl_mean,output_cmd = train_kd(x1_train, x2_train, y_train, x2_test, y_test, teacher_model, 
                                                                            [1.0, 1.0], device, mode,1000, 1, False)
    
    # baseline: train a model from modality 2 without KD
    _, no_kd_baseline_acc = train(x2_train, y_train, x2_test, y_test, device=device, n_epoch=1000)
    return teacher_acc, no_kd_baseline_acc, kd_student_acc,cmd_distance,error_output_mean,error_pl_mean,output_cmd


def exp(seed, n_runs, mode):
    
    error_t = []
    error_s = []
    overlap_dim_list = [0, 2, 4, 6, 8, 10]
    for overlap_dim in overlap_dim_list:
        gamma = overlap_dim / (20 - overlap_dim)
        acc_np = np.zeros((n_runs, 5))
        for i in range(n_runs):
            teacher_acc, no_kd_baseline_acc, kd_student_acc,cmd_distance,error_output_mean,error_pl_mean, output_cmd= run_mm(seed + i, overlap_dim, mode)
            
            acc_np[i, 0:1] = teacher_acc, 
            acc_np[i, 1:2] = no_kd_baseline_acc, 
            acc_np[i, 2:3] = kd_student_acc,
            acc_np[i, 3:4] = cmd_distance,
            acc_np[i, 4:5] = output_cmd,
            error_s.append(error_output_mean)
            error_t.append(error_pl_mean)
        error_stack_s = torch.stack(error_s, dim=0) 
        mean_s = torch.mean(error_stack_s, dim=0)
        error_stack_t = torch.stack(error_t, dim=0) 
        mean_t = torch.mean(error_stack_t, dim=0)
        
        print("mean_s = {}".format(mean_s))
        print("mean_t = {}".format(mean_t))
           
        delta = np.round((acc_np[:, 2] - acc_np[:, 1]) * 100, 2)
        log_mean = np.mean(acc_np, axis=0) 
        print(f'gamma = {gamma:.2f}')
        print(f'Teacher acc {log_mean[0]:.4f}')
        print(f'No KD acc {log_mean[1]:.4f}')
        print(f'KD student acc {log_mean[2]:.4f}')
        print(f'cmd distance {log_mean[3]:.2f}')
        print(f'output cmd distance {log_mean[4]:.2f}')
        print(f'Delta: {np.mean(delta):.2f} ± {np.std(delta):.2f}')
        print('-' * 60)


if __name__ == '__main__':
    # x_1 and x_2 here correspond to x^a and x^b in the main paper, respectively.
    device = torch.device("cpu")
    print('Exp 1: KD with KL')
    exp(seed=0, n_runs=100, mode='KD_KL')   # Figure 3 and 4 in the paper
    print('-' * 80)
    
    # print('Exp 2: KD with KL_PSSM')
    # exp(seed=0, n_runs=100, mode='KD_KL_PSSM')   # Table 1 in the paper
    # print('-' * 80)
    
    # print('Exp 3: KD with CRD')
    # exp(seed=0, n_runs=100, mode='crd')   # Table 1 in the paper
    # print('-' * 80)
    
    # print('Exp 4: KD with FitNet')
    # exp(seed=0, n_runs=100, mode='hint')   # Table 1 in the paper
    # print('-' * 80)    
    
    # print('Exp 5: KD with RKD')
    # exp(seed=0, n_runs=100, mode='rkd')   # Table 1 in the paper
    # print('-' * 80)    
    
    # print('Exp 6: KD with PKD') 
    # exp(seed=0, n_runs=100, mode='pkt')   # Table 1 in the paper
    # print('-' * 80)    

    # print('Exp 7: KD with SP') # similarity 
    # exp(seed=0, n_runs=100, mode='similarity')   # Table 1 in the paper
    # print('-' * 80)     

    # print('Exp 8: KD with DKD') # Decoupled KD
    # exp(seed=0, n_runs=100, mode='dkd')   # Table 1 in the paper
    # print('-' * 80)          
    
    # print('Exp 9: KD with DKD_PSSM') # Decoupled KD with PSSM
    # exp(seed=0, n_runs=100, mode='dkd_PSSM')   # Table 1 in the paper
    # print('-' * 80)          
