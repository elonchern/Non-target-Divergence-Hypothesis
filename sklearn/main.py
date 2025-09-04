import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from network.two_layer import TwoLayerNN
from dataloader.dataset import dataset
from distances.kl import kl_distance
from utils.functions import pd_diff
from distances.cmd import CMD
from copy import deepcopy
from distiller_zoo.CRD import CRDLoss
from utils.functions import ConvReg
from distiller_zoo.FitNet import HintLoss
from distiller_zoo.RKD import RKDLoss
from distiller_zoo.PKT import PKT
from distiller_zoo.SP import Similarity
from distances.jsd import js_divergence
from distiller_zoo.JS import JSDivergence
from distiller_zoo.DKD import dkd_loss

def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')
    
    # algorithm 1
    parser.add_argument('--mode', type=str, default='random', help='remove idx mode: alg1, random and reverse')
    parser.add_argument('--mask_dim', type=int, default=2, help='The effective dimensions of teacher network input data after masking')

    # NCE distillation
    parser.add_argument('--feat_dim', default=8, type=int, help='feature dimension')
    parser.add_argument('--nce_k', default=3000, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--n_data', default=6000, type=int, help='data number')
    parser.add_argument('--s_dim', default=4, type=int, help='student dim')
    parser.add_argument('--t_dim', default=4, type=int, help='teacher dim')
    opt = parser.parse_args()

    return opt



def error_distribution(x1, x2, y2, model,teacher_model):
    model.eval()
    teacher_model.eval()
    with torch.no_grad():
        output = model(x2,0)
        pl = teacher_model(x1,0)
        
        _, mask, mask_target, mask_other = pd_diff(pl,y2,output)
        
        indices = torch.nonzero(y2 == 0).squeeze()
        # 选择label=0的mask
        mask_0 = mask[indices]
        
        error_output = torch.softmax(output[indices][mask_0.bool()], dim=1)
        error_pl = torch.softmax(pl[indices][mask_0.bool()], dim=1)
        
        error_output_mean = torch.mean(error_output, dim=0)
        error_pl_mean = torch.mean(error_pl, dim=0)
        
    return error_output_mean,error_pl_mean

def mask(x, index):
    y = deepcopy(x)
    
    for i in index:
        y[:, i] = 0
        
    return y

def  my_shuffle(x, index, manner='in_row'):
    y = deepcopy(x)
    if manner == 'in_row':
        perm_index = torch.randperm(x.shape[1])
        for i in range(y.shape[0]):
            if i in index:
                y[i, :] = x[i, perm_index]
    elif manner == 'in_col':
        perm_index = torch.randperm(x.shape[0])
        for i in index:
            # for i in range(y.shape[1]):
            #     if i in index:
            y[:, i] = x[perm_index, i]
    return y




def train_for_overlap_tag(x1_train, x2_train, y_train, num_epoch=1000, plot=False):
    
    # teacher_model.eval()
    teacher_model = TwoLayerNN(50, 16, 0, 42)
    model = TwoLayerNN(50, 16, 0, 42)
    criterion = torch.nn.CrossEntropyLoss()
    criterion2 = JSDivergence(temperature=1.0)
    # criterion2 = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(list(teacher_model.parameters()) + list(model.parameters()), lr=0.01, momentum=0.9)
    loss_curve = np.zeros((num_epoch, 3), dtype=float)
    for cur_epoch in range(num_epoch):
        output_t = teacher_model(x1_train)  # teacher takes x1 as input
        output_s = model(x2_train)  # student takes x2 as input
        # loss = criterion(outputs1, train_y) + criterion(outputs2, train_y) + criterion2(outputs1, outputs2)
        tmp1 = criterion(output_t, y_train)
        tmp2 = criterion(output_s, y_train)
        tmp3 = criterion2(output_s, output_t, y_train)
        # tmp3 = criterion2(output_s, output_t)
        loss = tmp1 + tmp2 + tmp3
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_curve[cur_epoch] = tmp1.item(), tmp2.item(), tmp3.item()
        # if cur_epoch == num_epoch - 1:
        #     print(f"Epoch {cur_epoch}: train loss={tmp1:.3f},{tmp2:.3f},{tmp3:.3f}")

    if plot:
        plt.plot(loss_curve)
        plt.legend(['teacher ce loss', 'student ce loss', 'distance'])
        plt.show()

    # t_acc = eval_acc(teacher_model, x1_train, x2_train, y_train, 'eval_teacher') * 100
    # s_acc = eval_acc(model, x1_train, x2_train, y_train, 'eval_student') * 100
    # print(f'Training data, t acc {t_acc:.2f} | s acc {s_acc:.2f}')

    return teacher_model, model


def cal_overlap_tag(teacher_model, student_model, x1_train, x2_train, y_train, true_overlap_dim, mode, permu_repeat_num=10, plot=False):
    teacher_model.eval()
    student_model.eval()
    mse_loss = torch.nn.MSELoss()
    overlap_tag_for_x1, overlap_tag_for_x2 = np.zeros((permu_repeat_num, x1_train.shape[1])), np.zeros(
        (permu_repeat_num, x2_train.shape[1])) # [100,50]

    for j in range(permu_repeat_num):
        # fix x1, permute x2
        h1 = teacher_model(x1_train)
        for index in range(overlap_tag_for_x2.shape[1]):
            x2_train_permu = my_shuffle(x2_train, [index], manner='in_col')
            # x2_train_permu = mask(x2_train, [index])
            h2 = student_model(x2_train_permu)
            tcjsd, ncjsd = js_divergence(h1,h2,y_train)
            overlap_tag_for_x2[j, index] = ncjsd
            # overlap_tag_for_x2[j, index] = mse_loss(h1, h2)

        # fix x2, permute x1
        h2 = student_model(x2_train)
        for index in range(overlap_tag_for_x1.shape[1]):
            x1_train_permu = my_shuffle(x1_train, [index], manner='in_col')
            # x1_train_permu = mask(x1_train, [index])
            h1 = teacher_model(x1_train_permu)
            tcjsd, ncjsd = js_divergence(h1,h2,y_train)
            overlap_tag_for_x1[j, index] = ncjsd
            # overlap_tag_for_x1[j, index] = mse_loss(h1, h2)

    # linear normalization max-> 1, min -> 0
    overlap_tag_for_x1_mean = overlap_tag_for_x1.mean(axis=0)
    overlap_tag_for_x1_mean = (overlap_tag_for_x1_mean - np.min(overlap_tag_for_x1_mean)) / (
            np.max(overlap_tag_for_x1_mean) - np.min(overlap_tag_for_x1_mean))

    if mode == 'random':
        x1_overlap_idx = np.random.randint(0, 50, true_overlap_dim)
        x2_overlap_idx = np.random.randint(0, 50, true_overlap_dim)
    elif mode == 'alg1':
        
        x1_overlap_idx = (-overlap_tag_for_x1_mean).argsort()[:true_overlap_dim]
        x1_correct = np.intersect1d(x1_overlap_idx, np.arange(true_overlap_dim))
        
        overlap_tag_for_x2_mean = overlap_tag_for_x2.mean(axis=0)
        overlap_tag_for_x2_mean = (overlap_tag_for_x2_mean - np.min(overlap_tag_for_x2_mean)) / (
                np.max(overlap_tag_for_x2_mean) - np.min(overlap_tag_for_x2_mean))

        x2_overlap_idx = (-overlap_tag_for_x2_mean).argsort()[:true_overlap_dim]
        x2_correct = np.intersect1d(x2_overlap_idx, np.arange(true_overlap_dim))
       
    elif mode == 'reverse':
        x1_overlap_idx = (overlap_tag_for_x1_mean).argsort()[:true_overlap_dim]
        x1_correct = np.intersect1d(x1_overlap_idx, np.arange(true_overlap_dim))
        
        overlap_tag_for_x2_mean = overlap_tag_for_x2.mean(axis=0)
        overlap_tag_for_x2_mean = (overlap_tag_for_x2_mean - np.min(overlap_tag_for_x2_mean)) / (
                np.max(overlap_tag_for_x2_mean) - np.min(overlap_tag_for_x2_mean))

        x2_overlap_idx = (overlap_tag_for_x2_mean).argsort()[:true_overlap_dim]
        x2_correct = np.intersect1d(x2_overlap_idx, np.arange(true_overlap_dim))
        
    else:
        raise NotImplementedError  

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(np.arange(1, x1_train.shape[1] + 1), overlap_tag_for_x1_mean)
        plt.title("Overlap tag for x1")
        plt.subplot(1, 2, 2)
        plt.scatter(np.arange(1, x2_train.shape[1] + 1), overlap_tag_for_x2_mean)
        plt.title("Overlap tag for x2")
        plt.show()

    return x1_overlap_idx, x2_overlap_idx



def train(seed, mod, x_data, y_data, x_test, y_test, hidden_dim, epochs, eval_epochs, alpha, beta, dp, save_model=False, plot=True):
    model = TwoLayerNN(mod,hidden_dim, dp, seed).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    val_acc = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        x = x_data + alpha * abs(x_data).mean() * torch.randn(x_data.shape)
        x = x.to(device)
        y_data = y_data.to(device)
        y_pred = model(x, beta)
        
        # Compute Loss
        loss = criterion(y_pred, y_data)
        # Backward pass
        loss.backward()
        optimizer.step()
    
        model.eval()
        if epoch % eval_epochs == eval_epochs-1:
            with torch.no_grad():
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                outputs = model(x_test, 0)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == y_test).squeeze().sum()
            acc = c.item() /len(y_test) * 100
            val_acc.append(acc)
            
        if plot and (epoch % epochs == epochs-1):
            plt.plot(val_acc)
            plt.xlabel('epoch')
            plt.ylabel('test acc')
            print('Epoch: %d \t loss: %.2f \t test acc: %.2f' %(epoch, loss, acc))
    
    if save_model:
        return model, acc
    else:
        return acc
    
def train_kd(seed, mod, teacher_model, x1_data, x2_data, y_data, x2_test, y_test, hidden_dim, epochs,
             eval_epochs, alpha, beta, dp, weight, distill, save_model, plot,*args):
    opt = parse_option()
    teacher_model.eval()
    model = TwoLayerNN(mod, hidden_dim, dp, seed).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion_pl = torch.nn.KLDivLoss(reduction='batchmean').to(device)
    criterion_crd = CRDLoss(opt).to(device)
    criterion_hint = HintLoss().to(device)
    criterion_rkd = RKDLoss().to(device)
    criterion_pkt = PKT().to(device)
    criterion_sp = Similarity().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    val_acc = []
    cmd = CMD()
    
    for epoch in range(epochs):
        model.train()
        teacher_model.eval()
        optimizer.zero_grad()
        # Forward pass
        x1 = x1_data + alpha * abs(x1_data).mean() * torch.randn(x1_data.shape)
        x2 = x2_data + alpha * abs(x2_data).mean() * torch.randn(x2_data.shape)
        x1 = x1.to(device)
        x2 = x2.to(device)
        y_data = y_data.to(device)
        
        pl = teacher_model(x1, beta)
        y_pred = model(x2, beta)
        # Compute Loss
        loss_gt = criterion(y_pred, y_data)
         
        # calculate mask- probability distribution differences
        omega,mask_tf, mask_target, mask_other = pd_diff(pl,y_data,y_pred)

        if distill == 'kl':
            # criterion_pl(torch.log_softmax(y_pred,dim=1), torch.softmax(pl, dim=1))
            loss_pl = criterion_pl(torch.log_softmax(y_pred,dim=1), torch.softmax(pl, dim=1))
            
        elif distill == 'kl_PSSM':
            loss_pl = criterion_pl(torch.log_softmax(y_pred*omega.unsqueeze(1) ,dim=1), torch.softmax(pl*omega.unsqueeze(1), dim=1))
            
        elif distill == 'crd':
            
            index, sample_idx = map(lambda x: x.to(device),
                                    (args[0], args[1]))
            loss_pl = criterion_crd(y_pred, pl, index, sample_idx) + criterion_pl(torch.log_softmax(y_pred,dim=1), torch.softmax(pl, dim=1))
            
        elif distill == 'hint': # Fitnets: hints for thin deep nets
            regress_s = ConvReg(y_pred.shape, pl.shape).to(device)
            f_s = regress_s(y_pred)
            loss_pl = 0.1*criterion_hint(f_s, pl) + criterion_pl(torch.log_softmax(y_pred,dim=1), torch.softmax(pl, dim=1))
            
        elif distill == 'rkd':
            loss_pl = criterion_rkd(y_pred, pl) + 0.1*criterion_pl(torch.log_softmax(y_pred,dim=1), torch.softmax(pl, dim=1))
            
        elif distill == 'pkt':
            loss_pl = criterion_pkt(y_pred, pl) + criterion_pl(torch.log_softmax(y_pred,dim=1), torch.softmax(pl, dim=1))
            
        elif distill == 'similarity':
            g_s = [y_pred]
            g_t = [ pl]
            loss_group = criterion_sp(g_s, g_t)   
            loss_pl = sum(loss_group) + criterion_pl(torch.log_softmax(y_pred,dim=1), torch.softmax(pl, dim=1)) 
            
        elif distill == 'dkd':
            omega = torch.ones_like(omega)
            loss_pl = dkd_loss(y_pred, pl, y_data, 16, 1, 0.7, omega)
            
        elif distill == 'dkd_PSSM':
            loss_pl = dkd_loss(y_pred, pl, y_data, 16, 1, 0.7, omega)                  
            
        else:
            raise NotImplementedError


    
        # loss_pl = criterion_pl(torch.log_softmax(y_pred,dim=1), torch.softmax(pl, dim=1))
        # loss_pl = dkd_loss(y_pred, pl, y_data, 100, 1,0.7,omega)
       
        loss = weight[0] * loss_gt + weight[1] * loss_pl
        # Backward pass
        loss.backward()
        optimizer.step()
    
        model.eval()
        if epoch % eval_epochs == eval_epochs-1:
            with torch.no_grad():
                x2_test = x2_test.to(device)
                y_test = y_test.to(device)
                outputs = model(x2_test, 0)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == y_test).squeeze().sum()
            acc = c.item() /len(y_test) * 100
            val_acc.append(acc)
         
        if epoch == epochs -1:
            omega,mask_tf, mask_target, mask_other = pd_diff(pl,y_data,y_pred)
            error_output_mean,error_pl_mean = error_distribution(x1, x2, y_data, model,teacher_model)         
            # output_cmd = cmd(torch.softmax(y_pred[mask_tf.bool()] ,dim=1), torch.softmax(pl[mask_tf.bool()],dim=1),5)
            output_cmd = cmd(mask_target, mask_other,5)  
            tcjsd, ncjsd = js_divergence(pl,y_pred,y_data)
            
        if plot and (epoch % epochs == epochs-1):
            plt.plot(val_acc)
            plt.xlabel('epoch')
            plt.ylabel('test acc')
            print('Epoch: %d \t loss: %.2f \t test acc: %.2f' %(epoch, loss, acc))
   
    if save_model:
        return model,acc
    else:
        return acc,error_output_mean,error_pl_mean,ncjsd.cpu().detach().numpy()
    
    
def run_exp(seed_num, vari_dim, fixed_dim,n_runs,distill):
    opt = parse_option()
    np.random.seed(42)
    if distill == 'crd':
        X_train, X_t, y_train, y_t, idx, sample_idx = dataset(n_samples=10000, n_classes=4, n_features=50, n_informative=50, n_redundant=0, n_clusters_per_class=1,
                                            random_state=42,distill=distill,opt=opt)
    else:
        X_train, X_t, y_train, y_t = dataset(n_samples=10000, n_classes=4, n_features=50, n_informative=50, n_redundant=0, n_clusters_per_class=1,
                                            random_state=42,distill=distill,opt=opt)
        
    
    x2_train = np.random.randn(6000, 50)
    x2_train[:, 0:fixed_dim] = X_train[:, 0:fixed_dim]
    
    x1_train = np.random.randn(6000, 50)
    x1_train[:, fixed_dim-vari_dim:fixed_dim*2-vari_dim] = X_train[:, fixed_dim-vari_dim:fixed_dim*2-vari_dim] 
    
    x1_train = torch.FloatTensor(x1_train).view(-1,50)
    x2_train = torch.FloatTensor(x2_train).view(-1,50)
    
    y_train = torch.LongTensor(y_train)


    x2_test = np.random.randn(4000, 50)
    x2_test[:, 0:fixed_dim] = X_t[:, 0:fixed_dim]
    
    x1_test = np.random.randn(4000, 50)
    x1_test[:, fixed_dim-vari_dim:fixed_dim*2-vari_dim] = X_t[:, fixed_dim-vari_dim:fixed_dim*2-vari_dim] 
    
    x1_test = torch.FloatTensor(x1_test).view(-1,50)
    x2_test = torch.FloatTensor(x2_test).view(-1,50)

    y_test = torch.LongTensor(y_t)
    
    cmd = CMD()
    
    log1 = []
    log2 = []
    log3 = []
    log4 = []
    log5 = []
    log6 = []
    log7 = []
    log8 = []
    log9 = []
    epochs = 1000

    # three kinds of regularization, alpha / beta / dropout rate
    alpha = 0
    beta = 0
    rate = 0
    weight = [1.0,1.0]

    random.seed(seed_num)
   
    for j in range(n_runs):  
        seed = random.randint(0, 100)

        teacher_model, teacher_acc = train(seed, 50, x1_train, y_train, x1_test, y_test, 16, 
                                           epochs, epochs, alpha, beta, rate, save_model=True,plot=False )
        if distill == 'crd':
            kd_student_acc,error_output_mean,error_pl_mean,output_cmd = train_kd(seed, 50, teacher_model, x1_train, x2_train, y_train, x2_test, y_test,16, 
                                    epochs, epochs, alpha, beta, rate, weight,distill,False, False, idx, sample_idx)
        else:
            kd_student_acc,error_output_mean,error_pl_mean,output_cmd = train_kd(seed, 50, teacher_model, x1_train, x2_train, y_train, x2_test, y_test,16, 
                                    epochs, epochs, alpha, beta, rate, weight,distill,False,False)
            

        nokd_student_acc = train(seed, 50, x2_train, y_train, x2_test, y_test, 16, epochs, epochs, alpha, beta, rate,plot=False)

        
        model_t, model_s = train_for_overlap_tag(x1_train, x2_train, y_train,  plot=False)
        x1_overlap_idx, x2_overlap_idx = cal_overlap_tag(model_t, model_s, x1_train, x2_train, y_train, opt.mask_dim, opt.mode)
        
        
        
        x1_train_new = x1_train[:, x1_overlap_idx]
        x1_test_new = x1_test[:, x1_overlap_idx]
            
        model_t1, mask_t_acc = train(seed, opt.mask_dim, x1_train_new, y_train, x1_test_new, y_test, 16, 
                                           epochs, epochs, alpha, beta, rate, save_model=True,plot=False )
        mask_s_acc,_,_,_ = train_kd(seed, 50, model_t1, x1_train_new, x2_train, y_train, x2_test, y_test, 16, 
                                    epochs, epochs, alpha, beta, rate, weight,distill,False,False)
        
        
#         print(f'{j}th run | teacher acc {teacher_acc} | kd_student acc {kd_student_acc} | nokd_student acc {nokd_student_acc}')
        log1.append(teacher_acc)
        log2.append(kd_student_acc)
        log3.append(nokd_student_acc)
        log5.append(error_output_mean)
        log6.append(error_pl_mean)
        log7.append(output_cmd)
        log8.append(mask_t_acc)
        log9.append(mask_s_acc)
        
    
    # compute CMD Distance    
    for j in range(n_runs):  
        seed = random.randint(0, 100)
          
        X_train, X_t, y_train, y_t = dataset(n_samples=10000, n_classes=4, n_features=50, n_informative=50, n_redundant=0, n_clusters_per_class=1,random_state=seed)
        
        x2_train = np.random.randn(6000, 50)
        x2_train[:, 0:fixed_dim] = X_train[:, 0:fixed_dim]
        
        x1_train = np.random.randn(6000, 50)
        x1_train[:, fixed_dim-vari_dim:fixed_dim*2-vari_dim] = X_train[:, fixed_dim-vari_dim:fixed_dim*2-vari_dim] 
        
        x1_train = torch.FloatTensor(x1_train).view(-1,50)
        x2_train = torch.FloatTensor(x2_train).view(-1,50)
        
        cmd_distance = cmd(x1_train,x2_train,5)
        
        
        log4.append(cmd_distance)

    return log1,log2,log3,log4,log5,log6,log7,log8,log9

def exp(seed, n_runs,distill):
    overlap_dim_list =  [20,15,10,5,0]
    for overlap_dim in overlap_dim_list:
        
        gamma = overlap_dim/max(overlap_dim_list)
        log1, log2, log3, log4,log5,log6,log7,log8,log9 = run_exp(seed_num=seed, vari_dim=overlap_dim, fixed_dim=20,n_runs=n_runs,distill=distill)
        delta = np.round((np.array(log2)-np.array(log3))*1,2)
        
        error_stack_s = torch.stack(log5, dim=0) 
        mean_s = torch.mean(error_stack_s, dim=0)
        error_stack_t = torch.stack(log6, dim=0) 
        mean_t = torch.mean(error_stack_t, dim=0) 
        
        delta_mt = np.round((np.array(log8)-np.array(log1))*1,2)
        delta_ms = np.round((np.array(log9)-np.array(log3))*1,2)
          
        
        
        print( )
        # print("mean_s = {}".format(mean_s))
        # print("mean_t = {}".format(mean_t))
        print(f'gamma = {gamma:.2f}')
        print(f'Results over {n_runs} runs:')
        print(f'teacher {np.mean(log1)}')  # unimodal baseline
        print(f'student with KD {np.mean(log2)}')  
        print(f'student without KD (Baseline) {np.mean(log3)}')  # upper bound of our multimodal student 
        print(f'cmd distance {np.mean(log4)}')
        print(f'output cmd distance {np.mean(log7)}')
        print(f'Delta: {np.mean(delta):.2f} ± {np.std(delta):.2f}')
        print(f'teacher mask {np.mean(log8)}')
        print(f'student mask {np.mean(log9)}')
        print(f'Delta_mt: {np.mean(delta_mt):.2f} ± {np.std(delta_mt):.2f}')
        print(f'Delta_ms: {np.mean(delta_ms):.2f} ± {np.std(delta_ms):.2f}')
              
        print('-' * 60)
        
if __name__ == '__main__':
    device  = torch.device('cuda:0')
    
    print("Exp 1: Modality Specifi with increase gamma on KL")
    exp(seed=42, n_runs = 2,distill='kl')
    print('-' * 80)
    
    # print("Exp 2: Modality Specifi with increase gamma on kl_PSSM")
    # exp(seed=42, n_runs = 10,distill='kl_PSSM')
    # print('-' * 80)    
    
    
    # print('Exp 3: KD with CRD')
    # exp(seed=42, n_runs=10, distill='crd')   # Table 1 in the paper
    # print('-' * 80)
    
    # print('Exp 4: KD with FitNet')
    # exp(seed=42, n_runs=10, distill ='hint')   # Table 1 in the paper
    # print('-' * 80)    
    
    # print('Exp 5: KD with RKD')
    # exp(seed=42, n_runs=10, distill='rkd')   # Table 1 in the paper
    # print('-' * 80)    
    
    # print('Exp 6: KD with PKD') 
    # exp(seed=42, n_runs=10, distill='pkt')   # Table 1 in the paper
    # print('-' * 80)    

    # print('Exp 7: KD with SP') # similarity 
    # exp(seed=42, n_runs=10, distill='similarity')   # Table 1 in the paper
    # print('-' * 80)     

    # print('Exp 8: KD with DKD') # Decoupled KD
    # exp(seed=42, n_runs=10, distill='dkd')   # Table 1 in the paper
    # print('-' * 80)          
    
    # print('Exp 9: KD with DKD_PSSM') # Decoupled KD with PSSM
    # exp(seed=42, n_runs=10, distill='dkd_PSSM')   # Table 1 in the paper
    # print('-' * 80)          
        