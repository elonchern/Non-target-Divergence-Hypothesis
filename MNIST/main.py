import os
import sys
import torch
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from utils.helper import gen_data, train_network_distill, cal_overlap_tag, evaluate_allacc, train_network_baseline
# from utils.model import ImageNet, AudioNet
from network.LR_3 import MLP 
from utils.helper import evaluate
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from utils.helper import generate_heatmap


def heat_map(loader, device, args):
    stu_type = args.stu_type
    tea_type = 1 - stu_type
    # load teacher model
    tea_model = MLP(input_dim=28*28*3).to(device) if tea_type == 0 else MLP(input_dim=28*28).to(device)
    tea_model.load_state_dict(torch.load('/home/elon/Workshops/MSFD/MNIST/results/teacher_mod_' + str(tea_type) + '_overlap.pkl', map_location={"cuda:0": "cpu"}))
    print(f'Finish Loading teacher model')
    train_acc, val_acc, test_acc = evaluate_allacc(loader, device, tea_model, tea_type)
    print(f'Teacher train | val | test acc {train_acc:.2f} | {val_acc:.2f} | {test_acc:.2f}')

    place_t = args.place
    tea_dim = tea_model.get_feature_dim(place_t)

    if args.mode >= 0:
        if args.mode == 0:
            remove_idx = random.sample(range(tea_dim), int(args.ratio * tea_dim))
            print('Randomly remove idx')

        else:
            overlap_tag_for_tea_mean = np.load('/home/elon/Workshops/MSFD/MNIST/results/overlap_tag_teacher_place' + str(place_t) + '_repeat' + str(args.num_permute) + '.npy')
            sort_idx = (overlap_tag_for_tea_mean).argsort() if args.mode == 1 else (-overlap_tag_for_tea_mean).argsort()
            remove_idx = sort_idx[0: int(args.ratio * tea_dim)]
            print(f'Loading overlap tag')
            print('remove elements', overlap_tag_for_tea_mean[remove_idx[0:3]], overlap_tag_for_tea_mean[remove_idx[-3:-1]])
        print(f'tea dim {tea_dim}, remove dim {len(remove_idx)}')
        change_info_tea = ['freeze', place_t, remove_idx]

    else:
        change_info_tea = [None, None, None]  # baseline

    generate_heatmap(tea_model, loader['test'], device, change_info_tea,args.mode)
    
def train_baseline(loader, device, args):
    stu_type = args.stu_type
    tea_type = 1 - stu_type
    log_np = np.zeros((args.num_runs, 2))
    for run in range(args.num_runs):
        print(f'Run {run}')
        net = MLP(input_dim=28*28*3).to(device) if stu_type == 0 else MLP(input_dim=28*28).to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        val_best_acc, test_best_acc,model_best = train_network_baseline(stu_type, args.num_epochs, loader, net, device, optimizer, [None] * 3)
        log_np[run, 0:2] = val_best_acc, test_best_acc
    log_mean = np.mean(log_np, axis=0)
    log_std = np.std(log_np, axis=0)
    print(f'Finish {args.num_runs} runs')
    print(f'Student Val Acc {log_mean[0]:.3f} ± {log_std[0]:.3f} | Test Acc {log_mean[1]:.3f} ± {log_std[1]:.3f}')



def eval_overlap_tag(loader, device, args):
    stu_type = args.stu_type
    tea_type = 1 - stu_type
    # load teacher model
    tea_model = MLP(input_dim=28*28*3).to(device) if tea_type == 0 else MLP(input_dim=28*28).to(device)
    tea_model.load_state_dict(torch.load('/home/elon/Workshops/MSFD/MNIST/results/teacher_mod_' + str(tea_type) + '_overlap.pkl', map_location={"cuda:0": "cpu"}))
    print(f'Finish Loading teacher model')
    train_acc, val_acc, test_acc = evaluate_allacc(loader, device, tea_model, tea_type)
    print(f'Teacher train | val | test acc {train_acc:.2f} | {val_acc:.2f} | {test_acc:.2f}')

    place_t = args.place
    tea_dim = tea_model.get_feature_dim(place_t)

    if args.mode >= 0:
        if args.mode == 0:
            remove_idx = random.sample(range(tea_dim), int(args.ratio * tea_dim))
            print('Randomly remove idx')

        else:
            overlap_tag_for_tea_mean = np.load('/home/elon/Workshops/MSFD/MNIST/results/overlap_tag_teacher_place' + str(place_t) + '_repeat' + str(args.num_permute) + '.npy')
            sort_idx = (overlap_tag_for_tea_mean).argsort() if args.mode == 1 else (-overlap_tag_for_tea_mean).argsort()
            remove_idx = sort_idx[0: int(args.ratio * tea_dim)]
            print(f'Loading overlap tag')
            print('remove elements', overlap_tag_for_tea_mean[remove_idx[0:3]], overlap_tag_for_tea_mean[remove_idx[-3:-1]])
        print(f'tea dim {tea_dim}, remove dim {len(remove_idx)}')
        change_info_tea = ['freeze', place_t, remove_idx]

    else:
        change_info_tea = [None, None, None]  # baseline

    train_acc, val_acc, test_acc = evaluate_allacc(loader, device, tea_model, tea_type, change_info_tea)


    print(f'Freeze {args.ratio * 100}% dimension')
    print(f'After modifying: teacher train {train_acc:.2f}')

    log_np = np.zeros((args.num_runs, 2))
    for run in range(args.num_runs):
        print(f'Run {run}')
        net = MLP(input_dim=28*28*3).to(device) if stu_type == 0 else MLP(input_dim=28*28).to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        log_np[run, :] = train_network_distill(stu_type, tea_model, args.num_epochs, loader, net, [args.gt_weight, args.pl_weight],
                                               device, optimizer, [None] * 3, change_info_tea, args.distill)
    log_mean = np.mean(log_np, axis=0)
    log_std = np.std(log_np, axis=0)
    print(f'Finish {args.num_runs} runs')
    print(f'Student Val Acc {log_mean[0]:.3f} ± {log_std[0]:.3f} | Test Acc {log_mean[1]:.3f} ± {log_std[1]:.3f}')



def train_network(stu_type, epochs, loader, net, device, optimizer):
    val_best_acc, test_best_acc = 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        net.train()
        # train_loss = 0.0
        loss1, loss2 = 0, 0
        for i, data in enumerate(loader['train']):
            mnist, mnist_m, labels = data['MNIST'][:,0,:,:], data['MNIST_M'], data['label']
            mnist, mnist_m, labels = mnist.view(-1,28*28).to(device), mnist_m.view(-1,28*28*3).to(device), labels.long().to(device)

            if stu_type == 0:
                outputs = net(mnist_m)
                
            elif stu_type == 1:
                outputs = net(mnist)
                
            else:
                raise ValueError("Undefined training type in distilled training")
            optimizer.zero_grad()
            loss =  criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        train_loss, train_acc = evaluate(loader['train'], device, net, stu_type)
        val_loss, val_acc = evaluate(loader['val'], device, net, stu_type)
        test_loss, test_acc = evaluate(loader['test'], device, net, stu_type)

        if val_acc > val_best_acc:
            val_best_acc = val_acc
            test_best_acc = test_acc
            
        print(f"Epoch | All epochs: {epoch} | {epochs}")
        print(f"Train Loss: {train_loss:.3f} | GT Loss {loss1 / len(loader['train']):.3f} | PL Loss {loss2 / len(loader['train']):.3f}")
        print(f"Train | Val | Test Accuracy {train_acc:.3f} | {val_acc:.3f} | {test_acc:.3f}")
        print(f"Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}\n", '-' * 70)

    print(f'Training finish! Best Val | Test Accuracy | {val_best_acc:.3f} | {test_best_acc:.3f}')

    return val_best_acc, test_best_acc



def eval_student(loader, device, args):
    stu_type = args.stu_type
    tea_type = 1 - stu_type

    log_np = np.zeros((args.num_runs, 2))
    for run in range(args.num_runs):
        print(f'Run {run}')
        net = MLP(input_dim=28*28*3).to(device) if stu_type == 0 else MLP(input_dim=28*28).to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        log_np[run, :] = train_network(stu_type,args.num_epochs, loader, net,device, optimizer)
    log_mean = np.mean(log_np, axis=0)
    log_std = np.std(log_np, axis=0)
    print(f'Finish {args.num_runs} runs')
    print(f'Student Val Acc {log_mean[0]:.3f} ± {log_std[0]:.3f} | Test Acc {log_mean[1]:.3f} ± {log_std[1]:.3f}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # the parameters you might need to change
    parser.add_argument('--data_root', default='/home/elon/Workshops/paper_KL_version2/MNIST/dataset/', help='data root [default: xxx]')
    parser.add_argument('--image_size', type=int, default=28, help='resize the image size [default: 32]')
    parser.add_argument('--noise_level', type=int, default=1.0, help='noise_level from 0 to 5') # 1.5
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--stu-type', type=int, default=0, help='the modality of student unimodal network, 0 for MNIST-M, 1 for MNIST')
    parser.add_argument('--num-runs', type=int, default=1, help='num runs')
    parser.add_argument('--num-epochs', type=int, default=50, help='num epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size') # 128
    parser.add_argument('--num-workers', type=int, default=10, help='dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-3, help='lr') # 1e-2
    parser.add_argument('--num-permute', type=int, default=10, help='number of permutation')
    parser.add_argument('--first-time', default=False, action="store_true", help='train overlap model')
    parser.add_argument('--cal_tag', default=False, action="store_true", help='calculate the amount of modality-decisive information for each feature channel')
    parser.add_argument('--eval_tag', default=True,action="store_true", help='verify MFH based on our calculated tag')
    parser.add_argument('--base_line', default=False, action="store_true", help='train network on MNIST or MNIST-M dataset')
    parser.add_argument('--heat_map', default=False,action="store_true", help='visualize heat map')
    parser.add_argument('--ratio', type=float, default=0.75, help='remove feature dimension ratio')
    parser.add_argument('--place', type=int, default=1, help='overlap tag place')
    parser.add_argument('--mode', type=int, default=2, help='remove idx mode')
    parser.add_argument('--gt-weight', type=float, default=0.5, help='gt loss weight') # 0.5
    parser.add_argument('--pl-weight', type=float, default=0.5, help='pl loss weight')
    
    # distillation
    parser.add_argument('--distill', type=str, default='similarity', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'crd', 'kdsvd', 'dkd',
                                                                      'rkd', 'pkt', ])
    
    # NCE distillation
    parser.add_argument('--feat_dim', default=32, type=int, help='feature dimension')
    parser.add_argument('--nce_k', default=10000, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')
    parser.add_argument('--n_data', default=60000, type=int, help='data number')
    parser.add_argument('--s_dim', default=10, type=int, help='student dim')
    parser.add_argument('--t_dim', default=10, type=int, help='teacher dim')

    args = parser.parse_args()
    print(args)

    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))
    loader = gen_data(args)
    
    if args.cal_tag:
        args.batch_size = 256 # batch size for calculating the overlap tag
        loader_fb = gen_data(args)
        cal_overlap_tag(args.stu_type, loader, loader_fb, args.num_epochs, args.lr, device, args.num_permute, args.place, args.first_time)

    if args.eval_tag:
        eval_overlap_tag(loader, device, args)
        
    if args.base_line:
        train_baseline(loader, device, args)
        
    if args.heat_map:
        heat_map(loader, device, args)