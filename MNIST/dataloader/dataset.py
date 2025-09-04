import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import utils, datasets, transforms
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

class CrossModalData(Dataset):
    

    def __init__(self, root, train=False, transform=None,noise_level=0,is_sample=True,k=None): 
        self.root = root
        self.transform = transform
        self.is_sample = is_sample
        self.k = k
        self.noise_level = noise_level # 噪声水平，可调整
        self.mnist_file = 'mnist_data.pkl'
        self.mnist_m_file = 'mnist_m_data.pkl'

        if not self._check_exists():  # 检查文件在不在
            raise RuntimeError('Dataset not found.')

       
        with open(os.path.join(self.root, 'MNIST', self.mnist_file), 'rb') as f:
            data = pkl.load(f)
            
        with open(os.path.join(self.root, 'MNIST_M', self.mnist_m_file), 'rb') as f:
            data_m = pkl.load(f)
        
        if train:
            
            self.data_mnist = data['train']
            self.data_mnist_m = data_m['train']
            self.targets_mnist = data['train_label']
            self.targets_mnist_m = data_m['train_label'] 
            
        else:
            self.data_mnist = data['val']
            self.data_mnist_m = data_m['val']
            self.targets_mnist = data['val_label']
            self.targets_mnist_m = data_m['val_label'] 
            
        num_classes = 10
        if train:
            num_samples = len(self.data_mnist)
            label = self.targets_mnist
        else:
            num_samples = len(self.data_mnist)
            label = self.targets_mnist

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i], dtype=object) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i], dtype=object) for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive, dtype=object)
        self.cls_negative = np.asarray(self.cls_negative, dtype=object)            
                    

    def __getitem__(self, index):  # 给下标，然后返回该下标对应的item

        img_m,target_m = self.data_mnist_m[index],self.targets_mnist[index] #img 是tensor类型,shape [28,28]
        
        img_m = self._add_noise(img_m, self.noise_level)
        
        img,target = self.data_mnist[index],self.targets_mnist_m[index] #img 是tensor类型,shape [28,28]
        
        noisy_image = self._add_noise(img, self.noise_level*4)
        
        
        if self.transform is not None:
            img = Image.fromarray(img)  # 从一个numpy对象转换成一个PIL image 对象
            img_m = Image.fromarray(img_m)  # 从一个numpy对象转换成一个PIL image 对象
            noisy_image = Image.fromarray(noisy_image) 
            noisy_image = self.transform(noisy_image) # img 的size为[1,28,28]
            img_m = self.transform(img_m)

        if not self.is_sample:
            # directly return
            
            sample = {'MNIST': noisy_image, 'MNIST_M': img_m, 'label': target}
            return sample
        else:
            # sample contrastive examples
           
            pos_idx = index
           
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            sample = {'MNIST': noisy_image, 'MNIST_M': img_m, 'label': target, 'index':index, 'sample': sample_idx}
            return sample




        # return noisy_image, img_m, target


    def __len__(self):  # 返回数据的长度
        return len(self.data_mnist)


    def _check_exists(self):
        print(os.path.join(self.root, 'MNIST', self.mnist_file))
        print(os.path.join(self.root, 'MNIST_M', self.mnist_m_file))
        return (os.path.exists(os.path.join(self.root, 'MNIST', self.mnist_file)) and os.path.exists(
            os.path.join(self.root, 'MNIST_M', self.mnist_m_file)))

    def _add_noise(self, image, noise_level=0.5):
        # 将图像转换为NumPy数组
        # np_image = np.array(image)

        # 生成与图像大小相同的随机噪声
        noise = np.random.rand(*image.shape)

        # 将噪声与图像混合
        noisy_image = np.clip(image/255 + noise_level * noise, 0, 1)
        
        noisy_image = np.uint8(noisy_image*255)

        # 将NumPy数组转换回Tensor
        # noisy_image = torch.tensor(noisy_image)

        return noisy_image








# 测试代码

def parse_args(args):
    desc = "Pytorch MINIST Classification"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size [default: 30]')
    parser.add_argument('--image_size', type=int, default=28, help='resize the image size [default: 32]')
    parser.add_argument('--data_root', default='/home/elon/Workshops/paper_KL_version2/MNIST/dataset/', help='data root [default: xxx]')
   
    return parser.parse_args(args)



if __name__ == '__main__':
    args = parse_args(args=[])
    
    transform = transforms.Compose([transforms.Resize(args.image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

    # 第一步：构造dataset对象
    dataset = CrossModalData(args.data_root, transform=transform,noise_level=0.5,is_sample=False,k=None)
    
    # 第二步：构造dataloader对象
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 第三步：迭代 dataloader

    for batch_idx, batch in enumerate(dataloader):
            
        # 注意 ，这三个数据都是 FloatTensor
        image_data = batch['MNIST']    # (8,1,32,32) ，之所以通道在前，是因为应用了transforms.ToTensor()
        image_m_data = batch['MNIST_M']
        image_label = batch['label']
        
    # 可视化图片
    cross_data = next(iter(dataloader))
    grid_image = utils.make_grid(cross_data['MNIST'][:,0:1,:,:][:64]*0.5+0.5, nrow=8)
    pil_image = transforms.ToPILImage()(grid_image)
    pil_image.save('merged_image.jpg')
    
    grid_image = utils.make_grid(cross_data['MNIST_M'][:64]*0.5+0.5, nrow=8)
    pil_image = transforms.ToPILImage()(grid_image)
    pil_image.save('merged_image_m.jpg')