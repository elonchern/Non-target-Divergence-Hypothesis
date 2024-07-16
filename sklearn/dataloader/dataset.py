from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
import numpy as np
import torch

def dataset(n_samples=10000, n_classes=4, n_features=50, n_informative=50, n_redundant=0, n_clusters_per_class=1,random_state=42,distill='kl',opt=None):
    datasets = make_classification(n_samples=n_samples, n_classes=n_classes, n_features=n_features,
                                   n_informative=n_informative, n_redundant=n_redundant, 
                                   n_clusters_per_class=n_clusters_per_class,random_state=random_state)
    X, y = datasets
    # preprocess dataset, split into training and test part
    X = StandardScaler().fit_transform(X)
    X_train, X_t, y_train, y_t = train_test_split(X, y, test_size=.4, random_state=42)
    
    
    if distill == 'crd':
        # 构造正负样本的index
        num_classes = n_classes
        label = np.array(y_train).astype(int)
        num_samples = len(X_train)
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
        return X_train,X_t, y_train, y_t, torch.LongTensor(idx),torch.LongTensor(sample_idx)
    
    else:
          
        return X_train,X_t, y_train, y_t    
    
    
    
    