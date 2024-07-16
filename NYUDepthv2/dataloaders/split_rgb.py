import shutil
import os




class SplitRGB(object):
    def __init__(self, src_path,dst_root):
        self.filepaths = {'train':[],'test':[]}
        
        self.src_path = src_path
        
        self.dst_root = dst_root
        
       
        
        self.get_filepaths()
                
        
    def get_filepaths(self):
        
        with open(os.path.join(self.src_path, 'train.txt')) as f:
            for line in f.readlines():
                id = line.strip()
                img_name = 'NYU'+str(id)+'_colors.png'
                
                self.filepaths['train']+= [os.path.join(self.src_path,'RGB',img_name)]
                
        
        with open(os.path.join(self.src_path, 'test.txt')) as f:
            for line in f.readlines():
                id = line.strip()
                img_name = 'NYU'+str(id)+'_colors.png'
                
                self.filepaths['test']+= [os.path.join(self.src_path,'RGB',img_name)]    
        print(self.filepaths['test'])        
        return  
        
        
    def save_img(self):
        
        train_dir = os.path.join(self.dst_root, "train_rgb")
        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)
        
        test_dir = os.path.join(self.dst_root, "test_rgb")
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)     
              
        for image in self.filepaths['train']:
            # print("=========")
            # print(os.path.basename(image))
            # pass
            new_name = os.path.basename(image)[:-10]+'rgb'+ '.'+ image.split('.')[1]
            shutil.copy(image, os.path.join(train_dir, new_name))
            
            
        for image in self.filepaths['test']:
            
            new_name = os.path.basename(image)[:-10]+'rgb'+ '.'+ image.split('.')[1]
            shutil.copy(image, os.path.join(test_dir, new_name))
            
            
if __name__ == "__main__":
    
    src_path = '/data/elon/NYU_Raw/depthbin'
    dst_root = '/data/elon/NYU_Raw/depthbin'
    model = SplitRGB(src_path,dst_root)
    model.save_img()