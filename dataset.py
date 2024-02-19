from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
from config import RESIZE_TO
from tqdm.auto import tqdm
import random

class FaceDataset(Dataset):
    def __init__(self, root_dir, transforms = None) -> None:
        super().__init__()
        self.img_paths = glob.glob(os.path.join(root_dir, '*/*.jpg')) + glob.glob(os.path.join(root_dir, '*/*.png'))
        self.labels = [os.path.basename(os.path.abspath(os.path.join(path, os.pardir))) for path in self.img_paths]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path, label = self.img_paths[index], self.labels[index]
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
    
    def __len__(self):
        return len(self.img_paths)
    


class ValidationDataset(Dataset):
    def __init__(self, root_dir, transforms = None, max_files=10):
        self.transforms = transforms
        self.triplets = []
        folders = os.listdir(root_dir)

        for folder in tqdm(folders):
            path = os.path.join(root_dir, folder)
            files = list(os.listdir(path))[:max_files]
            num_files = len(files)

            for i in range(num_files-1):
                for j in range(i+1, num_files):
                    anchor = os.path.join(root_dir, folder, files[i])
                    positive = os.path.join(root_dir, folder, files[j])

                    neg_folder = folder
                    while neg_folder == folder:
                        neg_folder = random.choice(folders)
                    neg_file = random.choice( os.listdir(os.path.join(root_dir, neg_folder)))
                    negative = os.path.join(root_dir, neg_folder, neg_file)
                    self.triplets.append([anchor, positive, negative])
        
    def __getitem__(self, index):
        anchor, positive, negative = self.triplets[index]
        anchor, positive, negative = Image.open(anchor).convert('RGB'), Image.open(positive).convert('RGB'), Image.open(negative).convert('RGB')
        if self.transforms is not None:
            anchor, positive, negative = self.transforms(anchor), self.transforms(positive), self.transforms(negative)
        return anchor, positive, negative
    
    def __len__(self):
        return len(self.triplets)  


train_tfms = transforms.Compose([
    transforms.Resize(size=(RESIZE_TO, RESIZE_TO)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.25, contrast=0.25, saturation=0.25
    ),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    

val_tfms = transforms.Compose([
    transforms.Resize(size=(RESIZE_TO, RESIZE_TO)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    from config import TRAIN_DIR
    import random
    import cv2
    import numpy as np
    trainset = FaceDataset(root_dir=TRAIN_DIR[0], transforms=train_tfms)
    idx = random.randint(0,trainset.__getlen__() -1)
    ori_img = cv2.imread(trainset.img_paths[idx])
    transformed_img, label = trainset.__getitem__(idx)
    transformed_img = transformed_img.permute(1,2,0).numpy()
    transformed_img = cv2.cvtColor((transformed_img*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    delta_w = 224 - ori_img.shape[1]
    delta_h = 224 - ori_img.shape[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    ori_img = cv2.copyMakeBorder(ori_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

    img = cv2.hconcat([ori_img, transformed_img])
    cv2.imwrite('./out.png', img)
    print(label)
