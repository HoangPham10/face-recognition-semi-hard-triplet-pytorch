import os
import torch 
BATCH_SIZE = 32
RESIZE_TO = 224
NUM_EPOCHS = 100
NUM_WORKER = os.cpu_count()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_DIR = [
    #'/kaggle/input/vn-celeb/VN-celeb/1 percent/train',
    #'/kaggle/input/vn-celeb/VN-celeb/full/train',
    '/kaggle/input/celeb-faces/full/train',
]

VAL_DIR = [
    #'/kaggle/input/vn-celeb/VN-celeb/1 percent/val',
     #'/kaggle/input/vn-celeb/VN-celeb/full/val',
    '/kaggle/input/celeb-faces/full/val',
]
