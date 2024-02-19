import os
import argparse
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


from model import MobilenetV2Embedding, Resnet50Embedding
from dataset import FaceDataset, ValidationDataset,  train_tfms, val_tfms
from loss import SemiHardTripletLoss
from utils import BalancedBatchSampler
from config import TRAIN_DIR, VAL_DIR, DEVICE
from PIL import Image

if __name__ == "__main__":
    #TODO: Create ArgumentParser object 
    parser = argparse.ArgumentParser(description="Training script for Face Identification")
    # Add arguments
    parser.add_argument('--lr', type=float, default=0.001 , help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--n_classes', type=int, default=25, help='The number of chosen classes')
    parser.add_argument('--n_samples', type=int, default=10, help='the number of chosen samples in each class')
    parser.add_argument('--margin', type=float, default=0.5, help='acceptable margin')
    parser.add_argument('--max_files', type=int, default=5, help='the maximum number of files per folder in validation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping')
    parser.add_argument('--logdir', type=str, default='./exp', help='tensorboard')
    # Parse the command-line arguments
    args = parser.parse_args()

    #TODO: Hyper-parameters initialization
    print('-------------------- Hyper-parameters initialization -----------------------')
    LR, EPOCHS, N_CLASSES, N_SAMPLES, MARGIN, MAX_FILES, PATIENCE = args.lr, args.epochs, args.n_classes, args.n_samples, args.margin, args.max_files, args.patience
    writer = SummaryWriter(log_dir=args.logdir)

    #TODO: dataset initialization
    print('-------------------- Dataset initialization -----------------------')
    trainset = FaceDataset(TRAIN_DIR[0], train_tfms)
    print(f"Number of training samples: {trainset.__len__()}")
    valset = ValidationDataset(VAL_DIR[0], val_tfms, max_files=MAX_FILES)
    print(f"Number of validation samples: {valset.__len__()}")

    train_loader = DataLoader(trainset, batch_sampler=BalancedBatchSampler(trainset, N_CLASSES, N_SAMPLES), num_workers=os.cpu_count())
    val_loader = DataLoader(valset, batch_size=64, num_workers=os.cpu_count())

    #TODO: Model initialization
    model = MobilenetV2Embedding(192)
    model = model.to(DEVICE)

    # Total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    #TODO: Optimizer and loss function initialization
    optimizer = torch.optim.Adam(model.parameters(), LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    loss_fn = SemiHardTripletLoss(device=DEVICE, margin=MARGIN)

    best_epoch, best_acc = 0, 0

    #TODO: Training epochs
    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader)
        for idx, (images, labels) in enumerate(pbar):
            labels = torch.Tensor([int(v) for v in list(labels)]).to(DEVICE)
            images = images.to(DEVICE)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                embeddings = model(images)
                loss = loss_fn(embeddings, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            pbar.set_description("Epoch {}. Iteration {}/{} Loss {}".format(
                epoch+1, 
                idx+1, 
                len(train_loader), 
                loss.item()))
        scheduler.step()
        total_loss /= len(train_loader)
        writer.add_scalar('Loss', total_loss,global_step=epoch+1)
        
        # Validation
        model.eval()
        pbar = tqdm(val_loader)
        pos_scores,  neg_scores = [], []
        for idx, (anchors, positives, negatives) in enumerate(pbar):
            with torch.inference_mode():
                anchor_embeddings, positive_embeddings, negative_embeddings = model(anchors.to(DEVICE)),  model(positives.to(DEVICE)),  model(negatives.to(DEVICE))
            positive_distance = (anchor_embeddings - positive_embeddings).pow(2).sum(1).pow(.5).cpu().numpy().tolist()
            negative_distance = (anchor_embeddings - negative_embeddings).pow(2).sum(1).pow(.5).cpu().numpy().tolist()
            pos_scores += positive_distance
            neg_scores += negative_distance
    
        accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
#         ap_mean = np.mean(pos_scores)
#         an_mean = np.mean(neg_scores)
#         ap_stds = np.std(pos_scores)
#         an_stds = np.std(neg_scores)

        # Validation
#         model.eval()
#         accuracy = []
#         with torch.inference_mode():
#             anchors, samples = [], []
#             labels = os.listdir(VAL_DIR[0])
#             for i, label in tqdm(enumerate(labels)):
#                 files = sorted(os.listdir(os.path.join(VAL_DIR[0], label)))
#                 for idx, file in enumerate(files):
#                     image = Image.open(os.path.join(VAL_DIR[0], label, file))
#                     tf_image = val_tfms(image).unsqueeze(dim=0).to(DEVICE)
#             #         print(tf_image.shape)
#                     with torch.inference_mode():
#                         embedding = model(tf_image).squeeze()
#                     if idx < len(files)/4:
#                         anchors.append(dict(idx=i,label=label, embedding = embedding, img_path = os.path.join(VAL_DIR[0], label, file)))
#                     else:
#                         samples.append(dict(idx=i,label=label, embedding = embedding, img_path = os.path.join(VAL_DIR[0], label, file)))

#         anchor_embeddings = torch.stack([v['embedding'] for v in anchors])
#         n_true_samples, n_wrong_samples = 0, 0

#         for sample in tqdm(samples):
#             embedding = sample['embedding']
#             distance = (embedding - anchor_embeddings).pow(2).sum(1).pow(0.5)

#             pred_label = torch.argmin(distance).item()

#             if pred_label == int(sample['idx']):
#                 n_true_samples += 1
#             else:
#                 n_wrong_samples += 1
                
        
        
#         accuracy = n_true_samples/(n_true_samples + n_wrong_samples)
        
        writer.add_scalar('accuracy', accuracy, global_step=epoch+1)
        print("Epoch {}. Accuracy {}".format(
                    epoch+1, 
                    accuracy)
                )
        checkpoint = dict(
            model = model.state_dict(),
            optimizer = optimizer.state_dict()
        )
        torch.save(checkpoint, 'last.pth')
    
        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch
            torch.save(checkpoint, './best.pth')
        
        if epoch - best_epoch > PATIENCE:
            break
        
