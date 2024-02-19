import cv2
import torch
from torchvision import transforms
import os
import glob
from model import MobilenetV2Embedding
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

val_tfms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
if __name__ == '__main__':
    # face detection model initialization
    print('Loading face detection model ...')
    faceDet = cv2.CascadeClassifier('./weights/haarcascade_frontalface_default.xml')
    print('Done')


    # face recognition model initialization
    print('Loading face recognition model ...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    anchor_dir = './anchors'
    ckpt_path = './weights/best.pth'
    model = MobilenetV2Embedding()
    model = model.to(device)
    checkpoint = torch.load(ckpt_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print('Done')

    # anchor loading
    print('Loading anchors ... ')
    labels = os.listdir(anchor_dir)
    anchors = dict(labels=[], embeddings=[])
    for i, label in tqdm(enumerate(labels)):
        files = os.listdir(os.path.join(anchor_dir, label))
        for file in files:
            print(label, file)
            image = Image.open(os.path.join(anchor_dir, label, file))
            tf_image = val_tfms(image).unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                embedding = model(tf_image).squeeze()
            anchors['labels'].append(label)
            anchors['embeddings'].append(embedding)
    anchor_embeddings = torch.stack(anchors['embeddings']).cpu()
    print(anchor_embeddings.shape)
    print('Done')



    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    out = cv2.VideoWriter('./out.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        faces = faceDet.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                         scaleFactor = 1.1,
                                         minNeighbors = 4,
                                         minSize=(50, 50),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in faces:
            x1, y1, x2, y2 = x, y, x+w, y+h
            infer_img = frame[y1:y2, x1:x2]
            #infer_img = np.array(255*(frame[y1:y2, x1:x2]/255)**0.8,dtype='uint8')
            # cv2.imwrite('./out.jpg', infer_img)
            image = Image.fromarray(cv2.cvtColor(infer_img, cv2.COLOR_BGR2RGB))
            tf_image = val_tfms(image).unsqueeze(dim=0).to(device)
            with torch.inference_mode():
                embedding = model(tf_image).squeeze()
            distance = (embedding - anchor_embeddings).pow(2).sum(1).pow(0.5)

            pred_label, min_dist = torch.argmin(distance).item(), torch.min(distance).item()
            distance = distance.numpy().tolist()
            print('--------------------')
            for dist, label in zip(distance, anchors['labels']):
                print(label, dist)

            if min_dist > 0.5:
                label = 'Guest'
            else:
                label = anchors['labels'][pred_label]
            label = anchors['labels'][pred_label]

            cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} _ {min_dist}", (x,y), 1, 1, (0,0,255), 2)
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


