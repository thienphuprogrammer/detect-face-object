import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        self.bboxes = []
        self.landmarks = []
        self.flag = True
        with open(txt_path, 'r') as fr:
            lines = fr.readlines()
            for idx, line in enumerate(lines):
                line = line.strip().split(' ')
                if self.flag:
                    self.imgs_path.append(line[0])
                    self.words.append([])
                    self.bboxes.append([])
                    self.landmarks.append([])
                    word_nums = int(line[1])
                    for i in range(word_nums):
                        self.words[idx].append(line[2 + i])
                        self.bboxes[idx].append(list(map(float, line[2 + word_nums + 4 * i: 6 + word_nums + 4 * i])))
                        self.landmarks[idx].append(list(map(float, line[6 + word_nums + 4 * i: 14 + word_nums + 4 * i])))
                    if len(self.bboxes[idx]) == 0:
                        self.bboxes[idx] = [[0, 0, 0, 0]]
                    if len(self.landmarks[idx]) == 0:
                        self.landmarks[idx] = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
                self.flag = not self.flag

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        if self.flag:
            words = self.words[index]
            bboxes = self.bboxes[index]
            landmarks = self.landmarks[index]
            if self.preproc is not None:
                img, bboxes, landmarks = self.preproc(img, bboxes, landmarks)
            bboxes = np.array(bboxes)
            landmarks = np.array(landmarks)
            return img, bboxes, landmarks
        else:
            if self.preproc is not None:
                img, _, _ = self.preproc(img)
            return img
        

def preproc_for_test(img, bboxes, landmarks):
    img = img.astype(np.float32)
    img = img / 255.0
    return img, bboxes, landmarks

def preproc_for_train(img, bboxes, landmarks):
    img = img.astype(np.float32)
    img = img / 255.0
    return img, bboxes, landmarks

def collate_fn(batch):
    imgs = []
    bboxes = []
    landmarks = []
    for sample in batch:
        imgs.append(sample[0])
        bboxes.append(sample[1])
        landmarks.append(sample[2])
    return np.array(imgs), bboxes, landmarks


def train():
    from yolov9 import YOLOv9
    from loss import YOLOv9Loss
    from config import train_cfg
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import MultiStepLR
    from torchvision.transforms import transforms
    import os
    import time

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    yolo = YOLOv9().to(device)
    yolo.train()
    yolo.load_state_dict(torch.load(train_cfg['pretrain_model_path']))
    yolo.freeze_backbone()
    yolo.freeze_bn()
    yolo.freeze_head()
    yolo.freeze_body()
    yolo.freeze_neck()
    yolo.freeze_det_head()
    yolo.freeze_det_body()
    yolo.freeze_det_neck()
    yolo.freeze_det_head()
    yolo.freeze_det_body()
    yolo.freeze_det_neck()
    yolo.freeze_det_head()
    yolo.freeze_det_body()
    yolo.freeze_det_neck()

    train_dataset = WiderFaceDetection(train_cfg['train_txt_path'], preproc=preproc_for_train)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
    criterion = YOLOv9Loss()
    optimizer = Adam(yolo.parameters(), lr=train_cfg['lr'])
    scheduler = MultiStepLR(optimizer, milestones=train_cfg['milestones'], gamma=train_cfg['gamma'])
    
    for epoch in range(train_cfg['epochs']):
        yolo.train()
        for i, (imgs, bboxes, landmarks) in enumerate(train_loader):
            imgs = torch.from_numpy(imgs).to(device)
            bboxes = [torch.from_numpy(bbox).to(device) for bbox in bboxes]
            landmarks = [torch.from_numpy(landmark).to(device) for landmark in landmarks]
            pred = yolo(imgs)
            loss = criterion(pred, bboxes, landmarks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch: %d, Iter: %d, Loss: %.6f' % (epoch, i, loss.item()))
        scheduler.step()
        torch.save(yolo.state_dict(), 'yolov9_epoch_%d.pth' % epoch)


if __name__ == '__main__':
    train()
