import torch
import torch.nn as nn
from torchvision import transforms
import tqdm
from matchingnet import MatchingNet
from kitti_dataloader import KittiDataLoader, Resizer, Normalizer
import argparse
from logger import Logger
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--feat_dim', type=int, default=512, help='embedding dimention')
parser.add_argument('--epoch_num', type=int, default=100, help='training epoch')
parser.add_argument('--log_dir', type=str, default='log', help='log dir')
parser.add_argument('--data_dir', type=str, default='data', help='dataset dir')
parser.add_argument('--save_dir', type=str, default='model', help='model dir')
parser.add_argument('--seq_num', type=int, default=21, help='the total number of sequences')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')
args = parser.parse_args()

training_set = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 18, 19]
validation_set = [4, 9, 14, 20]
#training_set = [0]
#validation_set = [1]
testing_set = []
seq_num = args.seq_num
feat_dim = args.feat_dim
epoch_num = args.epoch_num
data_root_path = args.data_dir
image_root = 'image_02'
annotation_root = 'label_02'
det_root = 'det_02'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
lr = args.lr
use_cuda = args.use_cuda

class ModelWithLoss(nn.Module):
    def __init__(self, model, ld):
        super().__init__()
        self.model = model
        self.ld = ld
    
    def forward(self, gallery_images, gallery_labels, gallery_one_hot, query_images, query_labels, r_gallery_images, r_gallery_labels, r_gallery_one_hot, r_query_images, r_query_labels):
        preds_q, TD_pred_q = self.model(gallery_images, gallery_one_hot, query_images)
        r_preds_q, r_TD_pred_q = self.model(r_gallery_images, r_gallery_one_hot, r_query_images)
        valid_q_index = []
        valid_rq_index = []
        for i, q_l in enumerate(query_labels):
            if q_l != -1:
                valid_q_index.append(i)
        for i, rq_l in enumerate(r_query_labels):
            if rq_l != -1:
                valid_rq_index.append(i)
        
        TD_label = []
        for i, g_l in enumerate(gallery_labels):
            if g_l == -1:
                TD_label.append(0)
            else:
                TD_label.append(1)
        
        TD_label = torch.Tensor(TD_label)
        TD_label = torch.unsqueeze(TD_label, 1)
        if use_cuda:
            TD_label = TD_label.cuda()

        loss1 = F.cross_entropy(preds_q, query_labels)
        loss2 = F.cross_entropy(r_preds_q, r_query_labels)
        loss_td = F.binary_cross_entropy(TD_pred_q[0], TD_label)
        loss = loss1 + loss2 + loss_td

        values_q, indices_q = preds_q.max(1)
        accuracy_q = torch.mean((indices_q.squeeze() == query_labels).float())
        
        r_values_q, r_indices_q = r_preds_q.max(1)
        r_accuracy_q = torch.mean((r_indices_q.squeeze() == r_query_labels).float())

        accuracy = (accuracy_q + r_accuracy_q)/2

        return loss, accuracy


def main():
    LOG_DIR = args.log_dir + '/feat_dim_{}-epoch_{}'.format(feat_dim, epoch_num)
    logger = Logger(LOG_DIR)

    data = KittiDataLoader(data_root_path=data_root_path, image_root=image_root, 
                           annotation_root=annotation_root, det_root=det_root, 
                           training_set=training_set, validation_set=validation_set, 
                           testing_set=testing_set, seq_num=seq_num,
                           transform=transforms.Compose([Normalizer(mean=mean, std=std),
                                                         Resizer(img_size=224)]))

    matchingnet = MatchingNet(feat_dim=args.feat_dim)
    model = ModelWithLoss(matchingnet, ld=1)

    if use_cuda:
        model.cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    best_val = 0.
    for epoch in range(epoch_num):

        # training
        train_iter_num = data.get_iter_num('train')
        total_loss = 0
        total_accuracy = 0
        with tqdm.tqdm(total=train_iter_num) as pbar:
            for i in range(train_iter_num):
                gallery_images, gallery_labels, query_images, query_labels, num_class, r_gallery_images, r_gallery_labels, r_query_images, r_query_labels, r_num_class = data.get_batch(data_type='train')

                gallery_images = Variable(torch.from_numpy(gallery_images)).float()
                gallery_labels = Variable(torch.from_numpy(gallery_labels), requires_grad=False).long()
                query_images = Variable(torch.from_numpy(query_images)).float()
                query_labels = Variable(torch.from_numpy(query_labels), requires_grad=False).long()

                r_gallery_images = Variable(torch.from_numpy(r_gallery_images)).float()
                r_gallery_labels = Variable(torch.from_numpy(r_gallery_labels), requires_grad=False).long()
                r_query_images = Variable(torch.from_numpy(r_query_images)).float()
                r_query_labels = Variable(torch.from_numpy(r_query_labels), requires_grad=False).long()

                # one-hot-encode
                gallery_num = gallery_labels.size()[0]
                gallery_one_hot = torch.FloatTensor(gallery_num, num_class).zero_()
                gallery_one_hot.scatter_(1, torch.unsqueeze(gallery_labels, 1).data, 1)
                gallery_one_hot = Variable(gallery_one_hot)

                r_gallery_num = r_gallery_labels.size()[0]
                r_gallery_one_hot = torch.FloatTensor(r_gallery_num, r_num_class).zero_()
                r_gallery_one_hot.scatter_(1, torch.unsqueeze(r_gallery_labels, 1).data, 1)
                r_gallery_one_hot = Variable(r_gallery_one_hot)
                
                '''
                query_num = query_labels.size()[0]
                query_one_hot = torch.FloatTensor(query_num, num_class).zero_()
                query_one_hot.scatter_(1, torch.unsqueeze(query_labels, 1).data, 1)
                query_one_hot = Variable(query_one_hot)
                '''

                # reshape channels
                size = gallery_images.size()
                gallery_images = gallery_images.view(size[0],size[3],size[1],size[2])
                size = query_images.size()
                query_images = query_images.view(size[0],size[3],size[1],size[2])

                size = r_gallery_images.size()
                r_gallery_images = r_gallery_images.view(size[0],size[3],size[1],size[2])
                size = r_query_images.size()
                r_query_images = r_query_images.view(size[0],size[3],size[1],size[2])

                if use_cuda:
                    loss, accuracy = model(gallery_images.cuda(), gallery_labels.cuda(), gallery_one_hot.cuda(), query_images.cuda(), query_labels.cuda(), r_gallery_images.cuda(), r_gallery_labels.cuda(), r_gallery_one_hot.cuda(), r_query_images.cuda(), r_query_labels.cuda())
                else:
                    loss, accuracy = model(gallery_images, gallery_labels, gallery_one_hot, query_images, query_labels, r_gallery_images, r_gallery_labels, r_gallery_one_hot, r_query_images, r_query_labels)
                
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                iter_out = "iter: {}, train_loss: {}, train_accuracy: {}".format(i, loss.item(), accuracy.item())
                pbar.set_description(iter_out)

                pbar.update(1)

                total_loss += loss.item()
                total_accuracy += accuracy.item()

        total_loss /= train_iter_num
        total_accuracy /= train_iter_num

        print("Epoch {}: train_loss: {}, train_accuracy: {}".format(epoch, total_loss, total_accuracy))

        # validation
        val_iter_num = data.get_iter_num('val')
        total_val_loss = 0
        total_val_accuracy = 0
        with tqdm.tqdm(total=val_iter_num) as pbar:
            for i in range(val_iter_num):
                gallery_images, gallery_labels, query_images, query_labels, num_class, r_gallery_images, r_gallery_labels, r_query_images, r_query_labels, r_num_class = data.get_batch(data_type='val')
                
                with torch.no_grad():
                    gallery_images = Variable(torch.from_numpy(gallery_images)).float()
                    gallery_labels = Variable(torch.from_numpy(gallery_labels), requires_grad=False).long()
                    query_images = Variable(torch.from_numpy(query_images)).float()
                    query_labels = Variable(torch.from_numpy(query_labels), requires_grad=False).long()

                    r_gallery_images = Variable(torch.from_numpy(r_gallery_images)).float()
                    r_gallery_labels = Variable(torch.from_numpy(r_gallery_labels), requires_grad=False).long()
                    r_query_images = Variable(torch.from_numpy(r_query_images)).float()
                    r_query_labels = Variable(torch.from_numpy(r_query_labels), requires_grad=False).long()

                    # one-hot-encode
                    gallery_num = gallery_labels.size()[0]
                    gallery_one_hot = torch.FloatTensor(gallery_num, num_class).zero_()
                    gallery_one_hot.scatter_(1, torch.unsqueeze(gallery_labels, 1).data, 1)
                    gallery_one_hot = Variable(gallery_one_hot)

                    r_gallery_num = r_gallery_labels.size()[0]
                    r_gallery_one_hot = torch.FloatTensor(r_gallery_num, r_num_class).zero_()
                    r_gallery_one_hot.scatter_(1, torch.unsqueeze(r_gallery_labels, 1).data, 1)
                    r_gallery_one_hot = Variable(r_gallery_one_hot)
                    
                    '''
                    query_num = query_labels.size()[0]
                    query_one_hot = torch.FloatTensor(query_num, num_class).zero_()
                    query_one_hot.scatter_(1, torch.unsqueeze(query_labels, 1).data, 1)
                    query_one_hot = Variable(query_one_hot)
                    '''

                    # reshape channels
                    size = gallery_images.size()
                    gallery_images = gallery_images.view(size[0],size[3],size[1],size[2])
                    size = query_images.size()
                    query_images = query_images.view(size[0],size[3],size[1],size[2])

                    size = r_gallery_images.size()
                    r_gallery_images = r_gallery_images.view(size[0],size[3],size[1],size[2])
                    size = r_query_images.size()
                    r_query_images = r_query_images.view(size[0],size[3],size[1],size[2])

                    if use_cuda:
                        loss, accuracy = model(gallery_images.cuda(), gallery_labels.cuda(), gallery_one_hot.cuda(), query_images.cuda(), query_labels.cuda(), r_gallery_images.cuda(), r_gallery_labels.cuda(), r_gallery_one_hot.cuda(), r_query_images.cuda(), r_query_labels.cuda())
                    else:
                        loss, accuracy = model(gallery_images, gallery_labels, gallery_one_hot, query_images, query_labels, r_gallery_images, r_gallery_labels, r_gallery_one_hot, r_query_images, r_query_labels)

                    iter_out = "val_loss: {}, val_accuracy: {}".format(loss.item(), accuracy.item())
                    pbar.set_description(iter_out)
                    pbar.update(1)

                    total_val_loss += loss.item()
                    total_val_accuracy += accuracy.item()

        total_val_loss /= val_iter_num
        total_val_accuracy /= val_iter_num

        print("Epoch {}: val_loss: {}, val_accuracy: {}".format(epoch, total_val_loss, total_val_accuracy))

        logger.log_value('train_loss', total_loss)
        logger.log_value('train_acc', total_accuracy)
        logger.log_value('val_loss', total_val_loss)
        logger.log_value('val_acc', total_val_accuracy)

        if total_val_accuracy >= best_val:
            best_val = total_val_accuracy
            save_path = os.path.join(args.save_dir, 'matchingnet_{}.pth'.format(epoch))
            torch.save(model.model.state_dict(), save_path)



if __name__ == "__main__":
    main()
            
    
