import torch
import torch.nn as nn
from torchvision import transforms
import tqdm
from matchingnet import MatchingNet
from kitti_dataloader import KittiDataLoader
import argparse
from logger import Logger
import torch.nn.functional as F
from torch.autograd import Variable
import os

parser = argparse.ArgumentParser()
parser.add_argument('--feat_dim', type=int, default=512, help='embedding dimention')
parser.add_argument('--epoch_num', type=int, default=100, help='training epoch')
parser.add_argument('--log_dir', type=str, default='log', help='log dir')
parser.add_argument('--data_dir', type=str, default='data', help='dataset dir')
parser.add_argument('--save_dir', type=str, default='model', help='model dir')
parser.add_argument('--seq_num', type=int, default=21, help='the total number of sequences')
parser.add_argument('--lr', typr=float, default=1e-4, help='learning rate')
parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or not')
args = parser.parse_args()

training_set = []
validation_set = []
testing_set = []
seq_num = argps.seq_num
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
    
    def forward(self, gallery_images, gallery_labels, gallery_one_hot, query_images, query_labels, query_one_hot):
        preds_q, TD_pred_q = self.model(gallery_images, gallery_one_hot, query_images)
        pred_g, TD_pred_g = self.model(query_images, query_one_hot, gallery_images)
        valid_q_index = []
        valid_g_index = []
        TD_label = []
        for i, q_l in enumerate(query_images):
            if q_l != -1 and q_l is in gallery_labels:
                valid_q_index.append(i)
        for i, g_l in enumerate(gallery_images):
            if g_l != -1 and g_l is in query_labels:
                valid_g_index.append(i)
            if g_l == -1:
                TD_label.append(0)
            else:
                TD_label.append(1)
        TD_label_np = np.array(TD_label)
        loss1 = F.cross_entropy(preds_q[valid_q_index], query_labels[valid_q_index])
        loss2 = F.cross_entropy(pred_g[valid_g_index], gallery_labels[valid_g_index])
        loss_td = F.binary_cross_entrppy(TD_pred_q, TD_label_np)
        loss = loss1 + loss2 + loss_td

        values_q, indices_q = preds_q.max(1)
        accuracy_q = torch.mean((indices_q.squeeze()[valid_q_index] == query_labels[valid_q_index]))

        values_g, indices_g = preds_g.max(1)
        accuracy_g = torch.mean((indices_g.squeeze()[valid_g_index] == gallery_labels[valid_g_index]))

        accuracy = (accuracy_q + accuracy_g)/2

        return loss, accuracy


def main():
    LOG_DIR = args.log_dir + '/feat_dim_{}-epoch_{}'.format(feat_dim, epoch)
    logger = Logger(LOG_DIR)

    data = KittiDataLoader(data_root_path=data_root_path, image_root=image_root, 
                           annotation_root=annotation_root, det_root=det_root, 
                           training_set=training_set, validation_set=validation_set, 
                           testing_set=testing_set, seq_num=seq_num,
                           transform=transforms.Compose([Normalizer(mean=mean, std=std),
                                                         Resizer((288, 96))]))

    matchingnet = MatchingNet(feat_dim=args.feat_dim)
    model = ModelWithLoss(matchingnet)

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
                gallery_images, gallery_labels, query_images, query_labels, num_class = data.get_batch(data_type='train')

                gallery_images = Variable(torch.from_numpy(gallery_images)).float()
                gallery_labels = Variable(torch.from_numpy(gallery_labels), requires_grad=False).long()
                query_images = Variable(torch.from_numpy(query_images)).float()
                query_labels = Variable(torch.from_numpy(query_labels), requires_grad=False).long()

                # one-hot-encode
                gallery_num = gallery_labels.size()[0]
                gallery_one_hot = torch.FloatTensor(gallery_num, num_class).zero_()
                gallery_one_hot.scatter_(1, torch.unsqueeze(gallery_labels, 1).data, 1)
                gallery_one_hot = Variable(gallery_one_hot)

                query_num = query_labels.size()[0]
                query_one_hot = torch.FloatTensor(query_num, num_class).zero_()
                query_one_hot.scatter_(1, torch.unsqueeze(query_labels, 1).data, 1)
                query_one_hot = Variable(query_one_hot)

                if use_cuda:
                    loss, accuracy = model(gallery_images.cuda(), gallery_labels.cuda(), gallery_one_hot.cuda(), query_images.cuda(), query_labels.cuda(), query_one_hot.cuda())
                else:
                    loss, accuracy = model(gallery_images, gallery_labels, gallery_one_hot, query_images, query_labels, query_one_hot)
                
                optimizer.zero_grad()

                total_loss.backward()

                optimizer.step()

                iter_out = "train_loss: {}, train_accuracy: {}".format(loss.data[0], accuracy.data[0])
                pbar.set_description(iter_out)

                pbar.updata(1)

                total_loss += loss.data[0]
                total_accuracy += accuracy.data[0]

        total_loss /= train_iter_num
        total_accuracy /= train_iter_num

        print("Epoch {}: train_loss: {}, train_accuracy: {}".format(epoch, total_loss, total_accuracy))

        # validation
        val_iter_num = data.get_iter_num('val')
        total_val_loss = 0
        total_val_accuracy = 0
        with tqdm.tqdm(total=val_iter_num) as pbar:
            for i in range(val_iter_num):
                gallery_images, gallery_labels, query_images, query_labels, num_class = data.get_batch(data_type='val')

                with torch.no_grad():
                    gallery_images = Variable(torch.from_numpy(gallery_images)).float()
                    gallery_labels = Variable(torch.from_numpy(gallery_labels)).long()
                    query_images = Variable(torch.from_numpy(query_images)).float()
                    query_labels = Variable(torch.from_numpy(query_labels)).long()

                    # one-hot-encode
                    gallery_num = gallery_labels.size()[0]
                    gallery_one_hot = torch.FloatTensor(gallery_num, num_class).zero_()
                    gallery_one_hot.scatter_(1, torch.unsqueeze(gallery_labels, 1).data, 1)
                    gallery_one_hot = Variable(gallery_one_hot)

                    query_num = query_labels.size()[0]
                    query_one_hot = torch.FloatTensor(query_num, num_class).zero_()
                    query_one_hot.scatter_(1, torch.unsqueeze(query_labels, 1).data, 1)
                    query_one_hot = Variable(query_one_hot)

                    if use_cuda:
                        loss, accuracy = model(gallery_images.cuda(), gallery_labels.cuda(), gallery_one_hot.cuda(), query_images.cuda(), query_labels.cuda(), query_one_hot.cuda())
                    else:
                        loss, accuracy = model(gallery_images, gallery_labels, gallery_one_hot, query_images, query_labels, query_one_hot)

                    iter_out = "val_loss: {}, val_accuracy: {}".format(loss.data[0], accuracy.data[0])
                    pbar.set_description(iter_out)
                    pbar.updata(1)

                    total_val_loss += loss.data[0]
                    total_val_accuracy += accuracy.data[0]

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




            
    
