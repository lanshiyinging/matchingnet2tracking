import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F

class ResnetEncoder(nn.Module):
    def __init__(self, feat_dim):
        super(ResnetEncoder, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=feat_dim)
    
    def forward(self, inputs):
        feats = self.resnet(inputs)
        return feats

class AttentionalEmbed(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AttentionalEmbed, self).__init__()
        
        self.emb = nn.Sequential(
            nn.Linear(in_features=in_dim*2, out_features=out_dim),
            nn.Tanh()
        )
        self.softmax = nn.Softmax()
        
    def forward(self, query_encode, gallery_encode, gallery_label):
        gallery_encode_t = gallery_encode.t()
        inner_product = torch.matmul(query_encode, gallery_encode_t)
        att = torch.matmul(inner_product, gallery_label)
        softmax_att = self.softmax(att)
        r = torch.matmul(softmax_att, gallery_encode)
        x = torch.cat((query_encode, r), dim=1)
        outputs = self.emb(x)
        return outputs


class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, vector_dim):
        super(BidirectionalLSTM, self).__init__()

        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)

        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)
    
    def forward(self, inputs):
        c0 = Variable(torch.rand(self.lstm.num_layers*2, inputs.size()[1], self.lstm.hidden_size),
                      requires_grad=False).cuda()
        h0 = Variable(torch.rand(self.lstm.num_layers*2, inputs.size()[1], self.lstm.hidden_size),
                      requires_grad=False).cuda()
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output[0], hn, cn

class Classify(nn.Module):
    def __init__(self):
        super(Classify, self).__init__()
        self.softmax = nn.Softmax()
    
    def forward(self, query_encode, gallery_encode, gallery_label):
        gallery_encode_t = gallery_encode.t()
        inner_product = torch.matmul(query_encode, gallery_encode_t)
        att = torch.matmul(inner_product, gallery_label)
        preds = self.softmax(att)
        return preds

class MatchingNet(nn.Module):
    eps = 1e-10
    def __init__(self, feat_dim):
        super(MatchingNet, self).__init__()

        # embedding network
        self.encoder = ResnetEncoder(feat_dim)

        # fully context embedding
        self.CEN_Q = AttentionalEmbed(feat_dim, feat_dim)
        self.CEN_G = BidirectionalLSTM(layer_sizes=[int(feat_dim/2)], vector_dim=feat_dim)
        
        # softmax
        self.classify = Classify()

        # TD_clf
        self.TD_clf = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=1),
            nn.Sigmoid()
        )
        
    def forward(self, gallery_images, gallery_labels, query_images):
        
        # gallery embedding

        gallery_encode = self.encoder(gallery_images)
        gallery_encode = torch.unsqueeze(gallery_encode, 0)
        gallery_cen, hn, cn = self.CEN_G(gallery_encode)

        # query embedding
        query_encode = self.encoder(query_images)
        query_cen = self.CEN_Q(query_encode, gallery_cen, gallery_labels)

        preds = self.classify(query_cen, gallery_cen, gallery_labels)

        TD_pred = self.TD_clf(gallery_encode)
        
        return preds, TD_pred









        

