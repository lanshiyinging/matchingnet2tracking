import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from srnn.model import SRNN
from srnn.st_graph import ST_GRAPH
from srnn.helper import getCoef, sample_gaussian_2d
from srnn.criterion import Gaussian2DLikelihood
from kitti_dataloader_SA import iou


class ResnetEncoder(nn.Module):
    def __init__(self, feat_dim):
        super(ResnetEncoder, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=feat_dim)
    
    def forward(self, inputs):
        feats = self.resnet(inputs)
        return feats

class AttentionalEmbed(nn.Module):
    def __init__(self, args):
        super(AttentionalEmbed, self).__init__()
        
        # social attention net
        self.SA = SRNN(args)
        self.stgraph = ST_GRAPH(1, args.seq_length + 1)
        self.w = args.w
        self.h = args.h

        self.emb = nn.Sequential(
            nn.Linear(in_features=args.feat_dim*2, out_features=args.feat_dim),
            nn.Tanh()
        )
        self.softmax = nn.Softmax()
        
    def forward(self, query_encode, gallery_encode, gallery_label, x_batch, y_batch):
        x = self.stgraph.readGraph(x_batch[0])
        nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

        numNodes = nodes.size()[1]

        hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
        hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()

        cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size)).cuda()
        cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size)).cuda()

        outputs_sa, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1], hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs)
        
        mux, muy, sx, sy, corr = getCoef(outputs_sa)
        next_x, next_y = sample_gaussian_2d(mux[-2, :, :].data, muy[-2, :, :].data, sx[-2, :, :].data, sy[-2, :, :].data, corr[-2, :, :].data, nodesPresent[args.obs_length-1])

        next_x = torch.unsqueeze(next_x, 1)
        next_y = torch.unsqueeze(next_y, 1)

        pos_feat_q = torch.cat((next_x, next_y), axis=1)
        pos_feat_g = torch.Tensor(y_batch[0])
        pos_feat_g_t = pos_feat_g.t()

        inner_product_pos = torch.matmul(pos_feat_q, pos_feat_g_t)
        att_pos = torch.matmul(inner_product_pos, gallery_label)

        gallery_encode_t = gallery_encode.t()

        inner_product = torch.matmul(query_encode, gallery_encode_t)
        att = torch.matmul(inner_product, gallery_label)
        att_all = att + att_pos
        softmax_att = self.softmax(att_all)

        r = torch.matmul(softmax_att, gallery_encode)
        x = torch.cat((query_encode, r), dim=1)
        outputs = self.emb(x)

        loss_sa = Gaussian2DLikelihood(outputs_sa, nodes[1:], nodesPresent[1:], args.pred_length)
        return outputs, loss_sa


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
    def __init__(self, args):
        super(MatchingNet, self).__init__()

        # embedding network
        self.encoder = ResnetEncoder(args.feat_dim)

        # fully context embedding
        self.CEN_Q = AttentionalEmbed(args)
        self.CEN_G = BidirectionalLSTM(layer_sizes=[int(args.feat_dim/2)], vector_dim=args.feat_dim)
        
        # softmax
        self.classify = Classify()

        # TD_clf
        self.TD_clf = nn.Sequential(
            nn.Linear(in_features=args.feat_dim, out_features=1),
            nn.Sigmoid()
        )
        
    def forward(self, gallery_images, gallery_labels, query_images, x_batch, y_batch):
        
        # gallery embedding

        gallery_encode = self.encoder(gallery_images)
        gallery_encode = torch.unsqueeze(gallery_encode, 0)
        gallery_cen, hn, cn = self.CEN_G(gallery_encode)

        # query embedding
        query_encode = self.encoder(query_images)
        query_cen, outputs_sa = self.CEN_Q(query_encode, gallery_cen, gallery_labels, x_batch, y_batch)

        preds = self.classify(query_cen, gallery_cen, gallery_labels)

        TD_pred = self.TD_clf(gallery_encode)
        
        return preds, TD_pred, outputs_sa









        

