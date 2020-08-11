import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

class MatchNet(nn.Module):
    eps = 1e-10
    def __init__(self, keep_prob, \
                 batch_size,
                 num_channels,
                 fce,
                 ):
        super(MatchingNetwork, self).__init__()

        # embedding network
        self.encoder = Encoder()

        # fully context embedding
        if fce:
            self.CEN_Q = AttentionEmbed()
            self.CEN_G = BidirectionalLSTM()
        
        # softmax
        self.classify = Classify()

