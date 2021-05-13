
import torch
import torch.nn as nn
from helper import *


def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    """

    _,C,H,W = content_original.shape
    # The size of both contents will be same as they are activations of same layer
    content_current = content_current.reshape(C,H*W)
    content_original = content_original.reshape(C,H*W)
    loss = torch.sum(torch.square(content_original-content_current))
    loss = content_weight * loss
    return loss



def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    """
    gram = None

    N,C,H,W = features.shape
    features = features.reshape(N*C,H*W)
    gram = torch.mm(features,features.T)
    gram = gram.reshape(N,C,C)
    norm = H*W*C
    gram = gram/norm 

    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    """

    loss = 0
    for i in range(0,len(style_layers)):
      A = gram_matrix(feats[style_layers[i]].clone())
      loss += style_weights[i] * torch.sum(torch.square(style_targets[i]-A))

    return loss

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.

    """

    _,C,H,W = img.shape
    loss_1 = torch.sum(torch.square(img[:,:,:-1,:]-img[:,:,1:,:]))
    loss_2 = torch.sum(torch.square(img[:,:,:,:-1]-img[:,:,:,1:]))
    loss = tv_weight* (loss_1+loss_2)
    return loss
