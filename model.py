import numpy as np
import torch
import torch.nn as nn
import torchvision 
import torch.nn.functional as f


class DetrDecoder(nn.Module):
    '''
    Detr decoder 
    '''
    def __init__(self,number_of_embed=16,
                embed_dim=128,
                nhead = 32,
                numlayers = 3
                ) -> None:
        super(DetrDecoder,self).__init__()
        self.query = nn.Embedding(number_of_embed,embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead,batch_first=True)
        self.decoder  = nn.TransformerDecoder(decoder_layer, num_layers=numlayers)
    
    def forward(self,encoder_out):
        bs = encoder_out.size(0)
        decoder_out = self.decoder(self.query.weight.unsqueeze(0).repeat(bs, 1, 1),encoder_out)
        decoder_out = self.fc1(decoder_out)
        return decoder_out

class Keymodel(nn.Module):
    '''
    key model helps to get the dims of the encoder output to 
    decoder embedding dims
    '''
    def __init__(self):
        self.conv_1 = nn.Conv2d(2048,512,1)
        self.conv_2 = nn.Conv2d(512,64,1)
        self.fc_encoder = nn.Linear(49,128)
    
    def forward(self,encoder_output):
        bs = encoder_output.size(0)
        encoder = self.conv_2(self.conv_1(encoder_output))
        encoder = self.fc_encoder(encoder.reshape(bs,64,-1))
        return encoder



class DETR(nn.Module):
    '''
    model class for detr class
    '''
    def __init__(self,encoder=None,key_model=None,decoder=None):
        '''
        inputs:
            encoder_pre_model :encoder pretrained model 
        '''
        super(DETR, self).__init__()
        self.encoder = encoder
        self.key_model = key_model
        if  self.encoder is None:
            resnet = torchvision.models.resnet50(pretrained=True)
            self.encoder=torch.nn.Sequential(*(list(resnet.children())[:-2]))
            self.key_model = Keymodel()

        # using the pretrained encoder so no grad 
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.decoder = decoder
        if self.decoder is None:
            self.decoder = DetrDecoder()   
    
    def forward(self,img):
        encoder_out = self.encoder(img)
        encoder_key = self.key_model(encoder_out)
        return self.decoder(encoder_key)



class GraspFormer(nn.Module):
    def __init__(self, verbose=False):
        super(GraspFormer, self).__init__()
        self.verbose = verbose
        pass

    def forward(self, x):
        pass