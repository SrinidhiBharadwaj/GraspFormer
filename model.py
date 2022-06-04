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
        return decoder_out

class Keymodel(nn.Module):
    '''
    key model helps to get the dims of the encoder output to 
    decoder embedding dims
    '''
    def __init__(self):
        super(Keymodel,self).__init__()
        self.conv_1 = nn.Conv2d(2048,512,1)
        self.conv_2 = nn.Conv2d(512,64,1)
        self.fc_encoder = nn.Linear(49,128)

    def forward(self,encoder_output):
        bs = encoder_output.size(0)
        encoder = self.conv_2(self.conv_1(encoder_output))
        encoder = self.fc_encoder(encoder.reshape(bs,64,-1))
        return encoder
    
    # def forward(self,encoder_output):
    #     bs = encoder_output.size(0)
    #     encoder = self.conv_2(self.conv_1(encoder_output))
    #     encoder = encoder.reshape(bs,64,-1).permute(0,2,1).contiguous()
    #     encoder = self.fc_encoder(encoder)
    #     return encoder

class Decoder_head(nn.Module):
    '''
    shared linear layer used to decode the transform embeddings
    '''
    def __init__(self,input_dim,num_of_class,num_point) -> None:
        super(Decoder_head,self).__init__()
        self.fc1 = nn.Linear(input_dim,64)
        self.fc2 = nn.Linear(64,16)
        self.bound = nn.Linear(16,num_point)
        self.classification = nn.Linear(16,num_of_class)
        self.orientation = nn.Linear(16,20)
        self.relu  = nn.ReLU()
    
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        bound = self.bound(x).sigmoid()
        classification = self.classification(x)
        orientation = self.orientation(x)
        return (bound,classification)

class DETR(nn.Module):
    '''
    model class for detr class
    '''
    def __init__(self,encoder=None,key_model=None,decoder=None,
                num_class=20,num_point=4,embed_dim=128,nhead=4,
                numlayers=1,number_of_embed=16):
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
            self.decoder = DetrDecoder(number_of_embed=number_of_embed,
                embed_dim=embed_dim,
                nhead = nhead,
                numlayers = numlayers) 
            self.head = Decoder_head(input_dim=embed_dim,num_of_class=num_class,num_point=num_point)
    
    def forward(self,img):
        encoder_out = self.encoder(img)
        encoder_key = self.key_model(encoder_out)
        decoder_out = self.decoder(encoder_key)
        return self.head(decoder_out)



class GraspFormer(nn.Module):
    def __init__(self, verbose=False):
        super(GraspFormer, self).__init__()
        self.verbose = verbose
        pass

    def forward(self, x):
        pass