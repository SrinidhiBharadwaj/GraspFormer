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
        return (bound,classification)

class DETR(nn.Module):
    '''
    model class for detr class
    '''
    def __init__(self,encoder=None,key_model=None,decoder=None,
                num_class=20,num_point=4,embed_dim=128,nhead=32,
                numlayers=3,number_of_embed=16):
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




class detr_simplified(nn.Module):
    def __init__(self, num_classes, embed_dim=64, nhead=2,
                    num_encoders=2, num_decoders=2):
        
        super(detr_simplified, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = embed_dim
        self.nheads = nhead
        self.rot_classes = 20 #Set for cornell dataset

        self.encoder = torchvision.models.resnet18(pretrained=True)
        del self.encoder.fc

        for params in self.encoder.parameters():
            params.requires_grad = False #Do not train the encoder
        
        self.conv1 = nn.Conv2d(512, self.hidden_dim, kernel_size=1)

        self.transformer = nn.Transformer(d_model=self.hidden_dim, nhead=self.nheads, num_encoder_layers=num_encoders, \
                                          num_decoder_layers=num_decoders)
        
        self.row_pos_embed = nn.Parameter(torch.rand(50, self.hidden_dim // 2))
        self.col_pos_embed = nn.Parameter(torch.rand(50, self.hidden_dim // 2))
        self.query_pos_embed = nn.Parameter(torch.rand(8, self.hidden_dim))

        self.linear_bbox = nn.Linear(self.hidden_dim, 4)
        self.linear_class = nn.Linear(self.hidden_dim, num_classes)


    def forward(self, x, orientation_only=False):
        original_tensor = x
        bbox = None
        output = {}
        if not orientation_only:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)

            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)

            x = self.conv1(x)
    
            #Positional embeddings
            height, width = x.shape[-2:]
            
            embedding = torch.cat([self.col_pos_embed[:width].unsqueeze(0).repeat(height, 1, 1),
                self.row_pos_embed[:height].unsqueeze(1).repeat(1, width, 1)], dim=-1).flatten(0, 1).unsqueeze(1)

            enc_in = embedding + 0.1 * x.flatten(2).permute(2, 0, 1)
            x = self.transformer(enc_in,
                                self.query_pos_embed.unsqueeze(1).repeat(1, enc_in.size(1), 1)).transpose(0, 1)

            bbox = self.linear_bbox(x).sigmoid() * 224.
            rotation = self.linear_class(x)
        
        output["pred_boxes"] = bbox
        output["pred_logits"] = rotation
        return output

class DETRModel(nn.Module):
    def __init__(self,num_classes,num_queries):
        super(DETRModel,self).__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        
        self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        self.in_features = self.model.class_embed.in_features
        
        self.model.class_embed = nn.Linear(in_features=self.in_features,out_features=self.num_classes)
        self.model.num_queries = self.num_queries
        
    def forward(self,images):
        return self.model(images)
 

class GraspFormer(nn.Module):
    def __init__(self, verbose=False):
        super(GraspFormer, self).__init__()
        self.verbose = verbose
        pass

    def forward(self, x):
        pass