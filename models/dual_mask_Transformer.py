import torch
import torch.nn as nn
import numpy as np
from .transformer_layers import SelfAttnLayer
from .backbone import Backbone
from .csra import MHA
import math

def mask_replace(tensor,on_neg_1,on_zero,on_one):
    res = tensor.clone()
    res[tensor==-1] = on_neg_1
    res[tensor==0] = on_zero
    res[tensor==1] = on_one
    return res

def weights_init(module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        stdv = 1. / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class dual_mask_Transformer(nn.Module):
    def __init__(self ,num_labels ,label_mask, layers=3, heads=4, dropout=0.1, no_x_features=False):
        super(dual_mask_Transformer ,self).__init__()
        torch.manual_seed(1000)
        self.label_mask = label_mask

        self.no_x_features = no_x_features  # (for no image features)

        # ResNet backbone
        self.backbone = Backbone()
        hidden = 2048  # this should match the backbone output feature size


        self.conv_downsample = torch.nn.Conv2d(hidden ,hidden ,kernel_size=2 ,stride=2)

        # Label Embeddings
        self.pretreatment = torch.Tensor(np.arange(num_labels)).view(1 ,-1).long()
        self.Embedding1 = torch.nn.Embedding(num_labels ,hidden ,padding_idx=None)

        # State Embeddings
        self.Embedding2 = torch.nn.Embedding(3 ,hidden ,padding_idx=0)
        self.transformer_dim = 2048

        # Transformer
        self.Transformer = nn.ModuleList([SelfAttnLayer(hidden ,heads ,dropout) for _ in range(layers)])

        #Classification
        self.linear = torch.nn.Linear(hidden ,num_labels)
        self.CSRA = MHA(1 ,0.1 , self.transformer_dim ,num_labels)


        # self.sig=F.sigmoid()

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.Embedding1.apply(weights_init)
        self.Embedding2.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.Transformer.apply(weights_init)
        self.linear.apply(weights_init)

    def forward(self ,images ,mask):

        lables = self.pretreatment.repeat(images.size(0) ,1).cuda()

        init_label_embeddings = self.Embedding1(lables)

        features = self.backbone(images)

        self.f_backbone = features
        output_CSRA = self.CSRA(features)

        features = features.view(features.size(0) ,features.size(1) ,-1).permute(0 ,2 ,1)

        if self.label_mask:
            label_mask_embeddings = mask_replace(mask ,0 ,1 ,2).long()

            state_embeddings = self.Embedding2(label_mask_embeddings)

            init_label_embeddings += state_embeddings


        embeddings = torch.cat((features ,init_label_embeddings) ,1)

        embeddings = self.LayerNorm(embeddings)


        #attns = []

        for layer in self.Transformer:
            embeddings,attn = layer(embeddings,mask=None)
            #attns += attn.detach().unsqueeze(0).data


        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]
        output_Trans = self.linear(label_embeddings)
        output_Trans = (output_Trans*(torch.eye(output_Trans.size(1)).unsqueeze(0).repeat(output_Trans.size(0),1,1).cuda())).sum(-1)


        return output_Trans ,output_CSRA


    def get_fea(self):

        return self.f_backbone