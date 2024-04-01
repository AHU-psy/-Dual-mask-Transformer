import torch
import torch.nn as nn



class CSRA(nn.Module): # one basic block 
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA, self).__init__()
        #print(T)
        self.T = T      # temperature       
        self.lam = lam  # Lambda                        
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        # x (B d H W)
        # normalize classifier
        # score (B C HxW)
        #print(x.shape)
        # print(self.head(x).shape)
        # print(torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0,1).shape)
        score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0,1)
        #print(score.shape)
        score = score.flatten(2)
        #print(score.shape)
        base_logit = torch.mean(score, dim=2)
        #print(base_logit.shape)
        if self.T == 99: # max-pooling
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            #print(score_soft.shape)
            att_logit = torch.sum(score * score_soft, dim=2)
            #print(att_logit.shape)

        return base_logit + self.lam * att_logit

    


class MHA(nn.Module):  # multi-head attention
    temp_settings = {  # softmax temperature settings
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MHA, self).__init__()
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.ModuleList([
            CSRA(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        #print(x.shape)
        for head in self.multi_head:
            #print(head)
            logit += head(x)
            #print(logit.shape)
        return logit
