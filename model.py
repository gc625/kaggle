import torch
import torch.nn.functional as F
import torch.nn as nn



class BasicTransformer(nn.Module):


    def __init__(self,vocab_size) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,128,padding_idx=1)

        self.fc1 = nn.Conv1d(128,256,1)
        self.fc2 = nn.Conv1d(256,512,1)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)

        self.mha = nn.MultiheadAttention(embed_dim=512,num_heads=8)
        
        self.fc3 = nn.Conv1d(512,256,1)
        self.fc4 = nn.Conv1d(256,1,1)
        self.bn3 = nn.BatchNorm1d(256)






    def forward(self,x):


        x = self.embedding(x).transpose(2,1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))

        x = x.transpose(2,1)
        x,_ = self.mha(x,x,x)
        x = x.transpose(2,1)
        x = self.bn3(F.relu(self.fc3(x)))
        x,_ = torch.max(F.relu(self.fc4(x)),dim=-1)


        return x

    


        




        
        



