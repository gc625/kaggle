from cgi import test
from re import T
from torch.utils.data import Dataset
from collections import Counter
from collections import OrderedDict
from torchtext.vocab import vocab
from torchtext.data import get_tokenizer
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class review_dataset(Dataset):


    def __init__(self,file_path):

        f = open(file_path,"r")
        lines = f.readlines()
        tokenizer = get_tokenizer("basic_english")

        self.samples = lines
        self.max_length = 200
        self.sentence_to_token_idx = lambda x: voc(tokenizer(x))        

        # for line in tqdm(lines):
        #     curr = line.split(" ",1)
        #     labels += [int(curr[0][-1])-1]
        #     words = tokenizer(curr[1])
        
        #     longest_len = max(longest_len,len(words))
        #     reviews += [words]


        
        # assert(len(labels) == len(reviews)), "size mismatch"




    def __len__(self):
        
        return len(self.samples)

        


    def __getitem__(self,idx):
        
        current_sample = self.samples[idx].split(" ",1)
        review = current_sample[1]
        label = int(current_sample[0][-1])-1
        # tokenizer = get_tokenizer("basic_english")
        return review,label

    @staticmethod
    def create_vocab(file):

        f = open(file, "r")
        lines = f.readlines()
        labels = []
        reviews = []
        counter = Counter()
        for line in tqdm(lines):
            curr = line.split(" ",1)

            labels += [int(curr[0][-1])-1]
            review = curr[1]
            
            tokenizer = get_tokenizer('basic_english')
            words = tokenizer(review)
            counter.update(words)
            
        
        f.close()
        
        od = OrderedDict(counter)
        unk_token = '<unk>'
        pad_token = "<pad>"
        default_index = 0
        voc = vocab(od,min_freq=1, specials=[unk_token,pad_token],special_first= True) 
        voc.set_default_index(default_index)    
        torch.save(voc,"vocab.pt")


    def collate_batch(self, batch):
        
        label_list, text_list, = [], []
        
        for (_text,_label) in batch:
            label_list.append(_label)
            processed_text = torch.tensor(self.sentence_to_token_idx(_text), dtype=torch.int64)[:self.max_length]
            text_list.append(processed_text)
        
        label_list = torch.tensor(label_list, dtype=torch.int64)
        
        text_list = pad_sequence(text_list, batch_first=True, padding_value=1)
        
        return text_list,label_list,



if __name__ == "__main__":

    # review_dataset.create_vocab('train.ft.txt')
    # testset = review_dataset("train.ft.txt")
    # # testset.create_vocab("train.ft.txt")
    # voc = torch.load('vocab.pt')
    # dataloader = DataLoader(testset, batch_size=16, collate_fn=testset.collate_batch,shuffle=True)
    
    # from model import BasicTransformer
    # transformer = BasicTransformer(len(testset))
    # optimizer = optim.SGD(transformer.parameters(), lr=0.001, momentum=0.9)

    # loss_fn = nn.BCEWithLogitsLoss()
    

    # with tqdm(dataloader, unit="batch") as tepoch:
    #     for x,y in tepoch:
    #         optimizer.zero_grad()
            
    #         y = y.unsqueeze(1).float()
    #         # y.to('cuda')
    #         # x.to('cuda')
    #         # transformer.to('cuda')
            
    #         x = transformer(x)


    #         loss = loss_fn(x,y)
    #         loss.backward()
    #         optimizer.step()
            
    #         tepoch.set_postfix(loss=loss.item())



    #     # print(x,"Targets",y,"\n")
        
    # torch.save(transformer.state_dict(), 'testformer.pth')
    loss = nn.CrossEntropyLoss()
    input = torch.randn(1, 3, requires_grad=True)
    target = torch.empty(1, dtype=torch.long).random_(3)
    output = loss(input, target)
    output.backward()