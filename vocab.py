from cProfile import label
from itertools import count
import torch
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter
from collections import OrderedDict
from tqdm import tqdm 



def create_vocab(reviews):

    
    
    
    
    pass




def main(file):

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
    default_index = -1
    voc = vocab(od,min_freq=1, specials=[unk_token])
    voc.set_default_index(default_index)
    


    

    
    






    

if __name__ == "__main__":
    
    voc = torch.load('vocab.pt')
    print('b')
    main('train.ft.txt')