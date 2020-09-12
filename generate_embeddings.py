import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from transformers import BertTokenizer


#Edit these parameters to get text from

PATH_TO_TEXT = ''
EMBEDDING_COLUMN = ''

class JobTitleEncoder(nn.Module):

    def __init__(self, device=None, n_tokens=32):
        super(JobTitleEncoder, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.n_tokens = n_tokens

        self.device = device

    def forward(self, seq, attn_masks):

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask=attn_masks)
        #Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]
        return cls_rep

    def clean_text(self, string):

        tokens = self.tokenizer.tokenize(string)
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        if len(tokens) < self.n_tokens:
            tokens += ['[PAD]' for _ in range(self.n_tokens - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.n_tokens-1] + ['[SEP]'] #Pruning the list to be of specified max length
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)

        return tokens_ids

    def get_contextualized_representation(self, text, device=None):

        if text.__class__.__name__ == 'list' or text.__class__.__name__ == 'Series':
            cleaned_text = [self.clean_text(sentence) for sentence in text]
        elif text.__class__.__name__ == 'str':
            cleaned_text = [self.clean_text(text)]

        if self.device:
            seq = torch.tensor(cleaned_text).to(self.device)
        else:
            seq = torch.tensor(cleaned_text)

        attn_mask = [(sentence != 0).long() for sentence in seq]
        attn_mask = torch.stack(attn_mask)

        if self.device:
            attn_mask = attn_mask.to(self.device)
        return self.forward(seq, attn_mask).cpu().data.numpy()

if __name__ == '__main__':

    if 'encodings' not in os.listdir():
        os.mkrid('encodings')

    has_gpu = True if torch.cuda.device_count() > 0 else False
    device = torch.device('cuda') if has_gpu else torch.device('cpu')
    num_devices = torch.cuda.device_count()

    model = JobTitleEncoder(device=device, n_tokens=256)

    if has_gpu:
        if num_devices > 1:
            model = torch.nn.DataParallel(model)
            model.to(device)
            print(f'Using {num_devices} GPUs')
        else:
            model.cuda()
    model.eval()

    embedding_text = pd.read_csv(PATH_TO_TEXT)

    #My GPU can generate 14 embeddings simultaneously on 256 tokens - you may want to set this to
    #to something else depending on your GPU memory.
    num_text_splits = embedding_text.shape[0]//14 + 1

    embedding_text_split = np.array_split(embedding_text, num_text_splits)

    for i, batch in enumerate(job_titles):

        text_encoding = model.get_contextualized_representation(batch[EMBEDDING_COLUMN])

        text_encoding = [str(list(x)) for x in text_encoding]


        batch['SerializedEncoding'] = job_titles_encoding
        batch.to_csv(f'./encodings/{i}.csv', index=None)

