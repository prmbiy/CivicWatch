import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import logging
logging.basicConfig(level=logging.ERROR)
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
'''
!pip install transformers


'''

class categorize():
  def __init__(self, text, MAX_LEN = 128, TEST_BATCH_SIZE = 5, weightpath = '/content/drive/MyDrive/PS/Files/models/pytorch_distilbert_news_v001.bin'):
    if type(text) == list:
      self.textdf = pd.DataFrame()
      self.textdf['text'] = text
    elif isinstance(text, pd.Series):
      self.textdf = pd.DataFrame()
      self.textdf['text'] = text.values
    elif type(text) == str:
      self.textdf = pd.DataFrame()
      self.textdf['text'] = [text]
    
    self.MAX_LEN = MAX_LEN
    self.TEST_BATCH_SIZE = TEST_BATCH_SIZE
    self.modelB = DistilBERTClass()
    self.weightpath = weightpath
    

  def loadfunc(self, modelname = 'distilbert-base-uncased'):
    tokenizer = DistilBertTokenizer.from_pretrained(modelname, truncation=True, do_lower_case=True)
    self.modelB.load_state_dict(torch.load(self.weightpath), strict=False)
    texting_set = preloader2(self.textdf, tokenizer, self.MAX_LEN)

    test_params = {'batch_size': self.TEST_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0}
                
    texting_loader = DataLoader(texting_set, **test_params)
    return texting_loader

  def predicting(self, texting_loader):
    self.modelB.eval()
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(texting_loader):
            ids = data['ids']
            mask = data['mask']
            token_type_ids = data['token_type_ids']
            outputs = self.modelB(ids, mask, token_type_ids)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs

  def predict(self):
    texting_loader = self.loadfunc()
    t_outputs = self.predicting(texting_loader)
    t_final_outputs = [x==np.max(x) for x in t_outputs]
    returnable = []
    for i in t_final_outputs:
      returnable.append(LabelSeries[np.argmax(i) == LabelSeries.Integer].Category.values[0])

    return returnable