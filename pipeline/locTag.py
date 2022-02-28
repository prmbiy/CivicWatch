import pandas as pd
import numpy as np
#!pip install huggingface
#Need to install huggingface

import json
from pprint import pprint

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


class locTag():
  nlp = []

  def getLocModel():
      tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
      model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
      nlp = pipeline("ner", model=model, tokenizer=tokenizer)
  
  def onlyLOC(x):
      str1 = nlp(x)
      str2=[]
      for j in str1:
          if j["entity"][-3:] == "LOC":
            str2.append(j)
      return str2

  def joinLOC(self, x):
      str1 = self.onlyLOC(x)
      str2 = []
      for j in str1:
          if j["word"][0:2]=="##":
            str2[-1]["word"] += j["word"][2:]
          else:
            str2.append(j)
      return str2

  def locNames(self, x):
      str1 = self.onlyLOC(x)
      str2 = []
      for j in str1:
          if j["word"][0:2]=="##":
            str2[-1] += j["word"][2:]
          else:
            str2.append(j["word"])
      return np.unique(np.array(str2))

  def intersection(self, lst1, lst2):
      temp = set(lst2)
      lst3 = [value for value in lst1 if value in temp]
      return lst3

  def getLocTags(self, data):
    tagList = []
    for index, row in data.iterrows():
      headTags = self.locNames(row['articleHead'])
      bodyTags = self.locNames(row['articleBody'])
      if (headTags.size==0):
        tagList.append(bodyTags.tolist())
      else:
        tagList.append(self.intersection(headTags, bodyTags).tolist())

    return tagList