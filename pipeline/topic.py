#install gensim

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pandas as pd
stemmer = SnowballStemmer("english")
from tqdm import tqdm
import json
from pprint import pprint
import nltk
nltk.download('wordnet')
import datetime
from datetime import date

class topic():
    def loadData(self, filename):
        data = pd.read_csv(filename)
        return data

    def lemmatize_stemming(self, text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(self, text):
        result=[]
        for token in gensim.utils.simple_preprocess(text) :
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
                
        return result

    def preproc(self, data, date):

        proc_data = []
        for index, row in data.iterrows():
            if (datetime.datetime.strptime(row.publicationDate[5:16], '%d %b %Y') >= datetime.datetime.strptime(date, '%d %b %Y')):
            try:
                proc_data.append(self.preprocess(row["articleBody"]))
            except Exception:
                continue
        return proc_data

    def filemaker(self, proc_data, date):
        dictionary = gensim.corpora.Dictionary(proc_data)
        bow_corpus = [dictionary.doc2bow(doc) for doc in proc_data]
        lda_model =  gensim.models.LdaMulticore(bow_corpus, num_topics = 5, id2word = dictionary, passes = 10, workers = 2)

        top5 = ""
        aftertext = "\nAfter " + date + ",\n"
        
        for idx, topic in lda_model.print_topics(-1):
            #print("Topic: {} \nWords: {}".format(idx, topic ))
            top5 += str("Topic: {} \nWords: {}".format(idx, topic )) + "\n"
            #print("\n")

        text_file = open("top5.txt", "a+")
        n = text_file.write(aftertext)
        n = text_file.write(top5)
        text_file.close()
