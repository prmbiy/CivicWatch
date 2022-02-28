
# Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora

# NLTK
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk

# Other imp modules
import pandas as pd
from datetime import date, datetime


stemmer = SnowballStemmer("english")
LDA = gensim.models.ldamodel.LdaModel

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result=[]
    for token in simple_preprocess(text) :
        if token not in STOPWORDS and len(token) > 1:
            result.append(lemmatize_stemming(token))
          
    return result

def preproc(data):

    proc_data = []
    for index, row in data.iterrows():
        try:
            proc_data.append(' '.join(preprocess(row["articleHead"])))
        except Exception:
            continue
    return pd.DataFrame(proc_data, columns=['clean_text'])

def getBigram_pmi(clean_data):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_documents([comment.split() for comment in clean_data.clean_text])
    finder.apply_freq_filter(50)
    bigram_scores = finder.score_ngrams(bigram_measures.pmi)

    bigram_pmi = []
    if(len(bigram_scores)):
        bigram_pmi = pd.DataFrame(bigram_scores)
        bigram_pmi.columns = ['bigram', 'pmi']
    return bigram_pmi

def getTrigram_pmi(clean_data):
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = nltk.collocations.TrigramCollocationFinder.from_documents([comment.split() for comment in clean_data.clean_text])
    # Filter only those that occur at least 50 times
    finder.apply_freq_filter(50)
    trigram_scores = finder.score_ngrams(trigram_measures.pmi)

    trigram_pmi = []
    if(len(trigram_scores)):
        trigram_pmi = pd.DataFrame(trigram_scores)
        trigram_pmi.columns = ['trigram', 'pmi']
        trigram_pmi.sort_values(by='pmi', axis = 0, ascending = False, inplace = True)
    return trigram_pmi

# Filter for bigrams with only noun-type structures
def bigram_filter(bigram):
    tag = nltk.pos_tag(bigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['NN']:
        return False
    if bigram[0] in nltk.corpus.stopwords.words('english') or bigram[1] in nltk.corpus.stopwords.words('english'):
        return False
    if 'n' in bigram or 't' in bigram:
        return False
    if 'PRON' in bigram:
        return False
    return True

# Filter for trigrams with only noun-type structures
def trigram_filter(trigram):
    tag = nltk.pos_tag(trigram)
    if tag[0][1] not in ['JJ', 'NN'] and tag[1][1] not in ['JJ','NN']:
        return False
    if trigram[0] in nltk.corpus.stopwords.words('english') or trigram[-1] in nltk.corpus.stopwords.words('english') or trigram[1] in nltk.corpus.stopwords.words('english'):
        return False
    if 'n' in trigram or 't' in trigram:
         return False
    if 'PRON' in trigram:
        return False
    return True 

def get_bi_tri_grams(bigram_pmi, trigram_pmi):
    bigrams = []
    trigrams = []
    if(len(bigram_pmi)):
        filtered_bigram = bigram_pmi[bigram_pmi.apply(lambda bigram:
                                                bigram_filter(bigram['bigram'])
                                                and bigram.pmi > 5, axis = 1)][:500]
        bigrams = [' '.join(x) for x in filtered_bigram.bigram.values if len(x[0]) > 2 or len(x[1]) > 2]
    if(len(trigram_pmi)):
        filtered_trigram = trigram_pmi[trigram_pmi.apply(lambda trigram: 
                                                    trigram_filter(trigram['trigram'])
                                                    and trigram.pmi > 5, axis = 1)][:500]
        trigrams = [' '.join(x) for x in filtered_trigram.trigram.values if len(x[0]) > 2 or len(x[1]) > 2 and len(x[2]) > 2]
    
    return bigrams, trigrams

# Concatenate n-grams
def replace_ngram(x, bigrams, trigrams):
    for gram in trigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    for gram in bigrams:
        x = x.replace(gram, '_'.join(gram.split()))
    return x

def get_ngram(clean_data, bigrams, trigrams):
    clean_text_w_ngrams = clean_data.copy()
    clean_text_w_ngrams.clean_text = clean_text_w_ngrams.clean_text.map(lambda x: replace_ngram(x, bigrams, trigrams))
    clean_text_w_ngrams = clean_text_w_ngrams.clean_text.map(lambda x: [word for word in x.split()
                                                    if word not in nltk.corpus.stopwords.words('english')
                                                                and len(word) > 2])
    return clean_text_w_ngrams

# Filter for only nouns
def noun_only(x):
    pos_comment = nltk.pos_tag(x)
    filtered = [word[0] for word in pos_comment if word[1] in ['NN']]
    # to filter both noun and verbs
    #filtered = [word[0] for word in pos_comment if word[1] in ['NN','VB', 'VBD', 'VBG', 'VBN', 'VBZ']]
    return filtered

def getTopic(cluster_data, original_data):

    # Preprocess - 1
    clean_data = preproc(cluster_data)

    # Preprocess - 2
    bigram_pmi = getBigram_pmi(clean_data)
    trigram_pmi = getTrigram_pmi(clean_data)

    bigrams, trigrams = get_bi_tri_grams(bigram_pmi, trigram_pmi)
    clean_text_w_ngrams = get_ngram(clean_data, bigrams, trigrams)

    final_clean_data = clean_text_w_ngrams.map(noun_only)

    # MODEL 
    dictionary = corpora.Dictionary(final_clean_data)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in final_clean_data]

    ldamodel = LDA(doc_term_matrix, num_topics=1, id2word = dictionary, passes=50,
               iterations=400,  chunksize = 50, eval_every = None, random_state=0)
    
    topics = ldamodel.show_topics(1, num_words=4, formatted=False)[0][1]

    topic_list = []
    for topic in topics:
        topic_list.append(str(topic[0]))
    
    topic = ','.join(topic_list)
    for idx in list(cluster_data.index):
        original_data.loc[original_data['_id'] == cluster_data['_id'][idx], 'topic'] = topic


def getTopics(clusters, original_data):
    for cluster_label in list(clusters['label'].unique()):
        cluster = clusters[clusters['label'] == cluster_label]
        getTopic(cluster, original_data)
