# visualization of the SuicideWatch Posts in Post 16 - Post 25
import pandas as pd
import argparse
import glob
import csv
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

'''
Generate the corresponding language features given the files
'''
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def readallposts(files):
    '''
    read all tokens of all posts for the given type of the user files
    '''
    words = []
    stopwords = {'http', 'com', 'www', 'app', 'reddit', 'ift', 'draven', 'maria', 'imgur', 'br', 'gt', ''}

    for file in files:
        with open(file, 'r') as rfile:
            reader = csv.DictReader(rfile)
            for row in reader:
                post = row['tokens'].split(',')
                post = [x for x in post if x not in stopwords]
                ppost = VNfilter(post)
                words.append(' '.join(ppost))

    return words

def top_unigrams(documents, top=25):
    '''
    Given the tokens, return the top unigrams based on the frequency
    '''
    tokens = []
    for doc in documents:
        tokens += doc.split(' ')
    ngrams_list = ngrams(tokens, n=1, pad_right=True)
    ngrams_dict = nltk.FreqDist(ngrams_list)
    top_freq = ngrams_dict.most_common(top)

    print('***Top frequent unigrams***\n')
    print(top_freq)

def top_bigrams(documents, top=25):
    '''
    Given the tokens, return the top bigrams based on chi_square, frequency and PMI
    '''
    tokens = []
    for doc in documents:
        tokens += doc.split(' ')
    finder = nltk.BigramCollocationFinder.from_words(tokens)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder.apply_freq_filter(200)
    chi_top = finder.nbest(bigram_measures.chi_sq,top)
    pmi_top = finder.nbest(bigram_measures.pmi,top)
    freq_top = finder.nbest(bigram_measures.raw_freq, top)

    print('***Top frequent bigrams***\n')
    print('Chi square measure:',chi_top,'\n')
    print('PMI square measure:',pmi_top,'\n')
    print('Frequency square measure:',freq_top,'\n')


def ldamodel(documents, topics=10, topwords=15):
    '''
    Given the a list of documents, each document is a processed string
    return the topic distributions
    '''
    assert isinstance(documents, list)
    assert isinstance(documents[0], str)

    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=None,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    lda = LatentDirichletAllocation(n_components=topics, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, topwords)

def VNfilter(tokens):
    '''
    Given a string, filter out the nouns and verbs, return the new string
    '''
    assert isinstance(tokens,list)
    vn = []
    for word, tag in nltk.pos_tag(tokens):
        if tag == 'NN' or tag == 'VB':
            vn.append(word)

    return vn


if __name__ == "__main__":

    '''
    Visulize the features for control and sw users
    1. Compare the top ungrams of both users
    2. Compare the top bigrams of both users based on Chi_square, freq and PMI
    3. Compare the Topic models of two users
    '''

    swuser = glob.glob('./Suicide*' + '.csv')
    controluser = glob.glob('./controls*' + '.csv')
    swuser2 = glob.glob('./sw_users*' + '.csv')

    swwords = readallposts(swuser)
    conwords = readallposts(controluser)
    swwords2 = readallposts(swuser2)

    print ('*************Swuser Analysis***********')
    top_uni = top_unigrams(swwords)
    top_bi = top_bigrams(swwords)

    print('*************Control Analysis***********')
    top_uni = top_unigrams(conwords)
    top_bi = top_bigrams(conwords)

    print ('*************Swuser2 Analysis***********')
    top_uni = top_unigrams(swwords2)
    top_bi = top_bigrams(swwords2)

    ldamodel(swwords)
    ldamodel(swwords2)

    df = pd.read_csv('sw_users_0_12.csv')
    suicideWatch_df = df.loc[df['subreddit'] == 'SuicideWatch']
    suicideWatch_df.to_csv('SuicideWatch1.csv')

    test = ['i','']
    print (VNfilter(test))

