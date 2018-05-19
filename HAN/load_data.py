import csv
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import numpy as np
import sys
import string

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

csv.field_size_limit(sys.maxsize)


def nltkprocess(text):
    """
    :param text: give the raw text of each post
    :return: tokenize, lemmatize, remove stop words and punctuations
    """
    punct = set(string.punctuation)
    tokenizer = WordPunctTokenizer()
    new_text = []

    for token in (tokenizer.tokenize(text)):
        token = token.lower()

        # If punctuation, ignore token and continue
        if all(char in punct for char in token):
            continue

        new_text.append(token)

    return ','.join(new_text)


def preprocess(path):
    '''
    re tokenize
    '''
    fields = ['user_id','post_body','tokens','label']

    f2 = open('./csw.csv','w')
    writer = csv.DictWriter(f2, fieldnames=fields)
    writer.writeheader()

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            words = nltkprocess(row['post_body'])
            writer.writerow({
                "user_id": row['user_id'],
                "post_body"	: row['post_body'],
                "tokens": words,
                "label": row['label']
            })


def vocabulary_index(path):
    '''
    read the vocabulary and return a dictionary {'word':index}
    '''
    vocab = {}
    vocab['unknown'] = 0
    i = 1

    with open(path) as v:
        reader = csv.DictReader(v)
        for row in reader:
            if int(row['frequency']) >5 and len(row['word'])<15:
                vocab[row['word']] = i
                i += 1
        print(i)

    return vocab

def userspost():
    path = './crowd_data_sw_users.csv'
    with open(path) as user:
        i = 1073
        total = []
        labels = [1]
        count = 0
        reader = csv.DictReader(user)
        for row in reader:
            if int(row['user_id']) == i:
                count +=1
            else:
                total.append(count)
                count = 1
                i = int(row['user_id'])
                labels.append(int(row['label']))

    print(sum(total)/len(total))

    sw = []
    for x,y in zip(total, labels):
        if y==1:
            sw.append(x)

    print(sum(sw)/len(sw))

'''
ave post of sw 60
'''
max_posts = 120 #max number of posts per user
max_words = 80 #max number of words per post

def create_data(path, vocab_path):
    vocab = vocabulary_index(vocab_path)

    pcount = 0

    with open(path) as datafile:
        users = []
        user = []
        labels = [0]
        cur_user = -20001 # the 1st user_id in the file
        reader = csv.DictReader(datafile)
        i = 0

        for row in reader:
            if row['user_id'] == 'user_id':
                continue
            print((row['user_id']))
            if int(row['user_id']) == cur_user:
                if i < max_posts:
                    # add the post into this user
                    i += 1
                    user.append(words2index(row['tokens'], vocab))
                else:
                    continue
            else:
                # add the current user info to the users list
                # refresh i
                if len(user) < max_posts:
                    for j in range(max_posts - len(user)):
                        user.append([0]*max_words)
                users.append(user)
                user = []
                cur_user = int(row['user_id'])
                user.append(words2index(row['tokens'], vocab))
                # lab = [0,0]
                # lab[int(row['label'])] = 1
                if int(row['label']) == 1:
                    pcount += 1
                labels.append(int(row['label']))
                i = 1

        if len(user) < max_posts:
            for j in range(max_posts - len(user)):
                user.append([0] * max_words)

        users.append(user)

    pickle.dump((users, labels), open('trainingless_data', 'wb'))

    for i in users:
        print(len(i))

    print(len(labels), len(users), pcount)

def words2index(tokens, vocab):
    words = tokens.split(',')
    words_index = []
    for i in range(max_words):
        if i < len(words):
            words_index.append(vocab.get(words[i],0))
        else:
            words_index.append(0)

    return words_index

path = './trainingdata/trainingless.csv'
vocab_path = './vocabulary.csv'

create_data(path,vocab_path)




