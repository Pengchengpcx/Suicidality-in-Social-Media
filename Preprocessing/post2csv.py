# prerpocessing
import datetime
import csv
import nltk
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer
import nltk
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk import pos_tag
from nltk import WordNetLemmatizer
import string
########################################
## Specify sw_users/controls
dataType = './reddit_posts/controls_'
# dataType = './reddit_posts/sw_users_'

## Because the datafile is so huge, we can separate the data
## into several files, specify start file and end file
start = 21
end = 31
########################################
post_fields = ["post_id", "user_id", "date", "time", "subreddit", "post_title", "post_body", "tokens"]

post_body = ""
def processline(writer, line):
    line_info 	= line.split("\t")

    if len(line_info) == 5 and line_info[0] != '' and line_info[1] != '' and line_info[2] != '':
        # print(line_info)
        post_id 	= line_info[0]
        user_id 	= line_info[1]
        timestamp 	= datetime.datetime.fromtimestamp(int(line_info[2])).strftime('%Y-%m-%d %H:%M:%S')
        [date,time] = timestamp.split(" ")
        subreddit 	= line_info[3]
        post_title 	= line_info[4]
        post_body 	= ""
    elif len(line_info) > 5 and line_info[0] != '' and line_info[1] != '' and line_info[2] != '':
        # print(line_info)
        post_id 	= line_info[0]
        user_id 	= line_info[1]
        timestamp 	= datetime.datetime.fromtimestamp(int(line_info[2])).strftime('%Y-%m-%d %H:%M:%S')
        [date,time] = timestamp.split(" ")
        subreddit 	= line_info[3]
        post_title 	= line_info[4]
        post_body 	= line_info[5]
    else:
        # print(line_info)
        tokens = nltkprocess(line)
        tokens = ','.join(tokens)
        info_vec = []
        info_vec.append(line)
        info_vec.append(tokens)
        return info_vec

    newList = str(post_body) + str(post_title)
    tokens = nltkprocess(newList)
    tokens = ','.join(tokens)

    info_vec = []
    info_vec.append(post_id)
    info_vec.append(user_id)
    info_vec.append(date)
    info_vec.append(time)
    info_vec.append(subreddit)
    info_vec.append(post_title)
    info_vec.append(post_body)
    info_vec.append(tokens)

    return info_vec

def nltkprocess(text):
    """
    :param text: give the raw text of each post
    :return: tokenize, lemmatize, remove stop words and punctuations
    """
    stopwords = set(sw.words('english'))
    punct = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer('[a-zA-Z]\w+')

    def tokenlize(document):
        for token, tag in pos_tag(tokenizer.tokenize(document)):
            token = token.lower()
            if token in stopwords:
                continue

            # If punctuation, ignore token and continue
            if all(char in punct for char in token):
                continue

            # Lemmatize the token and yield
            lemma = lemmatize(token, tag)
            yield lemma

    def lemmatize(token, tag):
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return lemmatizer.lemmatize(token, tag)

    processed_text = [i for i in tokenlize(text)]

    return processed_text


if __name__ == "__main__":
    writeFileName = './controls_' + str(start) + '_' + str(end) + '.csv'
    writeFile = open(writeFileName,'w')
    writer = csv.DictWriter(writeFile, delimiter = ',', fieldnames = post_fields)
    writer.writeheader()
    info_vec = []
    prev_info_vec = []
    for i in range(start, end + 1):
        print(i)
        readFile = 'controls/'+ str(i) + '.posts'
        with open(readFile, mode = 'r') as infile:
            for line in infile:
                info_vec = processline(writer, line)
                if len(info_vec) < 5:
                    prev_info_vec[6] = prev_info_vec[6] + info_vec[0]
                    prev_info_vec[7] = prev_info_vec[7] + info_vec[1]
                else:
                    if(len(prev_info_vec) > 0):
                        writer.writerow({
                            "post_id"   : prev_info_vec[0],
                            "user_id"   : prev_info_vec[1],
                            "date"      : prev_info_vec[2],
                            "time"      : prev_info_vec[3],
                            "subreddit" : prev_info_vec[4],
                            "post_title": prev_info_vec[5],
                            "post_body" : prev_info_vec[6],
                            "tokens"    : prev_info_vec[7]
                        })
                    prev_info_vec = info_vec

            writer.writerow({
                        "post_id"   : prev_info_vec[0],
                        "user_id"   : prev_info_vec[1],
                        "date"      : prev_info_vec[2],
                        "time"      : prev_info_vec[3],
                        "subreddit" : prev_info_vec[4],
                        "post_title": prev_info_vec[5],
                        "post_body" : prev_info_vec[6],
                        "tokens"    : prev_info_vec[7]
                    })
