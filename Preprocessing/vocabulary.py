# get the volcabulary (freq >= 5)
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
from operator import itemgetter 

read_FileNames = [
	'reddit_posts/controls_1_10.csv', 
	'reddit_posts/controls_11_20.csv', 
	'reddit_posts/controls_21_31.csv',
	'reddit_posts/sw_users_0_12.csv', 
	'reddit_posts/sw_users_13_25.csv'
]
post_fields = ['word','frequency']
volcabularyDict = {}

def nltkprocess(text):
	"""
	:param text: give the raw text of each post
	:return: tokenize, lemmatize, remove stop words and punctuations
	"""
	stopwords = set(sw.words('english'))
	punct = set(string.punctuation)
	tokenizer = RegexpTokenizer('[a-zA-Z]\w+')

	def tokenlize(document):
		for token, tag in pos_tag(tokenizer.tokenize(document)):
			token = token.lower()
			# If punctuation, ignore token and continue
			if all(char in punct for char in token):
				continue
			if(volcabularyDict.get(token)):
				volcabularyDict[token] += 1
			else:
				volcabularyDict[token] = 1

	tokenlize(text)

if __name__ == "__main__":
	for readName in read_FileNames:
		print(readName)
		with open(readName) as csvfile:
			reader = csv.DictReader(csvfile)
			thisPost = ""
			for row in reader :
				post_title 	= row['post_title']
				post_body 	= row['post_body']
				thisPost += post_title + ' ' + post_body
			nltkprocess(thisPost)

	writeFileName = 'volcabulary.csv'
	writeFile = open(writeFileName,'w')
	writer = csv.DictWriter(writeFile, delimiter = ',', fieldnames = post_fields)
	writer.writeheader()

	discardFileName = 'volcabulary_discard.csv'
	discardFile = open(discardFileName,'w')
	discardwriter = csv.DictWriter(discardFile, delimiter = ',', fieldnames = post_fields)
	discardwriter.writeheader()

	for key, value in sorted(volcabularyDict.items(), key = itemgetter(1), reverse = True):	
		if(value >= 5):
			writer.writerow({
						"word"   		: key,
						"frequency"   	: value
			})
		else:
			discardwriter.writerow({
						"word"   		: key,
						"frequency"   	: value
			})
