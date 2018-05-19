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

crowd_FileName = './reddit_annotation/crowd.csv'
expert_FileName = './reddit_annotation/expert.csv'
vocab_FileName = './volcabulary.csv'

read_sw_users_FileNames = [
	'./reddit_posts/sw_users_0_12.csv', 
	'./reddit_posts/sw_users_13_25.csv'
]
this_user_id = -33333333
post_fields = ["user_id", "post_body", "tokens", "label"]
def nltkprocess(textArray):
	"""
	:param text: give the raw text of each post
	:return: tokenize, lemmatize, remove stop words and punctuations
	"""
	stopwords = set(sw.words('english'))
	punct = set(string.punctuation)
	lemmatizer = WordNetLemmatizer()
	tokenizer = RegexpTokenizer('[a-zA-Z]\w+')

	def tokenlize(document):
		for token in nltk.word_tokenize(document):
			token = token.lower()
			if token not in volcabularyList:
				continue

			# If punctuation, ignore token and continue
			if all(char in punct for char in token):
				continue
			# Lemmatize the token and yield
			# lemma = lemmatize(token, tag)
			yield token

	processed_text = [i for i in tokenlize(textArray)]

	return processed_text

if __name__ == "__main__":
	# get all the user_id from crowd.csv
	crowdDict = {}
	with open(crowd_FileName) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			user_id = int(row['user_id'])
			label = row['label']
			crowdDict[user_id] = label

	expertDict = {}
	with open(expert_FileName) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			user_id = int(row['user_id'])
			label = row['label']
			expertDict[user_id] = label

	volcabularyList = []
	with open(vocab_FileName) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			word = row['word']
			volcabularyList.append(word)

	writeFileName_expert = 'sw_users_data_expert.csv'
	writeFile_expert = open(writeFileName_expert,'w')
	writer_expert = csv.DictWriter(writeFile_expert, delimiter = ',', fieldnames = post_fields)
	writer_expert.writeheader()

	writeFileName_crowd = 'sw_users_data_crowd.csv'
	writeFile_crowd = open(writeFileName_crowd,'w')
	writer_crowd = csv.DictWriter(writeFile_crowd, delimiter = ',', fieldnames = post_fields)
	writer_crowd.writeheader()

# experts
	for readName in read_sw_users_FileNames:
		print(readName)
		with open(readName) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				user_id 	= int(row['user_id'])
				if expertDict.get(user_id):
					print('user_id: ',user_id)
					# empty the buff and record data for the new user
					post_title = row['post_title']
					post_body = row['post_body']
					post = post_title + ' ' + post_body
					post_info = nltkprocess(post)
					save_post_info = ','.join(post_info)
					writer_expert.writerow({
						"user_id"   : user_id,
                        "post_body" : post,
                        "tokens"    : save_post_info,
                        "label"		: int(expertDict[user_id])
	                })
	                
# crowd
	for readName in read_sw_users_FileNames:
		print(readName)
		with open(readName) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				user_id 	= int(row['user_id'])
				if crowdDict.get(user_id):
					print('user_id: ',user_id)
					# empty the buff and record data for the new user
					post_title = row['post_title']
					post_body = row['post_body']
					post = post_title + ' ' + post_body
					post_info = nltkprocess(post)
					save_post_info = ','.join(post_info)
					writer_crowd.writerow({
						"user_id"   : user_id,
                        "post_body" : post,
                        "tokens"    : save_post_info,
                        "label"		: int(crowdDict[user_id])
	                })
					

		