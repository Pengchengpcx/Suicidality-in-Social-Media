from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import dill
import csv
import numpy as np
from sklearn.metrics import roc_auc_score
import pickle

class Featurizer():
	def __init__(self, ngrams=2):
		self.df = CountVectorizer( ngram_range=(1,ngrams), stop_words='english')
		self.tfidf = TfidfTransformer()

	def train_feature(self, examples):
		tfidf_train = self.df.fit_transform(examples)
		tfidf_train = self.tfidf.fit_transform(tfidf_train)

		return tfidf_train

	def test_feature(self, examples):
		tfidf_test = self.df.transform(examples)
		tfidf_test = self.tfidf.transform(tfidf_test)

		return tfidf_test

	def show_top10(self, classifier, categories):
		feature_names = np.asarray(self.df.get_feature_names())
		if len(categories) == 2:
			top10 = np.argsort(classifier.coef_[0])[-10:]
			bottom10 = np.argsort(classifier.coef_[0])[:10]
			print("Pos: %s" % " ".join(feature_names[top10]))
			print("Neg: %s" % " ".join(feature_names[bottom10]))
			bad = np.argsort(np.absolute(classifier.coef_[0]))[:11000]
			with open("stopwords.csv", 'w') as stopwords:
				d = feature_names[bad].tolist()
				wr =csv.writer(stopwords)
				for i in d:
					wr.writerow([i,])
		else:
			for i, category in enumerate(categories):
				top10 = np.argsort(classifier.coef_[i])[-10:]
				print("%s: %s" % (category, " ".join(feature_names[top10])))

def recall(predict, label):
	pre = np.asarray(predict)
	lab = np.asarray(label)

	recall = np.sum(lab[pre == 1]) / np.sum(lab)
	precision = np.sum(lab[pre == 1]) / np.sum(pre)

	return recall, precision

def AUC(prediction, labels):
	'''
	:param prediction: [[prob_neg, prob_postive],...]
	:param labels:[[negative_index, positive_index],...]
	:return:
	'''
	score = roc_auc_score(labels, prediction[:,1])

	return score


def training(ngrams):
	with open('./svm_train', 'rb') as f:
		train_x_raw, train_y_raw = pickle.load(f)
	with open('./svm_test', 'rb') as f:
		test_x_raw, test_y_raw = pickle.load(f)

	feature_extractor = Featurizer(ngrams=ngrams)
	x_train = feature_extractor.train_feature(train_x_raw)
	y_train = train_y_raw

	x_test = feature_extractor.test_feature(test_x_raw)
	y_test = test_y_raw

	lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
	weights = np.ones([len(y_train),],)*12 + 1
	lr.fit(x_train, y_train, sample_weight=weights)

	feature_extractor.show_top10(lr, [0,1])

	print('training done...')
	accuracy = lr.score(x_test,y_test)
	prediction = lr.predict(x_test)
	log_predict = lr.predict_log_proba(x_test)
	rec, pre = recall(prediction,y_test)
	auc = AUC(log_predict, y_test)

	print('acc', accuracy, 'recall', rec, 'precision', pre, 'AUC', auc)

if __name__ == "__main__":
	training(1)