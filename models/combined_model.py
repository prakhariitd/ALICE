import models.bert as bert
import models.gpt3 as gpt3
import numpy as np
import models.sourcetrackingdataset as srctrk

# Returns final classification
# 0 if Unclassfieid
# Positive if True
# Negative if False
def classify(tweet):

	model_loc = './bert_model'
	pred1 = bert.classify(tweet, model_loc)

	return np.argmax(pred1), pred1 #only BERT being used

	pred2 = gpt3.classify(tweet)

	thresh_lo = 0.35
	thresh_hi = 0.65
	thresh_mid = 0.5

	pred3 = 0

	if (pred1[0] >= thresh_mid and pred2[0]>= thresh_mid): #both say YES
		pred3 = (pred1[0]+pred2[0])/2
	elif (pred1[1] >= thresh_mid and pred2[1]>= thresh_mid): #both say NO
		pred3 = (-1*(pred1[1]+pred2[1]))/2
	elif (pred1[0] >= thresh_hi and pred2[1] < thresh_hi):
		pred3 = pred1[0]
	elif (pred2[0] >= thresh_hi and pred1[1] < thresh_hi):
		pred3 = pred2[0]
	elif (pred2[1] >= thresh_hi and pred1[0] < thresh_hi):
		pred3 = pred2[1]
	elif (pred1[1] >= thresh_hi and pred2[0] < thresh_hi):
		pred3 = pred1[1]
	else:
		pred3 = 0

	return pred3

# classify("Covid is fake")