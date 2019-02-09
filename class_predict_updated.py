import itertools
import numpy as np
import re
import csv
from sklearn.svm import SVC

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


DATA_FILE = 'data_svm_org_new.csv'

TRAINING_FILE_HAS_HEADER = True
TESTING_FILE_HAS_HEADER = False


# METHOD INDICES
NOISY = 'noisy'
KLT = 'klt'
KLT_JABLOUN = 'klt_jabloun'
STSA = 'stsa'
LOGTSA = 'logstsa'
LOGSTA_NEST = 'logstsa_nest'
LOGSTA_SAP_Q = 'logstsa_sap_q'
WEUCLID = 'weuclid'
RDC = 'rdc'
RDC_NEST = 'rdc_nest'
MB = 'mb'
WAVTHRE = 'wavthre'
SCALART = 'scalart'
TSOUKALAS = 'tsoukalas'

# Samples Names
SAMPLE_NAMES = ['sp01', 'sp02', 'sp03', 'sp04', 'sp06', 'sp07', 'sp08', 'sp09', 'sp11', 'sp12', 'sp13', 'sp14', 'sp16', 'sp17', 'sp18', 'sp19']

CLASS_LABEL = {}
class_index = -1


# Function definition
def getIndex (s):
	global class_index
	if s not in CLASS_LABEL.keys():
		class_index = class_index + 1
		CLASS_LABEL[s] = class_index
	#print ('Class ' + str(s) + ' Index : ' + str(CLASS_LABEL[s]) )
	return CLASS_LABEL[s]


def splitData (dataFile, trainFile, testFile, testSamples):

	train = []
	test = []
	
	for line in csv.reader((open(dataFile))):
		matchedTestSample = False
		for t in testSamples:
			if line[0].find(t) != -1:
				matchedTestSample = True
				test.append(line)
				break
		if matchedTestSample == False:
			train.append(line)

	csv.writer(open(trainFile, 'w', newline='')).writerows(train)
	csv.writer(open(testFile, 'w', newline='')).writerows(test)

	#return True

def getSample (str):
	smpl = re.findall(r'\w+|\W+', str)[0]
	return smpl.split('_')[0]

def getClass (str):
	fileName = re.findall(r'\w+|\W+', str)[0]
	return fileName.split('_')[1] + '_' + fileName.split('_')[2]
	
def getMethod (str):
	mthd = re.findall(r'\w+|\W+', str)[0]
	mthd = '_'.join(mthd.split('_')[3:])
	if '' ==  mthd:
		mthd = NOISY
	return mthd
	
def process_data (file, has_header):
	# Read training data and create training set
	training_raw = {}

	for line in csv.reader((open(file))):
		if has_header:
			has_header = False
			continue
			
		sample = getSample(line[0])
		cls = getClass(line[0])
		mthd = getMethod(line[0])
		
		if cls not in training_raw.keys():
			tmp_mthd = {}
			tmp_smpl = {}
			tmp_mthd[mthd] = line[1]
			tmp_smpl[sample] = tmp_mthd
			training_raw[cls] = tmp_smpl
		elif sample not in training_raw[cls].keys():
			tmp_mthd = {}
			tmp_mthd[mthd] = line[1]
			training_raw[cls][sample] = tmp_mthd
		else:
			training_raw[cls][sample][mthd] = line[1]
		
	#print (training_raw)
	training_X = []
	training_Y = []

	for cls in training_raw.keys():
		for sample in training_raw[cls].keys():
			#print (cls + ' ' + sample + ' ' + str(training_raw[cls][sample]))
			tmp = {}
						
			tmp[NOISY] = training_raw[cls][sample][NOISY]
			tmp[KLT] = training_raw[cls][sample][KLT]
			tmp[KLT_JABLOUN] = training_raw[cls][sample][KLT_JABLOUN]
			tmp[STSA] = training_raw[cls][sample][STSA]
			tmp[LOGTSA] = training_raw[cls][sample][LOGTSA]
			tmp[LOGSTA_NEST] = training_raw[cls][sample][LOGSTA_NEST]
			tmp[LOGSTA_SAP_Q] = training_raw[cls][sample][LOGSTA_SAP_Q]
			tmp[WEUCLID] = training_raw[cls][sample][WEUCLID]
			tmp[RDC] = training_raw[cls][sample][RDC]
			tmp[RDC_NEST] = training_raw[cls][sample][RDC_NEST]
			tmp[MB] = training_raw[cls][sample][MB]
			tmp[WAVTHRE] = training_raw[cls][sample][WAVTHRE]
			tmp[SCALART] = training_raw[cls][sample][SCALART]
			tmp[TSOUKALAS] = training_raw[cls][sample][TSOUKALAS]

			#print (list(tmp.values()))
			training_X.append(list(tmp.values()))
			training_Y.append(cls)
	return training_X, training_Y


bestP = 0
bestPSamples = []
for ratio in range(1, 8):

	for testSamples in itertools.combinations(SAMPLE_NAMES, ratio):
		#print (' '.join(testSamples))
		splitData (DATA_FILE, 'train' + str(ratio) + '.csv', 'test' + str(ratio) + '.csv', testSamples)
			
		X, y = process_data ('train' + str(ratio) + '.csv', TRAINING_FILE_HAS_HEADER)		

		# train model
		clf = SVC(gamma='scale')
		clf.fit(X, y)

		right_prediction = 0
		wrong_prediction = 0

		testing_X, testing_Y = process_data('test' + str(ratio) + '.csv', TESTING_FILE_HAS_HEADER)

		results = clf.predict(testing_X)
		#print (results)
		
		confusion_matrix = np.zeros( ( len(set(testing_Y)), len(set(testing_Y)) ) )

		for (result,label) in zip(results, testing_Y):
			confusion_matrix[getIndex(result)][getIndex(label)] = confusion_matrix[getIndex(result)][getIndex(label)] + 1
			if ( result == label) :
				right_prediction = right_prediction + 1
			else:
				wrong_prediction = wrong_prediction + 1


		#print ('\nTesting Samples: ' + str(ratio) + '\nCorrect predictions: ' + str(right_prediction) + '\nIncorrect predictions: ' + str(wrong_prediction) + '\nAccuray: ' + str(right_prediction/(right_prediction + wrong_prediction)) + '\n')
		#print(confusion_matrix)

		ap = 0
		ar = 0
		for label in set(testing_Y):
			tp = 0
			fp = 0
			tn = 0
			fn = 0

			for i in range(0, len(set(testing_Y))):
				for j in range(0, len(set(testing_Y))):

					if i == getIndex(label):
						if j == getIndex(label):
							tp = tp + confusion_matrix[i][j]
						else:
							fp = fp + confusion_matrix[i][j]
					else:
						if j == getIndex(label):
							fn = fn + confusion_matrix[i][j]
						else:
							tn = tn + confusion_matrix[i][j]
			p = tp/(tp+fp)
			r = tp/(tp+fn)
			#print ('Precision for label : ' + label + ' is : ' + str(p))
			#print ('Recall for label : ' + label + ' is : ' + str(r))
			
			ap = ap + p
			ar = ar + r
		if bestP <= ap/len(set(testing_Y)):
			bestP = ap/len(set(testing_Y))
			bestPSamples = testSamples
		#print ('Average P : ' + str(ap/len(set(testing_Y))))
		#print ('Average R : ' + str(ar/len(set(testing_Y))))

print('Best Precision: ' + str(bestP))
print('Best Precision Samples: ' + ' '.join(bestPSamples))


