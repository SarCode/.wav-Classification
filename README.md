<p align="left"> <img src="https://komarev.com/ghpvc/?username=sarcode&label=Profile%20views&color=0e75b6&style=flat" alt="sarcode" /> </p>

[![LinkedIn][linkedin-shield]][linkedin-url]

# .wav-Classification
Classification of .wav audio file using methods for predicting interference like klt, klt_jabloun etc.

## Libraries ##
* Pandas
* sklearn
  * Pre-processing
  * Linear Model
  * Ensemble
  * SVM
  * Metrics
  * Train Test Split
  
## To run ##
Execute:

* create_dataset_from_original_file.py
* predicted_using_created_dataset.py

## Improving Code
The scripts create_dataset_from_original_file.py & predicted_using_created_dataset.py are for basic understanding of Data Wrangling and Pre-Processing.

If you want higher accuracy and have knowledge of python then execute only class_predict_updated.py.


## Dataset ##
data_svm_org_new_v2.csv contains 1793*2 data, varying between 1 - 5.

There are 8 target classes:
* babble_sn5
* car_sn5
* street_sn5
* train_sn5
* babble_sn10
* car_sn10
* street_sn10 
* train_sn10.

Each class has 16 samples: 
* sp01
* sp02
* sp03
* sp04
* sp06
* sp07
* sp08
* sp09
* sp11
* sp12
* sp13
* sp14
* sp16
* sp17
* sp18
* sp19 
(05,10,15 is not there).


## Data Wrangling ##

Data in column[0] is wrangled by delimiter "_".

Splitting data by "_" we create 14 columns that have methods to determine .wav file.

End product of data wrangling is to convert 1793 * 2 into 128 * 14.

## Machine Learning Models ##
* [Logistic Regression](<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>)

* [Random Forest Classifier](<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>)
 
* [SVC](<https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>)

## Performance Metric ##
[Accuracy Score](<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html>)


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/sarthak-agarwal-dell/
