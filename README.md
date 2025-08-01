# Fake account detection in Twitter

## Features
- Uses machine learning algorithms to classify user accounts into real or fake.
- Classifiers implemented are:
	- Naive Bayes
	- SVC
	- K-Nearest neighbour
- Calcualtes accuracy and error rate of classifiers.
- Displays comparison results of each classifier using bar graph.

## A glimpse of running the model
**Output Window**

<img src="/image/1.png" width="500">

**Custom User Account**

<img src="/image/2.png" width="500" >

**Sample Classifier accuracy**

<img src="/image/3.png" width="500">

**Classifier comparison**

<img src="/image/4.png" width="500">


## How to run?
- Install required libraries using
```bash
	pip install [library-name]
```
- Execute the Python code using
```bash
	python model.py
```
- Select the dataset "twitter_data.csv" or enter custom account details.
- Choose the classifier and results will be displayed in new tab.

