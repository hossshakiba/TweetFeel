# COVID-19 Tweets Sentiment Analysis :microbe: :bird:

## Overview
A data & sentiment analysis project on COVID-19 tweets. Tweets are classified into three categories of Positive, Neutral, and Negative. The classifier includes a fine-tuned BERT language model.

## Data Set :file_folder:
The tweets have been pulled from Twitter and manual tagging has been done then. The names and usernames have been given codes to avoid any privacy concerns. You can find the dataset [here](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification).<br><br>
**Columns:**
1) Location :round_pushpin:
2) Tweet At :date:
3) Original Tweet :bird:
4) Label :ok:


## Data Preprocessing :hammer:
We should preprocess and clean the data before feeding it to our model because poor quality or unclean data will likely result in inaccurate insights.<br><br>
Here are the preprocessing steps taken in this project:<br>
1) Deleting missing values
2) Deleting duplicate rows
3) Lowering texts
4) Removing mentions
5) Removing hashtags
6) Removing URLs
7) Removing punctuations
8) Removing html tags
9) Removing double spaces
10) Remving stop words
11) Removing numbers
12) Lemmatizing texts

## Exploratory Data Analysis (EDA) 📊
You can find some insightful diagrams in the project that come in handy for data analysis. For instance, here are the word clouds of each label:<br>
<p class="row" float="left" align="middle">
  <img src="/images/positive.png" width="300" height="250" title="Positive"/>
  <img src="/images/neutral.png" width="300" height="250" title="Neutral"/> 
  <img src="/images/negative.png" width="300" height="250" title="Negative"/>
</p>

## Results :white_check_mark:
Here are the results after **5** ephocs:<br>
Train accuracy: 97%<br>
Validation accuracy: 89%<br>
Test accuracy:  87%
<br>
## Confusion Matrix
The following figure is the confusion matrix of the test data set:
<p class="row" float="left" align="middle">
  <img src="/images/cfm.png" title="confusion matrix"/>
</p>
