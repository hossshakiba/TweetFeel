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
Here are some steps of preprocessing in this project:<br>
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

## Exploratory Data Analysis (EDA)
You can find some insightful diagrams in the project, and here are the word clouds of each label:<br>
<p class="row" float="left" align="middle">
  <img src="/images/normal.jpg" width="200" title="Positive"/>
  <img src="/images/cataract.jpg" width="200" title="Neutral"/> 
  <img src="/images/dr.jpeg" width="200" title="Negative"/>
</p>
