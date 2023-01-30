# COVID-19 Tweets Sentiment Analysis :microbe: :bird:

<p class="row" float="left" align="middle">
  <img style="width: 100%; height: auto;" src="/images/banner.jpg" title="confusion matrix"/>
</p>

## Overview
A data & sentiment analysis project on COVID-19 tweets. Tweets are classified into three categories of Positive, Neutral, and Negative. The classifier includes a fine-tuned BERT language model.

## Data Set :file_folder:
The tweets have been pulled from Twitter and manual tagging has been done then. The names and usernames have been given codes to avoid any privacy concerns. You can find the dataset [here](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification).<br><br>
**Columns:**
1) Location :round_pushpin:
2) Tweet At :date:
3) Original Tweet :bird:
4) Label 💬


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
  <img src="/images/positive.png" width="250" height="200" title="Positive"/>
  <img src="/images/neutral.png" width="250" height="200" title="Neutral"/> 
  <img src="/images/negative.png" width="250" height="200" title="Negative"/>
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

## Inference ✔️
Inference applies knowledge from a trained neural network model and a uses it to infer a result. So, when a new unknown data set is input through a trained neural network, it outputs a prediction based on predictive accuracy of the neural network.<br><br>
Here are some predictions of the trained model for single inputs:<br>
```
Text: this cant be worse. people are dying.
Predicted Sentiment: Negative
------------------------------
Text: Good news is coming.
Predicted Sentiment: Positive
------------------------------
Text: Another boring day in quarantine...
Predicted Sentiment: Negative
------------------------------
Text: Second wave of #COVID19 in Flanders..back to more #homework again...
Predicted Sentiment: Neutral
------------------------------
Text: OK I'm angry now. I cannot believe the stupidity there is out there about #COVID19.
Predicted Sentiment: Negative
------------------------------
Text: Thank GOD! Coronavirus: No deaths reported for 48 hours in Qatar.
Predicted Sentiment: Positive
------------------------------
Text: I lost my cousine because of CORONAVIRUS last week. can't believe it...
Predicted Sentiment: Negative
------------------------------
Text: Get the latest info and updates on #COVID19 on the Canada.ca
Predicted Sentiment: Neutral
------------------------------
Text: I got vacinated today. Hope this #COVID19 ends soon.
Predicted Sentiment: Positive
------------------------------
Text: supermarkets are running out of paper toilets. This is a shame for the gov!!!
Predicted Sentiment: Negative
------------------------------
```

# Contributing
If you are interested in contributing to the project, please follow these steps:
1. Fork the repository
2. Create a new branch for your changes
3. Commit your changes and open a pull request

