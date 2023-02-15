# COVID-19 Tweets Sentiment Analysis :microbe: :bird:

<p class="row" float="left" align="middle">
  <img style="width: 100%; height: auto;" src="/images/banner.jpg" title="confusion matrix"/>
</p>

## Overview
The COVID-19 pandemic has significantly impacted the world, and social media platforms have become a major source of information and communication for people during these times. With millions of tweets being shared daily, it becomes increasingly important to analyze these tweets to understand public sentiment towards COVID-19. In this project, we aim to perform sentiment analysis on COVID-19 tweets using the BERT (Bidirectional Encoder Representations from Transformers) classifier model. 

## Data Set
The tweets have been pulled from Twitter and manual tagging has been done then. The names and usernames have been given codes to avoid any privacy concerns. You can find the dataset [here](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification).<br><br>
**Columns:**
1) Location :round_pushpin:
2) Tweet At :date:
3) Original Tweet :bird:
4) Label 💬


## Data Preprocessing
The data will be preprocessed using the Preprocessor class, which performs various operations such as removing URLs, hashtags, mentions, and stop words. The class also performs lemmatization on the text.<br><br>
These preprocessing operation were performed on the data set:<br>
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

## Data Analysis
In the data analysis part of the project, we started by examining the sentiment distribution in our dataset. We found that out of all tweets related to COVID-19, 19% were classified as *neutral*, 44% as *positive*, and 37% as *negative*.

* To get a sense of the most common words used in each category, we generated word clouds for the tweets in each sentiment category. The word clouds gave us a visual representation of the most frequent words used in each sentiment category, with larger words indicating higher frequency. Here are the word clouds for each category:
<p class="row" float="left" align="middle">
  <img src="/images/positive.png" width="250" height="200" title="Positive"/>
  <img src="/images/neutral.png" width="250" height="200" title="Neutral"/> 
  <img src="/images/negative.png" width="250" height="200" title="Negative"/>
</p>

* I also created a bar chart to display the most frequent words for each category. This can help us identify the common themes or topics associated with each category of tweets.

* I further explored the data to determine the most frequent origin countries of tweets. The most frequent origin countries of tweets in the analyzed dataset are: `Unknown`, `England`, `United States`, and `India`.

* I also analyzed the most frequent hashtags and mentions in the dataset. The most frequently used hashtags were `#coronavirus`, `#covid_19`, `#Coronavirus`, `#COVID2019`, and `#COVID19`. The most frequently mentioned accounts were `@realdonaldtrump`, `@youtube`, `@borisjohnson`, `@tesco`, and `@amazon`.

Overall, these analyses provided valuable insights into the sentiment and content of COVID-19 related tweets, as well as the countries and accounts most commonly associated with these tweets.


## Model
The `BertClassifier` class is used for building the model. It uses the **BERT** model as a base and adds fully connected layers on top for classification. The forward method of the class takes in the input text and its attention mask and returns the classification results.
```python
class BertClassifier(nn.Module):
    def __init__(self, class_num):
        super(BertClassifier, self).__init__()
        self.class_num = class_num
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, self.class_num),
        )

    def forward(self, input_ids, mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=mask, return_dict=False)
        out = self.fc(pooled_output)

        return out
```
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

## Inference Results
We can also use the trained model to make predictions on new text data. We provided the model with a few examples and it made predictions on their sentiment. Here are some of the examples:<br>
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

