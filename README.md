# Abstractive-Summarisation
Using a deep learning model that takes advantage of LSTM and a custom Attention layer, we create an algorithm that is able to train on reviews and existent summaries to churn out and generate brand new summaries of its own.

![Header Image](https://github.com/jaotheboss/Abstractive-Summarisation/blob/master/images/result_example.png)

## Objective:
To be able to generate new sentences of summaries in order to create a more smooth flowing reading experience.

## Context:
Prior to this project, I have been working on extractive summaries with an altered Google PageRank-ing algorithm. Instead of ranking pages, I ranked sentences in an article. Afterwhich, I generated the more relevant sentences in that article. Just as the objective stated, this abstractive summarisation project aims to smoothen and improve the reading experience of the summary by improving the lexical chains between sentences and the grammatical consistencies.

## Methodology:
For this case, I wanted to have the machine learning model to learn the context of the text itself. In order for this to be done, some form of training had to be executed; unlike the extractive summarisation algorithm, which did not require a training set whatsoever. 

Hence, a training set had to be curated. For this project, the training set would be reviews from Amazon followed by their summaries. In a nutshell, this model trains by feeding the full text review into the system and adjusting the weights by backpropogation using the summary as the response value. 

Let's dive in a little deeper into the way the training works. For the LSTM model to be able to churn out a brand new sentence, we need to train it to know how to churn out a sentence, **word by word**. Thus, we also have to feed the summaries into the model, where we:

   1. Feed the 1st word of the summary with the 2nd word as the response,
   2. Feed the 2nd word of the summary with the 3rd word as the response,
   3. ...
 
The point being, we need to feed the 1st to the 2nd last word of the summary into the model as the input, while we have the 2nd word to the last word of the summary be the true response of the model.

There are 3 parts to the entire model: The Encoder, The Decoder, The Attention Layer.
   1. **The Encoder** - Is an LSTM model that takes in the full text of the review to learn how to transform the context of those reviews into neural network parameters
   2. **The Decoder** - Is an LSTM model that takes in the learned neural network parameters from The Encoder **and** the 1st to 2nd last word of the text summary data to learn the patterns of the summary **given/with respect to** the full text review
   3. **The Attention Layer** - Adjusts the attention of The Decoder based on the contextual understanding of **both** the full text review and the full text summary. The intuition behind the Attention Layer is basically finding and focusing on the essence of the question or text. For example, if the question was, "What animal do you like?", simply focusing on the word 'animal' would get you to consider all animals for this context. Thereafter, focusing on the word 'like' would get you to answer with your favorite animal straightaway. Hence, instead of fully considering all 5 words, by focusing on less than half the question and 'blurring' the rest, we are able to generate a response already.
   
![Flowchart](https://github.com/jaotheboss/Abstractive-Summarisation/blob/master/images/Abstractive%20Summarisation%20Flowchart.png)

The flowchart above aims to show a super simplistic perspective of how the absractive summarisation model works. 

## Reflection:
- There are definitely better ways to abstractively summarise text using unsupervised means. However, this method still seems to be at the basis of how to work on a seq2seq model. 
- Translation can be considered for this particular model.

## Current Performance Enhancing Strategies:
- Cutting out unique words. About 80% of the unique words in the reviews and summaries occur less than 4-5 times. Having said that, the remaining 20% of unique words cover about 90% of the reviews and summaries. Hence, we are able to cut down on the vocabulary size by 80% while keeping 90% of the data. 
- Cutting down on summaries that do not contain any words after preprocessing
- Included dropout and recurrent_dropout rates
