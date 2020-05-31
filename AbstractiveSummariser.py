"""
Reference: https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/

The way the dual-LSTM model works:
2 LSTM models are trained. 
The first LSTM model that will be trained on the full text is called the encoder.
The second LSTM model that will be trained on the summary is called the decoder.
After the first LSTM model is trained, its weights will be transferred to the decoder, which is then used to train on the summary.
So, it's something like telling the decoder, "based on these encoder weights, this summary would be written".

Each LSTM layer in the encoder LSTM model is fed one word at a time to capture the contextual information present in the input sequence.
Each LSTM layer in the decoder LSTM model is trained to predict the next word given the current word. 
Since the target sequence is unknown while decoding the test sequence, we make it such that we start predicting the target sequence by passing <start> tokens, and end it when we pass <end> tokens.

The predicting/inference process:
1. We encode the entire input sequence. We initialise the decoder with the internal states of the encoder
2. Pass <start> token as input to the decoder
3. Run decoder for one timestep with internal states
4. Output will be the probability for the next word. We select the word with the highest probability
5. Pass this new word into the decoder in the next timestep, while updating the internal states of the next time step with the current time step (basically the next decoder will have info of the previous word)
6. Repeat step 3 to 5 till step 4 generates the <end> token or the maximum length of the target sequence is met

"""

import os
os.chdir('/Users/jaoming/Documents/Active Projects/Abstractive Summarisation')

# importing relevant modules
from attention import AttentionLayer             # NN layer that helps pay attention to more important words

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras import models
from keras.callbacks import EarlyStopping
from keras import backend as K
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

# importing dataset
data = pd.read_csv('Reviews.csv')
data = data.loc[:, ['Text', 'Summary']]
data.drop_duplicates('Text', inplace = True)     # drop duplicate reviews
data.dropna(axis = 0, inplace = True)            # drop rows that have NA

## altering the size of the dataset
effective = 0.01
data = data.iloc[:int(data.shape[0]*effective),:]

# data preprocessing
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}

stop_words = set(stopwords.words('english'))

## for cleaning the main text data (non summarised version)
def text_cleaner(text):
       """
       """
       # convert everything to lower case
       cleaned = text.lower()

       # remove HTML tags
       cleaned = BeautifulSoup(cleaned, "lxml").text

       # remove ('s), punctuations, special chars and any text inside parenthesis
       cleaned = re.sub(r"\([^)]*\)", "", cleaned)
       cleaned = re.sub('"', '', cleaned)
       cleaned = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in cleaned.split(" ")])    
       cleaned = re.sub(r"'s\b","",cleaned)
       cleaned = re.sub("[^a-zA-Z]", " ", cleaned) 
       
       tokens = [w for w in cleaned.split() if w not in stop_words]
       
       long_words = []
       for i in tokens:
              if len(i) >= 3:
                     long_words.append(i)
       return (" ".join(long_words)).strip()

cleaned_text = [text_cleaner(text) for text in data['Text']]

## for cleaning the summary text data
def summary_cleaner(text):
       """
       """
       cleaned = re.sub('"', '', text)
       cleaned = " ".join([contraction_mapping[t] if t in contraction_mapping else t for t in cleaned.split(" ")])
       cleaned = re.sub(r"'s\b", "", cleaned)
       cleaned = re.sub("[^a-zA-Z]", " ", cleaned)
       cleaned = cleaned.lower()

       tokens = cleaned.split()
       cleaned = ""
       for i in tokens:
              if len(i) > 1:
                     cleaned = cleaned + i + " "
       return cleaned

cleaned_summary = [summary_cleaner(text) for text in data['Summary']]

## putting together the new data
data['cleaned_text'] = cleaned_text
data['cleaned_summary'] = cleaned_summary
data['cleaned_summary'].replace('', np.nan, inplace = True)    # replace empty strings with NA
data.dropna(axis = 0, inplace = True)                          # remove rows that are NA 
data['cleaned_summary'] = data['cleaned_summary'].apply(lambda x: 'startsum ' + x + ' endsum')
data.reset_index(drop = True, inplace = True)

## checking out the cleaned versions
for i in range(5):
       print('Review:', data['cleaned_text'][i])
       print('Summary:', data['cleaned_summary'][i])
       print('\n')

# EDA for the preprocessed data
length_of_texts = []
length_of_summaries = []

for i in range(len(data.index)):
       length_of_texts.append(len(data.at[i, 'cleaned_text'].split()))
       length_of_summaries.append(len(data.at[i, 'cleaned_summary'].split()))

lengths_df = pd.DataFrame({'text': length_of_texts, 
                            'summary': length_of_summaries})
lengths_df.hist(bins = 30)
plt.show()

pd.Series(length_of_texts).value_counts()[:10]
pd.Series(length_of_summaries).value_counts()[:10]

# Data Transformation
max_len_text = 50
max_len_summary = 8
min_text_word_occurence = 4
min_summary_word_occurence = 3

""" train_x, test_x, train_y, test_y = train_test_split(data['cleaned_text'], 
                                                 data['cleaned_summary'], 
                                                 test_size = 0.1, 
                                                 random_state = 69, 
                                                 shuffle = True) """
train_x, train_y = data['cleaned_text'], data['cleaned_summary']

## Text Tokenizer
text_tokenizer = Tokenizer()
text_tokenizer.fit_on_texts(train_x)

### sieving out the rare words
word_freq = text_tokenizer.word_counts.values()
text_voc_size = len(text_tokenizer.word_counts)  # there are 10498 unique words
rare_text_voc_size = sum([1 for i in word_freq if i < min_text_word_occurence])  # there are 7011 unique words that are used less than min_text_word_occurence times
remaining_coverage = int(1000*sum([i for i in word_freq if i >= min_text_word_occurence])/sum([i for i in word_freq]))/10  # without these unique words, the 'common' words still appear 91.7% of the time
top_n_words_text = text_voc_size - rare_text_voc_size

text_tokenizer = Tokenizer(top_n_words_text)
text_tokenizer.fit_on_texts(train_x)

### convert text sequences into integer sequences
train_x = text_tokenizer.texts_to_sequences(train_x)
# test_x = text_tokenizer.texts_to_sequences(test_x)

### padding up to the maximum length
train_x = pad_sequences(train_x, maxlen = max_len_text, padding = 'post')
# test_x = pad_sequences(test_x, maxlen = max_len_text, padding = 'post')

text_voc_size = len(text_tokenizer.word_index) + 1 

## Summary Tokenizer
summary_tokenizer = Tokenizer()
summary_tokenizer.fit_on_texts(train_y)

### sieving out the rare words
word_freq = summary_tokenizer.word_counts.values()
summary_voc_size = len(summary_tokenizer.word_counts)   # there are 2530 unique words
rare_summary_voc_size = sum([1 for i in word_freq if i < min_summary_word_occurence])  # there are 1783 unique words that are used less than min_summary_word_occurence times
remaining_coverage = int(1000*sum([i for i in word_freq if i >= min_summary_word_occurence])/sum([i for i in word_freq]))/10  # without these unique words, the 'common' words still appear 89.5% of the time
top_n_words_summary = summary_voc_size - rare_summary_voc_size

summary_tokenizer = Tokenizer(top_n_words_summary)
summary_tokenizer.fit_on_texts(train_y)

### convert text sequences into integer sequences
train_y = summary_tokenizer.texts_to_sequences(train_y)
# test_y = summary_tokenizer.texts_to_sequences(test_y)

### padding up to the maximum length
train_y = pad_sequences(train_y, maxlen = max_len_summary, padding = 'post')
# test_y = pad_sequences(test_y, maxlen = max_len_summary, padding = 'post')

summary_voc_size = len(summary_tokenizer.word_index) + 1

## Deleting summaries that only have 'startsum' and 'endsum'
startsum, endsum = summary_tokenizer.word_index['startsum'], summary_tokenizer.word_index['endsum']
i = [i for i, v in enumerate(train_y) if v[0] == startsum and v[1] == endsum]

train_x, train_y = np.delete(train_x, i, axis = 0), np.delete(train_y, i, axis = 0)


# Model Building - for training of weights
"""
Return Sequences = True: LSTM produces the hidden state and cell state for each timestep
Return State = True: LSTM produces the hidden state and cell state of the last timestep only
"""
K.clear_session()
embedding_dim = 500
latent_dim = 500

## Encoder 
encoder_inputs = layers.Input(shape = (max_len_text, )) 
encoder_emb = layers.Embedding(text_voc_size, embedding_dim, trainable = True)(encoder_inputs) 

## LSTM 1 
encoder_lstm1 = layers.LSTM(latent_dim, return_sequences = True, return_state = True) 
encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_emb) 

## LSTM 2 
encoder_lstm2 = layers.LSTM(latent_dim, return_sequences = True, return_state = True) 
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1) 

## LSTM 3 
encoder_lstm3 = layers.LSTM(latent_dim, return_state = True, return_sequences = True) 
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2) 

## Set up the decoder. 
decoder_inputs = layers.Input(shape = (None, )) 
decoder_emb_layer = layers.Embedding(summary_voc_size, embedding_dim, trainable = True) 
dec_emb = decoder_emb_layer(decoder_inputs) 

## LSTM using encoder_states as initial state
decoder_lstm = layers.LSTM(latent_dim, return_sequences = True, return_state = True) 
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state = [state_h, state_c]) 

## Attention Layer
attn_layer = AttentionLayer(name = 'attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])              # based on the encoder and decoder outputs, determine which nodes are more important

## Concat attention output and decoder LSTM output 
decoder_concat_input = layers.Concatenate(axis = -1, name = 'concat_layer')([decoder_outputs, attn_out])

## Dense layer
decoder_dense = layers.TimeDistributed(layers.Dense(summary_voc_size, activation='softmax')) 
decoder_outputs = decoder_dense(decoder_concat_input) 

## Define the model
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs) 
model.summary() 

model.compile(optimizer = 'rmsprop',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['acc'])

early_stop = EarlyStopping(monitor = 'val_loss',
                            mode = 'min', 
                            verbose = 1,
                            patience = 1)

history = model.fit(
       [train_x, train_y[:, :-1]],
       train_y.reshape(train_y.shape[0], train_y.shape[1], 1)[:, 1:],
       epochs = 25,
       batch_size = 128,
       validation_split = 0.1,
       callbacks = [early_stop]
)
"""
the input of the model takes in the full review and all but the last word of the summary
the output will be all but the first word of the summary

the reason for this is because, we are trying to train the model to recognize that GIVEN
the full text review, the correct prediction of the 1st word in the summary is the 2nd word, 
2nd word in the summary is the 3rd word, ..., 2nd last word in the summary is the last word. 
we are training the model to iteratively predict the next word in the summary given the prior 
and the full text review. hence, the input is all but the last and the output is all but the first
"""

# Evaluation
plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'test')
plt.legend()
plt.show()

# Model Building - For predicting of summary
text_index_word = text_tokenizer.index_word
summary_index_word = summary_tokenizer.index_word

text_word_index = text_tokenizer.word_index
summary_word_index = summary_tokenizer.word_index

## Encode the input sequence to get the feature vector
encoder_model = models.Model(
       inputs = encoder_inputs,
       outputs = [encoder_outputs, state_h, state_c]
)

## Decoder setup
### These declarations are tensors that will hold the states of the previous time step
decoder_state_input_h = layers.Input(shape = (latent_dim, ))
decoder_state_input_c = layers.Input(shape = (latent_dim, ))
decoder_hidden_state_input = layers.Input(shape = (max_len_text, latent_dim))

### Getting the embeddings of the decoder sequence
dec_emb2 = decoder_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, 
                                                 initial_state = [decoder_state_input_h, decoder_state_input_c])

### Attention Layer
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = layers.Concatenate(axis = -1, name = 'concat')([decoder_outputs2, attn_out_inf])

### Dense softmax layer to generate the prob of each possible summary word
decoder_outputs2 = decoder_dense(decoder_inf_concat)

### Final decoder model
decoder_model = models.Model(
       [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
       [decoder_outputs2] + [state_h2, state_c2]
)

## Actual decoding function
def decode_sequence(input_seq):
       """
       Function:     To convert the vector of a text review into an abstracted review

       Inputs:       Text review in vector form. For this case, the vector has to be of length 100.

       Returns:      Abstracted Review
       """
       # Encode the input as state vectors
       e_out, e_h, e_c = encoder_model.predict(input_seq)

       # Generate empty target sequence of length 1
       target_seq = np.zeros((1, 1))

       # Populate the first word of the target sequence with the start word
       target_seq[0, 0] = summary_word_index['startsum']

       stop_condition = False
       decoded_sentence = ""

       while not stop_condition:
              output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

              # Sample a token
              sampled_token_index = np.argmax(output_tokens[0, -1, :])
              sampled_token = summary_index_word[sampled_token_index]

              if sampled_token != 'endsum':
                     decoded_sentence += sampled_token + " "
              
              # Exit condition: either hit max length or find stop word ie. endsum
              if sampled_token == 'endsum' or len(decoded_sentence.split()) >= max_len_summary - 1:
                     stop_condition = True
              
              # Update the target sequence (of length 1)
              target_seq = np.zeros((1, 1))
              target_seq[0, 0] = sampled_token_index

              # Update internal states
              e_h, e_c = h, c
       
       return decoded_sentence

# Defining some helper functions 
def seq2summary(input_seq):
       """
       Function:     To convert a sequence into a summary

       Inputs:       A vector of 10, based on review summaries

       Returns:      A worded summary
       """
       result = ""
       for i in input_seq:
              if i != 0 and i != summary_word_index['startsum'] and i != summary_word_index['endsum']:
                     result += summary_index_word[i] + " "
       return result

def seq2text(input_seq):
       """
       Function:     To convert a sequence into a text review

       Inputs:       A vector of 100, based on reviews

       Returns:      A worded review
       """
       result = ""
       for i in input_seq:
              if i != 0:
                     result += text_index_word[i] + " "
       return result

for i in range(0,10):
    print("Review:", seq2text(train_x[i]))
    print("Original summary:", seq2summary(train_y[i]))
    print("Predicted summary:", decode_sequence(train_x[i].reshape(1, max_len_text)))
    print("\n")


"""
Ways to improve:
1. Increase the training data
2. Using Bi-Directional LSTM
3. Using beam search strategy
4. Evaluate the model based on BLEU score
5. Implement pointer-generator networks and coverage mechanisms
"""
