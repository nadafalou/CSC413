# CSC413 - Final Project
Neural Networks and Deep Learning: final group project

### The Model: Introduction

##### Brief Overview

The purpose of our assignment is to create a Transformer model, using only the encoder block of the transformer, which takes in a tweet/news article related to the COVID-19 topic and predicts whether the article is "real" or "fake". 

##### Type of Task

It is clear from this brief description that our task is a binary classification task which has an input of _________ (sequence?). 

##### Transformer: Input Embeddings

First we will look at the input embeddings. We take in an article/tweet in the form of a list of words and assign each word it's own unique integer. Once we have our list of integers, we will pass this through our embedding layer. 

##### Transformer: Positional Embeddings

We will also have a positional embedding which will assign values to each word in the article. This is done by putting the odd indices in a cosine function and the even indices in a sine function. 

##### Transformer: Multi-Head Attention

Next, we will add multi-head attention. In this aspect of the model, we take a vector of queries, keys, and values to essentially determine which set of keys compares strongly to the set of queries using a dot product. The larger this product the more compatible the queries and keys are for our sequence of words. The dot product vector is placed into a vector and normalized so that the dot products are non-negative. This will output a vector with a representation of a word in a position (i.e. a linear combination of the dot products in each row).

##### Transformer: Add & Norm

With our transformer model, we will apply a residual connection and normalize our layer after the Multi-Head Attnetion layer and the Forward Pass. IDK WHAT TO DO HERE BROOOOOOOOOOOOOOO

##### Transformer: Forward Pass

In this layer, we will pass input values through a simple activation function. In our model, we use the default ReLU activation function.

##### Transformer: Linear and Softmax

We will only be using the encoder in our model, we will not be using the decoder block. After the encoder block, we will then pass our information through a Linear layer and output the softmax of the Linear Layer

### Model Architecture w

CAN SOMEONE DRAW AN IMAGE AND PASTE IT? 

### Model Parameters

NEED TO DO THE MATH

### Model Prediction Examples

ONCE MODEL IS COMPLETE

### The Data

##### Data Source
Our data source is from a different model with the same task, uploaded at https://github.com/diptamath/covid_fake_news. The paper on the dataset itself is https://arxiv.org/ftp/arxiv/papers/2011/2011.03327.pdf.

##### Data Summary

The dataset consists of 10,700 tweets, split into two group labels for whether the information is real or fake. The data contains 37,505 unique words, 5,141 of theare in both the real and fake data groups. The average length of each tweet is 27 words with large variations of length of tweets (i.e. tweets of length greater than 27). We have ensured that each set contains a near equal number of each class to prevent the possibility of training with biased data. The 10 most common words with their frequencies are (the, 6919), (of, 4610), (to, 4108), (in, 3853), (a, 3037), (and, 2710), (is, 2030), (for, 1819), (cases, 1505), and (are, 1496). We have 27,140 unique words in the training dataset, along with 173,342 total words, meaning there are 6 times the number of occurrences of a word compared to the number of words, ensuring variety of its usage. With this, we can conclude that we have enough data.

##### Data Transformation

The data was transformed by opening each of the training, valiation, and testing datasets as a csvfile, and reading each row as a string. The tweet (second column of each row) was split into a list of strings for each word, and the label (third column of the row) was converted to an integer (0 for 'real', 1 for 'fake'). The tweet was then stored in an X list applicable for the type of data (training, validation, and testing) as well as a master list of tweets for all 3 datasets. The label was stored in a respective T list for the type of data. Then, using the master list, a sorted list of all unique words in all datasets was created, and assigned indices for each word in alphabetical order in a dictionary. The X lists were then converted to lists of lists of indices using the dictionary, and the X and T lists were then converted to numpy arrays, to be usable by the neural network modules.

During the splitting of sentence strings into lists of word strings, punctation was separated and counted as words. This was done because official statements in tweets, a.k.a. real news, are generally well-written, while it is common to find tweets with fake news to have poor punctation or exaggeration, such as multiple exclamation marks. 

Additionally, words that appeared infrequently were replaced with "<low-freq-word>" to reduce the size of the vocabulary dictionary. This is to help with runtime and to reduce -- and not exceed the maximum -- amount of RAM needed while training the model. Finally, the extremely long sentences were removed from the dataset (few ones that were practically stray points) and all sentences were padded to match the length of the new longest sentence. Again, this was done to reduce computation time and to make inputs have similar format.

##### Data Split

The data was pre-split in the original dataset, into training (~60% of observations), validation (~20% of observations), and testing sets (~20% of observations). The data contains an almost even number of real (~52%) and fake (~48%) examples, with this ratio maintained in every dataset, to ensure the integrity of the model across training, validation, and testing.

### Training Curve

TODOOOOOOOOOOOO

### Hyper Parameter Tuning

This was arguably one of the hardest and most time consuming tasks. To begin with, we had started off with some typical default values for our learning rate, weight decay, batch and embedding size. We had started off with a learning rate of 0.001, weight decay of 0.05, batch size of 32, embedding size of 284 and 7 epochs. After this value we felt our model was essentially flipping a coin. This was due to the fact that we had a training accuracy of about 89%, but our validation accuracy was approximately 51% constantly. This gave us an indication of overfitting, hence we increased our batch size, and increased our learning rate. To be on the safer side of things, we wanted to observe how our model would perform if we had trained with more epochs, so we increased our epochs to a constant value of 25.
FINISH THE REST HERE

### Quantitative Measures


### Quantitative and Qualitative Results


### Results 

A very simple baseline model was implemented to compare to. The baseline model creates a dictionary of all the words that appear in the training set and counts the number of times the word appears in fake and real tweets. Thus after training, the probability of each word being in a real or fake sentence can be determined based on the training data. To predict, the average probablility of all the words in the sentence being real (or fake) is calculated, and the one with higher prbabililty is the prediction. The baseline model had a training accuracy of 81%, while the test accuracy was 51%. (Note that for the base model, there are no hyperparameters to tune and so the validation set was absorbed into the training set)

### Ethical Considerations

COVID-19 has affected our world quite greatly. Citizens across the world have suffered great losses which can be quite easily be explained by misinformation or by ignorance itself. By classifying fake news from real news, people can then use this to make more correctly informed decisions. 

##### Unethical 

Fake news detection is informational, however, it’s crucial to observe that the model is only as objective as the data we train on. In essence, it can be highly likely that the data we train on will be biased in some way, shape or form. Fortunately, there are articles that have been proven by a great number of credible sources and have been fact checked by science as well, but there are also other articles that cannot be proven using science. These articles can be manipulated to force a population of the world to believe a myth may be true about COVID-19, and that facts may be false (i.e. telling the population that masks trap the virus in your face making it easier to contract COVID-19). Hence, it is crucial to understand that this model can be used to manipulate a fraction of the world, if not then perhaps even the whole world. 

##### Ethical

With an unethical purpose, there will always be an ethical purpose. We can use this model and train on data specificly known to be scientifically true and scientifically false by organizations that come to unanimous decisons. Since science is a more objective topic than it is subjective, this will reduce the bias in our data which can help us believe that our model is more robust. A more robust model will then help the world be informed about COVID-19 and spread awareness in a confident manner. This will create a chain that hopefully will help us tackle COVID-19 in multiple angles.

### Authors

