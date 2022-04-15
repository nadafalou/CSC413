# CSC413 - Final Project
This project is a final submission for the University of Toronto course, CSC413: Neural Networks and Deep Learning. 

### The Model: Introduction

##### Brief Overview

Fake news is widespread and harmful, as it misleads people and is often used forrevenue. Our model’s task is to take a tweet, which is represented by a sequence of English words and emojis, and then perform binary classification to determine whether the tweet is presenting real or fake information related to COVID-19.

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

We used data gathered by [Patwa et. al (2021)](https://arxiv.org/ftp/arxiv/papers/2011/2011.03327.pdf) in an effort to "fight an infodemic". The data files are from a different machine learning model uploaded on a [GitHub repo](https://github.com/diptamath/covid_fake_news) with the same task, which uses the MIT license and specifies that we are allowed to use a copy of the software without restriction, free of charge.

##### Data Summary

The dataset consists of around 10,700 tweets labeled real or fake depending on the information contained. We have ensured that each of the train, validation and test sets contains a near equal number of each class (fake and real) to prevent the possibility of training, validateing and testing with biased data. 

The average length of each tweet is 29 words and 138 characters, with some tweets being much longer. The data contains 37,505 unique words, with 5,141 of them appearing in both real and fake tweets. The 10 most common words (note that we will talk about punctuation as words in the next section) with their _percentage_ frequencies are ('.', 6.79), ('the', 3.37), ('#', 2.27), ('of', 2.17), ('https://t', 2.11), ('to', 2.0), ('in', 1.87), ('a', 1.52), ('and', 1.37), ('is', 1.03). We have 19,092 unique words in the training dataset, with a total of 160,198 words, meaning there are 8 times the number of occurrences of a word compared to the number of words, ensuring variety of usage of words. With this, we can conclude that we have enough data.

##### Data Transformation

The data was transformed by opening each of the training, valiation, and testing datasets as a csvfiles, and reading each row as a string. The tweet (second column of each row) was split into a list of strings for each word, and the label (third column of the row) was converted to an integer (0 for 'real', 1 for 'fake'). The tweet was then stored in an X list applicable for the type of data (training, validation, and testing) as well as a master list of tweets for all 3 datasets. The label was stored in a respective T list for the type of data. Then, using the master list of tweets, a sorted list of all unique words in all datasets was created, and assigned indices for each word in alphabetical order in a dictionary. The X lists were then converted to lists of lists of indices using the dictionary, and the X and T lists were then converted to numpy arrays, to be usable by the neural network modules.

During the splitting of sentence strings into lists of word strings, punctation was separated and counted as words. This was done because official statements in tweets, a.k.a. real news, are generally well-written, while it is common to find tweets with fake news to have poor punctation or exaggeration, such as multiple exclamation marks. Thus, we felt the need to keep punctuation in the tweets as they may contribute to the prediction of whether the information is real or fake.

According to [Patwa et. al (2021)](https://arxiv.org/ftp/arxiv/papers/2011/2011.03327.pdf), most of the dataset consisted of tweets, but some datapoints were general social media posts. In our model, we padded inputs to have same length, which is why we decided to remove all non-tweet datapoints (i.e. datapoints with more than 280 charachters) to improve our training runtime and efficiency. Removing tweets longer than 280 characters resulted in removing 816 tweets from the training data, leaving us with 5,604 datapoints to train on. Given the spike in efficiency, we determined that losing these datapoints is worth it. 

Note: similar analyses were done on validation and test datasets and yeilded similar results. Exact statistics can be found in data.py.

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

COVID-19 has affected our world quite greatly. As mentioned before, fake news is harmful as it can mislead the general public into making decisions that benefit groups, such as pharmaceutical companies, political parties, etc. By classifying fake news from real news, people can then use this to make more correctly informed decisions. 

Fake news detection is informational, however it is crucial to observe that the model is only as objective as the data we train it on. If the data we train it on is biased towards a certain ideology or opinion, then determining what is real or fake is subjective, which again can manipulate or misinform others. We can try to be objective by labelling the datasets according to credible scientific sources, but even those sometimes conflict, especially when it comes to new theories and experiments. In terms of COVID-19, this can materialize in things that groups of medical experts themselves disagree on, leading to whoever manually labeled the data to take one opinion over the other and causing the data to be biased.

### Authors

This project is brought to you by Joshua Fratarcangeli, Nada El-Falou, Raghav Sharma and Yash Dave. Overall, we "pair programmed" and worked synchronously on calls, switching around tasks and helping each other constantly. However, the general task split was as follows:
- Joshua: data loading and exploration, model write-up
- Nada: data pre-processing and exploration, write-up
- Raghav: transfromer model design and implementation
- Yash: model training, write-up