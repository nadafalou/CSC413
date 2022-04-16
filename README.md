# CSC413 - Final Project
This project is a final submission for the University of Toronto course, CSC413: Neural Networks and Deep Learning.

### The Model: Introduction

##### Brief Overview

Fake news is widespread and harmful, as it misleads people and is often used for revenue. Our model’s task is to take a post, which is represented by a sequence of English words and emojis, and then perform binary classification to determine whether the post is presenting real or fake information related to COVID-19.

##### Type of Task

The task is a binary classification problem, were input sequences of words are classified as 'real' or 'fake' information, and then compared to their label to determine the accuracy of those predictions. Our model is a Transformer model with a Glove Embedding layer, a positional encoding layer with dropout, 2-attentional and feed-forward layers, followed by a linear layer.

##### Transformer: Input Embeddings

First we will look at the input embeddings. We take in an article/tweet in the form of a list of words and assign each word it's own unique integer. Once we have our list of integers, we will pass this through our embedding layer, which uses pretrained Glove Embeddings to transform the integers into a vector representation to derive a distance from each word.

##### Transformer: Positional Embeddings

We will also have a fixed positional embedding which will assign values to each word in the article. This is done by putting the odd indices in a cosine function and the even indices in a sine function, in order to model both relative and absolute position in a sentence, then add it to the embeddings from the previous layer.

##### Transformer: Multi-Head Attention

Next, we will add multi-head attention. In this aspect of the model, we take a vector of queries, keys, and values to essentially determine which set of keys compares strongly to the set of queries using a dot product. The larger this product the more compatible the queries and keys are for our sequence of words. The dot product vector is placed into a vector and normalized so that the dot products are non-negative. This will output a vector with a representation of a word in a position (i.e. a linear combination of the dot products in each row). This is done for 5-attention heads, which has 5 copies of queries, keys, and values, then concatenates the attention heads, and dot products it with a final matrix to transform the output back into the embedding dimension.

##### Transformer: Add & Norm

After the multi-head attention sub-layer, the output is added to the input before the attentional layer and normalized to have 0 mean and variance 1, so as to reduce the effect of the gradient on the softmax function within the attentional and final layers being to small for extreme values. Then that output is passed to a feed-forward network consisting of two fully connected layers, with dropout. The feed-forward layer is then normalized with its own input and outputs just like the attentional layer.


##### Transformer: Linear and Softmax

The embeddings have their means taken in the embedding dimension, then passed to a linear layer to get the raw scores for binary classification.


##### Transformer: Forward pass

The input of sequences of indices is transformed into a Glove embedding, then passed into a positional encoder. The output of that is then passed into N encoder layers. The attentional and feed-forward layers form a single encoder layer,  and there are a total of 2 encoder layers. The output of the encoder is then passed through the linear layer, to output the raw scores for each class. The softmax is taken to find the probabilities, and the argmax is taken to make a prediction on which class the input belongs to.

### Model Architecture

![](transformers.png)

### Model Parameters

For our parameter calculations, we first list some of our hyperparameters. Our embedding size is 300, the attentional sublayers have 5 heads, and there are 2 Transformer encoder layers.
The Glove embedding layer is from a pretrained pytorch module, so there are no parameters from it, and the Positional encoder is based on a fixed encoding, requiring no tunable weights. The Transformer encoder layer contains an attentional sub-layer, which has 5 heads, each with a key, query, and value weight matrices, with each of them of size 300 x 300, with a bias of size 300 each. Since we are using multi-head attention, there is also a weight matrix with dimension 300 x 300, with a bias of 300. The norm after this layer has a weight vector of 300, as well as a bias vector of 300. This creates 3 * 300 * 300 + 3 * 300 + (300)^2 + 300 + 300 + 300 = 361,800 parameters in the attentional sublayer. The feed-forward sublayer consists of two fully connected layers, one with 300 * 2048 weights and 2048 biases, and one with 2048 * 300 weights and 300 biases. After the feed-forward sublayer there is a normalization weight of size 300 as well as a bias of size 300, which results in the feed-forward sublayer totaling 1,231,748 parameters. There is a single linear layer after the transformer encoder with 300 * 2 weights and 2 biases. Thus there are 2 * (361,800 + 1,231,748) + 300 * 2 + 2 = 3,187,698 parameters.

### Model Prediction Examples

ONCE MODEL IS COMPLETE

### The Data

##### Data Source

We used data gathered by [Patwa et. al (2021)](https://arxiv.org/ftp/arxiv/papers/2011/2011.03327.pdf) in an effort to "fight an infodemic". The data files are from a different machine learning model uploaded on a [GitHub repo](https://github.com/diptamath/covid_fake_news) with the same task, which uses the MIT license and specifies that we are allowed to use a copy of the software without restriction, free of charge.

##### Data Summary

The dataset consists of around 10,700 posts labeled real or fake depending on the information contained. We have ensured that each of the train, validation and test sets contains a near equal number of each class (fake and real) to prevent the possibility of training, validateing and testing with biased data.

The average length of each post is 29 words and 138 characters, with some posts being much longer. The data contains 37,505 unique words, with 5,141 of them appearing in both real and fake posts. The 10 most common words (note that we will talk about punctuation as words in the next section) with their _percentage_ frequencies are ('.', 6.79), ('the', 3.37), ('#', 2.27), ('of', 2.17), ('https://t', 2.11), ('to', 2.0), ('in', 1.87), ('a', 1.52), ('and', 1.37), ('is', 1.03). We have 19,092 unique words in the training dataset, with a total of 160,198 words, meaning there are 8 times the number of occurrences of a word compared to the number of words, ensuring variety of usage of words. With this, we can conclude that we have enough data.

##### Data Transformation

The data was transformed by opening each of the training, valiation, and testing datasets as a csvfiles, and reading each row as a string. The post (second column of each row) was split into a list of strings for each word, and the label (third column of the row) was converted to an integer (0 for 'real', 1 for 'fake'). The post was then stored in an X list applicable for the type of data (training, validation, and testing) as well as a master list of posts for all 3 datasets. The label was stored in a respective T list for the type of data. Then, using the master list of posts, a sorted list of all unique words in all datasets was created, and assigned indices for each word in alphabetical order in a dictionary. The X lists were then converted to lists of lists of indices using the dictionary, padded with 0s on their right sides to account for variable input length, per the max length of each dataset. The X and T lists were then converted to numpy arrays, to be usable by the neural network modules.

During the splitting of sentence strings into lists of word strings, punctuation was separated and counted as words. This was done because official statements in posts, a.k.a. real news, are generally well-written, while it is common to find posts with fake news to have poor punctuation or exaggeration, such as multiple exclamation marks. Thus, we felt the need to keep punctuation in the posts as they may contribute to the prediction of whether the information is real or fake.

According to [Patwa et. al (2021)](https://arxiv.org/ftp/arxiv/papers/2011/2011.03327.pdf), most of the dataset consisted of posts, but some datapoints were general social media posts. In our model, we padded inputs to have same length, which is why we decided to remove all non-post datapoints (i.e. datapoints with more than 280 characters) to improve our training runtime and efficiency. Removing posts longer than 280 characters resulted in removing 816 posts from the training data, leaving us with 5,604 datapoints to train on. Given the spike in efficiency, we determined that losing these datapoints is worth it.

Note: similar analyses were done on validation and test datasets and yielded similar results. Exact statistics can be found in data.py.

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

A very simple baseline model was implemented to compare to. The baseline model creates a dictionary of all the words that appear in the training set and counts the number of times the word appears in fake and real posts. Thus after training, the probability of each word being in a real or fake sentence can be determined based on the training data. To predict, the average probablility of all the words in the sentence being real (or fake) is calculated, and the one with higher prbabililty is the prediction. The baseline model had a training accuracy of 81%, while the test accuracy was 51%. (Note that for the base model, there are no hyperparameters to tune and so the validation set was absorbed into the training set)

### Ethical Considerations

COVID-19 has affected our world quite greatly. As mentioned before, fake news is harmful as it can mislead the general public into making decisions that benefit groups, such as pharmaceutical companies, political parties, etc. By classifying fake news from real news, people can then use this to make more correctly informed decisions.

Fake news detection is informational, however it is crucial to observe that the model is only as objective as the data we train it on. If the data we train it on is biased towards a certain ideology or opinion, then determining what is real or fake is subjective, which again can manipulate or misinform others. We can try to be objective by labelling the datasets according to credible scientific sources, but even those sometimes conflict, especially when it comes to new theories and experiments. In terms of COVID-19, this can materialize in things that groups of medical experts themselves disagree on, leading to whoever manually labeled the data to take one opinion over the other and causing the data to be biased.

### Authors

This project is brought to you by Joshua Fratarcangeli, Nada El-Falou, Raghav Sharma and Yash Dave. Overall, we "pair programmed" and worked synchronously on calls, switching around tasks and helping each other constantly. However, the general task split was as follows:
- Joshua: data loading and exploration, model write-up
- Nada: data pre-processing and exploration, write-up
- Raghav: transformer model design and implementation
- Yash: model training, write-up
