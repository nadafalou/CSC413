# CSC413 - Final Project
Neural Networks and Deep Learning: final group project

## Data Source
Our data source is from a different model with the same task, uploaded at https://github.com/diptamath/covid_fake_news. The paper on the dataset itself is https://arxiv.org/ftp/arxiv/papers/2011/2011.03327.pdf.

## Data Summary
The dataset consists of 10,700 tweets, split into two group labels for whether the information is real or fake. The data contains 37,505 unique words, 5,141 of which are in both the real and fake data groups. The average length of each tweet is 27 words. The 10 most common words with their frequencies are (the, 6919), (of, 4610), (to, 4108), (in, 3853), (a, 3037), (and, 2710), (is, 2030), (for, 1819), (cases, 1505), and (are, 1496). We have 27,140 unique words in the training dataset, along with 173,342 total words, meaning there are 6 times the number of occurrences of a word compared to the number of words, ensuring variety of its usage.

## Data Transformation
The data was transformed by opening each of the training, valiation, and testing datasets as a csvfile, and reading each row as a string. The tweet (second column of each row) was split into a list of strings for each word, separated by spaces, and the label (third column of the row) was converted to an integer (0 for 'real', 1 for 'fake'). The tweet was then stored in an X list applicable for the type of data (training, validation, and testing) as well as a master list of tweets for all 3 datasets. The label was stored in a respective T list for the type of data. Then, using the master list, a sorted list of all unique words in all datasets was created, and assigned indices for each word in alphabetical order in a dictionary. Tweets that were over 280 characters were removed from the dataset. The X lists were then converted to lists of lists of indices using the dictionary, padded with a 0 character on the right side so that all the data has the same length, and the X and T lists were then converted to numpy arrays, to be usable by the neural network modules.

## Data Split
The data was pre-split in the original dataset, into training (~60% of observations), validation (~20% of observations), and testing sets (~20% of observations). The data contains an almost even number of real (~52%) and fake (~48%) examples, with this ratio maintained in every dataset, to ensure the integrity of the model across training, validation, and testing.
