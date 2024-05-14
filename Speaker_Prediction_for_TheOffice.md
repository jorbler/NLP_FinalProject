## Abstract
During this project, I aim to create a Neural Network to predict the speaker of a quote from *The Office (US)*. The original dataset contained all lines from the show, however, I only trained the model to predict lines from the top five most frequent speakers: Michael, Jim, Pam, Dwight, and Andy. I did tons of testing and adjustments to the architecture of the neural network to determine the optimal hyperparameters and layers for predicting the speaker of different lines. I trained a neural network that utilized Long Short-Term Memory, Bidirectional, Dropout, and Dense layers. The maximum accuracy I was able to obtain was around .37 and the minimum loss was around 2.2.

## Introduction
*The Office (US)* was an iconic mockumentary style comedy series that ran from 2005-2015. The show follows the Scranton, PA branch of Dunder Mifflin, a fictional mid-sized paper company headed by Regional Manager Michael Scott. All of the characters on *The Office* were big personalities with distinct traits and mannerisms. For example, Salesman Dwight Schrute is known for loving bears, beets, and Battlestar Galactica, while his fellow salesman Jim Halpert is known for being the office prankster and for bringing a very chill attitude to the office, and the Receptionist Pam Beasley is very shy and soft-spoken.
When watching the show, it is so interesting to watch their personalities mesh and bounce off of one another. These powerful characters that were developed and evolved over the 10 seasons of the show inspired me to attempt to create model that predicts the speaker of any given line from the office. 

### Literature
I place this project in with the author prediction neural networks, because each line spoken by a specific charcter is unique to them as an author's writing is to them. People, especially characters in the office speak about different things, have different mannerisms, use different slang, etc. Given this, I looked at existing research that exists for author prediction in machine learning. Douglass Bagnall of New Zealand published a paper where he used a multi-headed recurrent neural network for authorship prediction. Bagnall also employed a "control corpus," which he describes as a large body of text intended to help precent overfitting in the recurrent layer ^1^. Tribhuwan Singh and Jain Yashvardhan created a deep learning model for author prediction using the Corpus of English Novels (CEN) that was designed to allow tracking of short-term language change. They created three different neural networks to get the best accuracy: Feedforward Neural Network (FNN), Convolusion Neural Network (CNN), and a Recurrent Neural Network (RNN) using LSTM. All three of their models had great accuracy and minimal loss on both the testing and validation. They also use Stanford's POSTagger and GloVe to make their models train faster. They also trained the neural networks on both the original text and grammatical structure, totaling six models created ^2^.

## Methodology/Dataset
### Dataset
I was lucky to find my dataset on Kaggle (https://www.kaggle.com/datasets/fabriziocominetti/the-office-lines/data) where it had already been separated by character and labelled by season and episode.

<img src="images/datahead.png" width="700"/>

### EDA
There are 59,909 lines/rows in the dataset. I first checked to see if there were any null values and what the data types were for each feature:

<img src="/images/datainfo.png" width="550"/>

I first wanted to look at how many lines there were per season:

<img src="/images/lines_per_season.png" width="600"/>

I then wanted to look at the distribution of the lines among all of the characters. Because there are many minor characters in the show that may be present for only one episode or even only one scene, I chose to visualize only the top 20 most frequent speakers.

<img src="/images/lines_per_speaker.png" width="600"/>

The graph highlights the first problem that I encountered with this data: the class imbalance. Being the lead of the show, it is no surprise that Michael had the most lines out of all of the characters, however there is a steep drop off between Michaels total lines (around 12,000) and the character with the next most lines, which is Dwight (around 8,000). Additionally, there is a drop off between the fifth and sixth most frequent speakers, Andy and Kevin with around 4,000 and 2,000 lines respectively. I thought that using only the 20 most frequent speakers would be sufficient at first however the model was not learning any of the characters from Kevin and beyond, so I decided to only use the top five most frequent speakers: Michael, Jim, Dwight, Pam, and Andy. 

### Data Preprocessing

<img src="/images/preprocessing.png" width="700"/>

This is the process that I used for preprocessing my data. When I first ran the model, I had removed the stop words from the data and the model was overfitting really quickly (after 2 or 3 epochs). I decided to keep the stop words in the data to add some noise to the training data. 

### Model Architecture

<img src="/images/architecture.png" width="600"/>

inspired by https://www.tensorflow.org/text/tutorials/text_classification_rnn ^3^

## Results
The highest accuracy that I got for a model on one train-test-split was .3758. 
I performed a K-Fold Cross Validation with k = 6 to verify model results.
The model overfit very easily, which I tried to combat in several ways, including noise (stop words) and adding a dropout layer, however I continued to see test accuracy being significantly higher than the validation accuracy at the end of the epoch within 3-4 epochs. This can be seen especially in the K-Fold Cross Validation in every split. This graph shows the change in the test loss and validation loss for each epoch during the K-fold Cross-Validation with k = 5: 

<img src="/images/loss_kfcv.png" width="650"/>.

And this graph shows the change in the test accuracy and validation accuracy over the K-Fold Cross Validation:

<img src="/images/accuracy_kfcv.png" width="650"/>.


## Discussion
I was hoping to get at least 50% for the accuracy, but it is clear from the results of my neural network that there is more work to do. I think investigating the cause of the overfitting and employing more methods to fight against it will be crucial for making this model a better predictor of who the speaker of a given line is. I was also having a lot of trouble with input and output sizes for the layers of the neural network.
Looking at the research that has been performed in this area, I would be interested to try different kinds of neural networks (FNN, CNN) like Singh and Yashvardhan to see if there were more promising results with a different model architecture. I am curious about the multi-headed recurrent neural network that Bagnall created. In addition, I think that his approach of using a "control corpus" to prevent overfitting could be very beneficial to my model and could solve the overfitting problem that I could not seem to beat. I would also be interested in simplifying my model and trying to minimize the amount of hidden layers, which may also be controbuting to overfitting. At the beginning of the testing of neural network design for the best performance, I was constantly getting around .25 accuracy and it would not budge, but slowly wtih small tweaks and changes, I managed to increase it by .1.
One unfortunate aspect of the dataset that I chose is that there is a finite amount of data that will ever exist for training and testing purposes. The show has ended, and even with all of the lines from the show, it is a relatively small dataset compared to what other researchers in the field are using. I do not believe that this model is super specific to this data so I would be interested in trying it on some data that is more abundent and could provide more training data.
I was considering trying to do a binary classification with this data where a quote would either be classified as "Michael Scott" or "Not Michael Scott" in a different attempt to address the class imbalance and possibly improve the performance of the model. If I could get the model performing well on binary classification, I would have a solid framework for moving into higher level categorical classicification, such as for the top 5 most frequent speakers. 
For future work on this project, I would be interested in using one hot encoding for the inputs and labels. I could then try different kinds of loss, such as categorical cross entropy. In addition, if I had the oppurtunity to use a machine with high computational power to train on larger datasets/train on deeper neural networks. This would also remove time barrier with models taking a long time to train when they are not connected to powerful GPUs.

## Conclusion
I am happy with the progress that I was able to make in terms of accuracy with this neural network. I feel like I learned a lot and have a much deeper understanding of what different kinds of layers do in a neural network, as well as its applications. I know that this model is no where close to being done and I am excited to continue to work on it as I gain more knowledge through other classes and research. This was my first neural network that I built from scratch, so there are things that I already would like to do more research on and see if what I currently have is the best fit for the model and its use. I look forward to improving accuracy and loss, branching out on datasets, and exploring new types of layers and network designs.

## References
1. Bagnall, Douglass. “Author identification using multi-headed recurrent neural networks.” arXiv, https://arxiv.org/pdf/1506.04891. Accessed 12 May 2024.
2. Singh, Tribhuwan, and Jain Yashvardhan. “Author Prediction Using Deep Learning.” Github, https://github.com/J-Yash/author-prediction-using-deep-learning/tree/master. Accessed 12 May 2024.
3. Tensorflow. “Text classification with an RNN.” TensorFlow, 23 March 2024, https://www.tensorflow.org/text/tutorials/text_classification_rnn. Accessed 12 May 2024.
