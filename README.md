##  kaggle credicard data fraud detection 
#### Simple Multivariate gaussian distribution assumption is used to predict the probability that a given transaction is fraudulent.   

- The raw data has ~250,000 normal transactions and ~500 fraudulent transactions  - This is highly imbalanced data but such is the nature of credit card data.   
- So regular  ML techniques like random forests and logistic regression on raw data may give high accuracy but will be not very useful in detecting the fraud transactions. Because predicting every transaction as not fraudulent still gives an accuracy of 250000/250500 i.e. 99.8% accuracy. 
- One way to deal with this problem is resampling from the raw data to get a balanced sample and then run logistic regression or random forests  
- Another alternative is that we can use anomaly detection algorithms.   

In this notebook I apply the simple multivariate gaussian distribution assumption to calculate the probability that a given transaction is fraudulent.  First I create a training data which contains only normal transactions. 

I split the transactions into normal and fraudulent transactions. 

* Train Data:   60% of the the normal transactions 
* Cross Validation Data: 20% of normal transactions and 50% of the fraudulent transactions 
* Test Data: 20% of normal transactions and 50% of fraudulent transactions  

All the features are assumed to be independent of each other and are assumed to follow normal distribution. From the training data I calibrate the mean and variance of each feature. With this I calculate the probability of each transaction feature having a given value in the cross validation data. Now from the cross validation probabilities, I come up with an epsilon probability which can be used to classify the transaction.   Increasing epsilon decreases false positives (normal but classified as fraud) and also the true positives (fraud classified as fraud). Using a very small epsilon increases the true positive rate but also dramatically increases the false positive rate. It is important to strike a balance here
