# Review-Summarization-for-Quick-Purchase-Decision

We analyzed around 25000 Amazon reviews for mobile phones. Firstly, we scraped the reviews to create our dataset. Then, we formed clusters based on similar features for ex. battery, camera, etc. using TFIDF. We had rating of each of the reviews. But since we converted the entire reviews into atomic sentences(single sentences), we lacked the ground truth for each individual sentences. This was the main hurdle because let's say there is a person who has said something bad about camera and good about storage and still has given a rating as 5. This might be the case where a bad camera doesnt matter for him much. But the main aim of the project was to give a proper sentiment behind every feature of the phone. Hence, we cannot assign same rating to all the sentences of a particular review. Now, hand-labelling the entire dataset was a very tedious task and we were restricted by time. Hence, for getting the ground truth labels for our dataset, we used a pretrained model which was random forest classifier trained on an entirely different dataset. Later, we trained a Gated Recurrent Neural Network(GRNN) for the polarity of reviews in our dataset. We achieved a test accuracy of about 61%. We had 3 classes for the reviews namely positive, neutral and negative(sentiment) for each of the features. These three sentiment polarity clusters were created for each feature. A short summary of reviews in each of these clusters was created using extractive summarization. We used Maximum Coverage Minimum Redundancy approach to ease the comprehension for an individual. We implemented summarization as a 0/1 Knapsack Algorithm based on Branch and Bound approach. This project would be beneficial for people to decide whether to purchase the product, given the summary. A paper based on the same has been published successfully in ‘International Journal of Engineering Research and Applications’, Volume 8 - Issue 9 (Part 3) Sep 2018. 

Here's the link to our paper if you are interested in reading in depth.
Link: https://www.ijera.com/papers/vol8no9/p3/G0809035357.pdf


## Steps

1. Firstly, execute the file train_random_forest.py. This will train a random forest classifier on the reviews present in the train_data.csv file present in the data_rf/ folder.

2. Then, run the file preprocess.py to create the clusters for each of the features like camera, battery, memeory etc. This same file can be used for getting the output frequency of the words. You can find the part of code in the commented section. We have already created the clusters for each of the features which you will be able to find under sentence_clusters_json/ directory

3. Later, use the trained model (random forest) to get the true labels for your dataset. So our dataset will act as the test dataset for the model. Use the file test_randomforest_finalclusters.py for the same. We have already created the labelled review sentences which you can directly get from the Labelled_Reviews/ directory.

4. These labelled reviews will be used to train our actual network for sentiment analysis. Refer files train_sentiment_2class.py and train_sentiment_3class.py for getting an idea about the models trained for 2 classes ( positive and negative) and 3 classes (positive, negative and neutral) respectively. You can see how your model is behaving using the file plot_loss_curve.py. Refer the plot of our model after 50 epochs.

5. Once our sentiment model is trained, we group together the final reviews from each class for a particular feature for ex. camera, battery etc. and summarize it using Branch and Bound approach. You can refer the file summarization.py for this.


