# Sentiment Analysis with Naïve Bayes and SVM

This repository contains the implementation and comparison of two sentiment analysis models: a Naïve Bayes classifier and a Support Vector Machine (SVM) classifier. The objective is to analyze the improvements made in the SVM classifier and understand why these improvements led to increased classification accuracy.

## Table of Contents

1. [Overview](#overview)
2. [Required Libraries and Installation](#required-libraries-and-installation)
3. [Data Pre-processing and Augmentation](#data-pre-processing-and-augmentation)
4. [Feature Representation](#feature-representation)
5. [Classifier Algorithms](#classifier-algorithms)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Results and Analysis](#results-and-analysis)
8. [Further Reading](#further-reading)
9. [References](#references)

## Overview

The comparison between the Naïve Bayes and SVM classifiers demonstrates significant differences in performance across key metrics such as accuracy, precision, recall, and F1-score. The SVM classifier outperforms the Naïve Bayes classifier, attributing its superior ability to correctly classify sentiment in the given text data to the SVM's ability to find the optimal decision boundary between the two classes and its utilization of the kernel trick for handling non-linear relationships in the feature space.

## Required Libraries and Installation

To run this notebook, you'll need to install the following Python libraries:

```bash
pip install pandas numpy scikit-learn nltk matplotlib
```

**Note: Make sure you have Python 3.6 or higher installed before attempting to install these packages.**

## Data Pre-processing and Augmentation

The SVM classifier uses the `TextTfidfSynonymAugmentedPipeline`, which introduces several enhancements compared to the pipeline used for the Naïve Bayes classifier. One significant improvement is the use of synonym replacement for data augmentation. This technique helps to expand the training dataset by replacing words with their synonyms, increasing the model's exposure to different wordings and phrasings and making it more robust and generalizable to unseen data.

## Feature Representation

While the Naïve Bayes classifier utilizes a `CountVectorizer` for the bag-of-words representation, the SVM classifier adopts the `TfidfVectorizer` to generate TF-IDF features. The TF-IDF approach weighs terms based on their importance in the document and the entire corpus, reducing the impact of common but less informative words. Furthermore, the SVM classifier includes n-grams, capturing more contextual information and enhancing the model's ability to recognize patterns in the text.

## Classifier Algorithms

The Naïve Bayes classifier is based on Bayes' theorem and assumes conditional independence between features. This assumption might not always hold true for text data, leading to suboptimal classification performance. On the other hand, the SVM classifier aims to find the optimal hyperplane that maximizes the margin between classes. SVM inherently deals with non-linear decision boundaries using kernel functions, allowing for more accurate and flexible classification.

## Hyperparameter Tuning

The SVM classifier employs a more extensive grid search for hyperparameter tuning compared to the Naïve Bayes classifier. This extensive search allows the SVM model to find a better combination of hyperparameters, potentially leading to higher accuracy. The tuned hyperparameters include the cost parameter 'C,' kernel function, polynomial degree, gamma value, and coef0.

## Results and Analysis

The improvements in data preprocessing and augmentation, feature representation, classifier algorithm, and hyperparameter tuning contribute to the SVM classifier's enhanced performance. By addressing the shortcomings of the Naïve Bayes classifier and employing a more powerful classification algorithm, the SVM classifier achieves higher accuracy in sentiment analysis tasks.

## Further Reading

- Recurrent neural networks (RNN)
- Transformers
- Sentiment lexicons
- Feature selection
- Dimensionality reduction

## References

- Wikipedia's contributors. Bag of words. Wikipedia, The Free Encyclopedia. Available from: https://en.wikipedia.org/wiki/Bag-of-words_model [Accessed 10 March 2023].

- scikit-learn developers. CountVectorizer. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html [Accessed 15 March 2023].

- NLTK Documentation (n.d.). Available from: https://www.nltk.org/. [Accessed 15 March 2023].

- Towards Data Science. Multinomial Naïve Bayes for Documents Classification and Natural Language Processing (NLP). Available from: https://towardsdatascience.com/multinomial-naïve-bayes-for-documents-classification-and-natural-language-processing-nlp-e08cc848ce6 [Accessed 16 March 2023].

- scikit-learn developers. MultinomialNB. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html [Accessed 16 March 2023].

- scikit-learn developers. GridSearchCV, RandomizedSearchCV. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html [Accessed 22 March 2023].

- Wikipedia contributors. Confusion Matrix. Wikipedia, The Free Encyclopedia. Available from: https://en.wikipedia.org/wiki/Confusion_matrix [Accessed 22 March 2023].

- Wikipedia contributors. Receiver Operating Characteristic (ROC). Wikipedia, The Free Encyclopedia. Available from: https://en.wikipedia.org/wiki/Receiver_operating_characteristic [Accessed 22 March 2023].

- Analytics Vidhya. Understanding AUC - ROC Curve. Available from: https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/ [Accessed 22 March 2023].

- Kaggle. Data Augmentation by Synonym Replacement. Available from: https://www.kaggle.com/code/rohitsingh9990/data-augmentation-by-synonym-replacement [Accessed 23 March 2023].

- Wikipedia contributors. TF-IDF. Wikipedia, The Free Encyclopedia. Available from: https://en.wikipedia.org/wiki/Tf–idf [Accessed 24 March 2023].

- In Pursuit of Artificial Intelligence. Brief Introduction to N-Gram and TF-IDF Tokenization. Available from: https://medium.com/in-pursuit-of-artificial-intelligence/brief-introduction-to-n-gram-and-tf-idf-tokenization-e58d22555bab [Accessed 24 March 2023].

- scikit-learn developers. TfidfVectorizer. Available from: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html [Accessed 24 March 2023].

- Wikipedia's contributors. Recurrent Neural Networks (RNN). Wikipedia, The Free Encyclopedia. Available from: https://en.wikipedia.org/wiki/Recurrent_neural_network [Accessed 01 April 2023].

- Wikipedia's contributors. Transformer (machine learning model). Wikipedia, The Free Encyclopedia. Available from: https://en.wikipedia.org/wiki/Transformer_(machine_learning_model) [Accessed 01 April 2023].

- Springer. Sentiment lexicons. Available from: https://link.springer.com/article/10.1007/s10115-020-01497-6 [Accessed 01 April 2023].

- Machine Learning Mastery. Feature Selection with Real and Categorical Data. Available from: https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/ [Accessed 02 April 2023].

- Wikipedia's contributors. Dimensionality Reduction. Wikipedia, The Free Encyclopedia. Available from: https://en.wikipedia.org/wiki/Dimensionality_reduction [Accessed 02 April 2023].
