# Boost a massively imbalanced dataset

In this blueprint we will use gretel-synthetics to boost a minority class's representation within a machine learning dataset. This blueprint can be used to improve model accuracy for fraud, cyber security, or any classification problem with an extremely limited minority class. Our goal is to generate additional samples of minority class records that will help our classifier generalize and better classify the minority class records in our test set.

## Objective
In this blueprint, we will boost accuracy on an imbalance dataset by training a generative synthetic data model to create additional records records of the minority class. Building on a few concepts from SMOTE (Synthetic Minority Oversampling TEchnique), the synthetic model will incorporate features from both minority-class records and their nearest neighbors, which are different classes but may contain relevant features or information to help the classifier generalize against new data.

## Steps
1. Click "Transform" on the project NavBar
2. Copy your Project URI key from the Console
3. Select the "Boost massively imbalanced dataset" notebook
4. If using Colab, click Runtime->Change project runtime and change to "GPU"
5. Click "Run all" to generate a dataset, or provide your own CSV or DataFrame.
6. Check out your boosted dataset in the "Records" view of this project! 

