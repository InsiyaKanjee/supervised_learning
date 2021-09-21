# Supervised Machine Learning - Predicting Credit Risk

A machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not is built. 

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

This data has been used to create machine learning models to classify the risk level of given loans. Specifically, comparing the Logistic Regression model and Random Forest Classifier.

### Retrieve the data

In the `Generator` folder in `Resources`, there is a [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) notebook that will download data from LendingClub and output two CSVs: 

* `2019loans.csv`
* `2020Q1loans.csv`

Using an entire year's worth of data (2019), prediction will be made for the the credit risk of loans from the first quarter of the next year (2020).

Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

## Preprocessing: Convert categorical data to numeric

Created a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. Similarly, created a testing set from the 2020 loans, also using `pd.get_dummies()`. Note! There are categories in the 2019 loans that do not exist in the testing set. Used code to fill in the missing categories in the testing set. 

## Consider the models

Created and compared two models on this data: a logistic regression, and a random forests classifier. Before creating, fitting, and scoring the models, made predictions as to which model should perform better. 

## Fit a LogisticRegression model and RandomForestClassifier model

Created a LogisticRegression model, fit it to the data, and print the model's score. Did the same for a RandomForestClassifier. 

## Revisit the Preprocessing: Scale the data

The data going into these models was never scaled, an important step in preprocessing. Used `StandardScaler` to scale the training and testing sets. 

Fitted and scored the LogisticRegression and RandomForestClassifier models on the scaled data. 

### References

LendingClub (2019-2020) _Loan Stats_. Retrieved from: [https://resources.lendingclub.com/](https://resources.lendingclub.com/)


