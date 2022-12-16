# Stores Sales - Time Series Forecasting

* **This repository holds an attempt to apply time-series forecasting techniques using data from
the Store Sales - Time Series Forecasting Kaggle challenge (https://www.kaggle.com/competitions/store-sales-time-series-forecasting).**

## Overview

* The task was to use the training data provided by the Kaggle challenge to predict the target sales for 15 days after the last date in the training data set.
* My approach was to look at the trend and seasonality of the data to perform these predictions as well as the use of hybrid models and machine learning algorithms.
* I have not checked the performance of my model as of yet.

## Summary of Workdone

### Data

* Data (124.76MB):
  * train.csv - Training data set with features such as store_nbr, family, onpromotion, and sales
  * test.csv - Testing data set with 15 days after last date from training data
  * sample_submission.csv
  * stores.csv - Includes features that give more info on stores like city, state, type, and cluster
  * oil.csv - Daily oil prices
  * holiday_events.csv - Holiday/events metadata

#### Preprocessing / Clean up

* Modified original dataframe to show average sales of each day so it would be easier to visualize and use the data.

#### Data Visualization

### Problem Formulation

* Define:
  * Input: Features (store_nbr, family, onpromotion)
  * Output: Sales
  * Models
    * First, I looked at the trend of the average sales by making a trend model and tried predicting the sales of the next 15 days using a validation set and checking the Root Mean Squared Logarithmic Error (RMSLE) scores.
    * Then, I looked at the seasonality of the data set and tried to get better predictions this way by taking into account holiday data.
    * I also attempted using a Hybrid model scikitlearn's LinearRegression model and xgboost's XGBRegressor model

### Training

* When trying to train data using hybrid model, an error occurred, so I was not able to determine how long the training would take.

### Conclusions

* When working with time series data, it is important to take into consideration the three patterns of dependence (trend, seasons, and cycles) and use this information to design and use machine learning algorithms that effectively predict future sales.

### Future Work

* I would first fix my implementation of a hybrid model to see how well prediections are.
* I would then move on to applying other machine learning from scikit-learn to see which would be best to use.

### Software Setup
* numpy, pandas, matplotlib, statsmodels, scikitlearn, learntools
* Installing nonstandard packages (learntools):
``` 
pip install git+https://github.com/Kaggle/learntools.git 
```

### Data

* You can download the data at https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
