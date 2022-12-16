![](UTA-DataScience-Logo.png)
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
  * train.csv - Training data set with features such as id, date, store_nbr, family, onpromotion, and sales
  * test.csv - Testing data set with 15 days after last date from training data
  * sample_submission.csv
  * stores.csv - Includes features that give more info on stores like city, state, type, and cluster
  * oil.csv - Daily oil prices
  * holiday_events.csv - Holiday/events metadata
*Training data set holds 3,000,888 instances with 6 features while test set has around 30,000 instances with the same features as the training set.

#### Preprocessing / Clean up

* In order to preprocess the data, I 
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

# How to reproduce results

* Results can be reproduced by downloading the code from the jupyter notebook in this repository and running it. 
* Note that data from csv files must be downloaded and loaded for the jupyter notebook to run the code.
   * For instance, if the name for training set is different, then you must manually change the string that initializes the DataFrames.
``` df_train = pd.read_csv(YOURSTRINGHERE) ```
* Also note that required packages must be installed before running code, which are provided below.

### Overview of files in repository

* The general information about this project is all in this README file, and the code that I worked on, which can be downloaded to reproduce the results I obtained, is found in the JupyterNotebook folder. 

### Software Setup
* The packages that were used for this project are numpy, pandas, matplotlib.pyplot, seaborn, statsmodels, scikitlearn, xgboost, and learntools.
  * CalendarFourier and DeterministicProcess were imported from statsmodels.
  * LinearRegression, train_test_split, mean_squared_log_error, and LabelEncoder were imported from scikitlearn.
  * XBGRegressor was imported from xgboost.
  * From learntools, plot_periodogram and seasonal_plot were imported. 
* Installing nonstandard packages (learntools):
``` 
pip install git+https://github.com/Kaggle/learntools.git 
```

### Data

* You can download the data at https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
