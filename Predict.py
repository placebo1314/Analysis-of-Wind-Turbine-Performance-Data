import array
import os
import joblib
import numpy as np
from numpy import vectorize
import pandas as pd
from pyspark import SparkConf
from pyspark.ml.feature import VectorAssembler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor
from pyspark.sql import SparkSession
from sklearn.neural_network import MLPRegressor


# pip install pyspark
# pip install keras
# pip install tensorflow

from Main import optimize_dateformat


def predict(wind_speed, month, wind_direction=None, hour=None):
    # Load the trained machine learning model
    model = joblib.load("wind_turbine_model.pkl")

    # input features
    input_dict = {
        "wind_speed": wind_speed,
        "month": month
    }

    # Add optional features if they are provided
    if wind_direction is not None:
        input_dict["wind_direction"] = wind_direction
    if hour is not None:
        input_dict["hour"] = hour

    # Make the prediction using the loaded model and the input features
    predicted_output = model.predict([input_dict])[0]

    return predicted_output

def train_model_spark(data):
    #Solving the hadoop error:
    os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.hadoop.fs.s3a.experimental.input.fadvise=sequentialRead --conf spark.hadoop.fs.s3a.experimental.output.fadvise=sequential --conf spark.hadoop.fs.s3a.fast.upload=true pyspark-shell'
    
    # Create a Spark session
    conf = SparkConf().set('spark.executor.memory', '4g')
    spark = SparkSession.builder.config(conf=conf) \
    .appName('WindTurbineAnalysis') \
    .getOrCreate()
    
    data['month'] = data.index.month
    data['hour'] = data.index.hour
    df_spark = spark.createDataFrame(data)
    # #In pandas:
    #train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    splits = df_spark.randomSplit([0.8, 0.2])
    train_df = splits[0]
    test_df = splits[1]
    # Define the input and output features
    input_features = ['Wind Speed (m/s)', 'month', 'Wind Direction (°)', 'hour']
    output_feature = 'LV ActivePower (kW)'

    # Transform the training and testing data using the vector assembler
    assembler = VectorAssembler(inputCols=input_features, outputCol='features')
    train_df = assembler.transform(train_df).select('features', output_feature)
    test_df = assembler.transform(test_df).select('features', output_feature)

# Convert the Spark DataFrames to NumPy arrays
    train_X = array([vectorize.dense(x) for x in train_df.rdd.map(lambda x: x.features).collect()])
    train_y = array(train_df.select(output_feature).rdd.map(lambda x: x[0]).collect())

# Train a random forest regressor model
    rf_model = RandomForestRegressor()

# Fit the model to the training data
    rf_model.fit(train_X, train_y)

    # Evaluate the model on the testing data
    predictions = rf_model.predict(array([vectorize.dense(x) for x in test_df.select('features').collect()]))
    mse = mean_squared_error(test_df.select(output_feature).rdd.map(lambda x: x[0]).collect(), predictions)
    print('Mean squared error:', mse)

    # Save the trained model to disk
    joblib.dump(rf_model, 'wind_turbine_model.pkl')

    spark.stop()

def create_neuraln_model(input_shape):
    model = Sequential()
    model.add(Dense(16, input_shape=input_shape, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model_pandas(data, neuraln_model):
    data['month'] = data.index.month
    data['hour'] = data.index.hour

    # split data into training and testing sets
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    # Define the input and output features
    input_features = ['Wind Speed (m/s)', 'month', 'Wind Direction (°)', 'hour']
    output_feature = 'LV ActivePower (kW)'

    # Prepare the training data
    train_X = train_df[input_features].values
    train_y = train_df[output_feature].values

    # Train models
    rf_model = RandomForestRegressor()

    lin_model = LinearRegression()

    dec_model = DecisionTreeRegressor()

    # Fit the models
    rf_model.fit(train_X, train_y)

    lin_model.fit(train_X, train_y)

    dec_model.fit(train_X, train_y)

    neuraln_model.fit(train_X, train_y, epochs=50, batch_size=32)
    selector = SequentialFeatureSelector(estimator=neuraln_model)

    # Prepare the testing data
    test_X = test_df[input_features].values

    # Evaluate the model on the testing data
    predictions_rf = rf_model.predict(test_X)
    rf_mse = mean_squared_error(test_df[output_feature].values, predictions_rf)
    print('Mean squared error of RandomForest:', rf_mse)

    predictions_lin = lin_model.predict(test_X)
    lin_mse = mean_squared_error(test_df[output_feature].values, predictions_lin)
    print('Mean squared error of LinearRegression:', lin_mse)

    predictions_dec = dec_model.predict(test_X)
    dec_mse = mean_squared_error(test_df[output_feature].values, predictions_dec)
    print('Mean squared error of DecisionTreeRegressor:', dec_mse)

    predictions_neuraln = neuraln_model.predict(test_X)
    neuraln_mse = mean_squared_error(test_df[output_feature].values, predictions_neuraln)
    print('Mean squared error of Neural Network:', neuraln_mse)

    # Save the trained model to disk
    joblib.dump(rf_model, 'wind_turbine_model.pkl')
    
def main():
    data = optimize_dateformat(pd.read_csv('wind_turbine_data.csv'))
    # Create the neural network model
    neuraln_model = create_neuraln_model(input_shape=(4,))

    # Train the models using the neural network
    train_model_pandas(data, neuraln_model)
    
main()