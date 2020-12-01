# Importing external libaries
import pandas as pd
import numpy as np
from sklearn import clone
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def preprocess(df2):
    """
        Preprocess data
    """
    # remove null values
    df2.dropna(axis=0, inplace=True)

    # convert date to proper format
    df2['date'] = pd.to_datetime(df2['date'])

    # create new features
    df2['year'] = df2['date'].dt.year
    df2['month'] = df2['date'].dt.month
    df2['day'] = df2['date'].dt.day

    # create target variable
    df2['target'] = np.where(df2['close'] < df2['open'], 0, 1)
    y_train = df2['target'].copy()

    df2.drop(columns=['date', 'target'], inplace=True)

    # One-Hot Encoding
    df2 = pd.get_dummies(df2, columns=['Name'])

    # Scaling data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df2)

    return X_train, y_train


def train_clf(df1):
    """
        Train classifier
    """
    X_train, y_train = preprocess(df1)

    # train model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    return model


def predict(ticker, date, data):
    """
        Predict 0 for Red and 1 for Green result.
    """
    # read data from csv file
    df_raw = pd.read_csv("data.csv")

    # get trained model
    model = train_clf(df_raw)

    # preprocess data for prediction
    data['Name'] = ticker
    data['date'] = date
    X_test, _ = preprocess(data)

    y_pred = model.predict(X_test)

    return y_pred


# sample testing data. Replace this with your testing data.
d = {'open': [16.07], 'high': [14.51], 'low': [
    14.25], 'close': [14.46], 'volume': [8126000]}
data = pd.DataFrame(data=d)

# getting predictions
predictions = predict("GHBE", "2015-02-09", data)

print(predictions)
