# IMPORT PACKAGES ---------------------------------------------------------------
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# LOAD DATASET -------------------------------------------------------------------
# Read data and set first column as index
df = pd.read_csv("ClassifiedData", index_col=0)
df.info()
df.head()
df.describe()

# SPLIT DATA INTO TRAIN AND TEST -------------------------------------------------
from sklearn.model_selection import train_test_split

X = df.drop(["TARGET CLASS"], axis = 1)
y = df["TARGET CLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# STANDARDIZE THE FEATURES (X DATA) ----------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# fit transform on training X data and just transform for the testing X data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#change the scaled data into df
X_train_scaled
X_train


