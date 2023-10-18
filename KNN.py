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
X_train_scaled = pd.DataFrame(X_train_scaled, columns=[X_train.columns])
X_test_scaled = pd.DataFrame(X_test_scaled, columns=[X_train.columns])

# TRAIN KNN MODEL ON SCALED DATA USING RANDOM K VALUE ----------------------------
from sklearn.neighbors import KNeighborsClassifier

# Create instance
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train_scaled, y_train)

# USE MODEL TO PREDICT TEST VALUES -----------------------------------------------
predictions = KNN.predict(X_test_scaled)

# EVALUATE MODEL -----------------------------------------------------------------
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,predictions))
print("\n")
print(classification_report(y_test,predictions))

# OPTIMIZE VALUE OF K (USING ELBOW METHOD) ---------------------------------------
error_rate = []

for i in range(1,60):
    # Predict value with new k value
    KNN_i = KNeighborsClassifier(n_neighbors=i)
    KNN_i.fit(X_train_scaled,y_train)
    predictions_i = KNN_i.predict(X_test_scaled)

    # Find mean of all the times y_test doesn'r equal predictions
    error =  np.mean(y_test != predictions_i)
    error_rate.append(error)

error_rate

# Make a df with k value and error_rate
error_rate_df = pd.DataFrame(data = error_rate, columns=["error rate"], index=range(1,60))

# Make a line graph of the df
plt.figure(figsize=(10,6))
plt.plot(error_rate_df, linestyle = "--", marker="o", markerfacecolor="red", markersize=7)
plt.title("Error Rate vs. K Value")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.show()