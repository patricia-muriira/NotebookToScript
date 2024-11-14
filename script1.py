# Activate environment- .\.venv\Scripts\activate
# Necessary imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Data loading and preprocessing


def DataLoad():
    cancer_df = pd.read_csv('cervical_cancer.csv')
    cancer_df = cancer_df.replace('?', np.nan)
    cancer_df = cancer_df.drop(
        ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
    cancer_df = cancer_df.apply(pd.to_numeric)
    cancer_df = cancer_df.fillna(cancer_df.mean())
    return cancer_df

# Splitting target and features


def SplitData(cancer_df):
    target_df = cancer_df['Biopsy']
    input_df = cancer_df.drop(['Biopsy'], axis=1)
    X = np.array(input_df).astype('float32')
    y = np.array(target_df).astype('float32').reshape(-1, 1)
    return X, y

# Scaling features


def ScaleFeatures(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

# Splitting into train, validation, and test sets


def SplitTrainValTest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Initializing and training the model


def TrainModel(X_train, y_train):
    model = xgb.XGBClassifier(
        learning_rate=0.1, max_depth=50, n_estimators=100)
    model.fit(X_train, y_train)
    pickle.dump(model, open('cervical-cancer-model.pkl', 'wb'))
    return model


# Evaluating the model
def EvaluateModel(model, X_train, y_train, X_test, y_test):
    result_train = model.score(X_train, y_train)
    result_test = model.score(X_test, y_test)
    return result_train, result_test


cancer_df = DataLoad()
X, y = SplitData(cancer_df)
X = ScaleFeatures(X)
X_train, X_val, X_test, y_train, y_val, y_test = SplitTrainValTest(X, y)
model = TrainModel(X_train, y_train)
train_score, test_score = EvaluateModel(
    model, X_train, y_train, X_test, y_test)

<<<<<<< HEAD
print(f"Training Score: {train_score}")
print(f"Test Score: {test_score}")
=======
#clasification report
y_predict = model.predict(X_test)
print(classification_report(y_test, y_predict))
#confusion matrix
cm = confusion_matrix(y_predict, y_test)
sns.heatmap(cm, annot = True)
plt.show()

#add loggers
#modularize
>>>>>>> 1ca189bc83c5fc7f9cb58304abcc8c8456506822
