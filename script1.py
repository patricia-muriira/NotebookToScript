#Activate environment- .\.venv\Scripts\activate
# Necessary imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os
print(os.getcwd())


# Data loading and preprocessing
cancer_df = pd.read_csv('cervical_cancer.csv')
cancer_df = cancer_df.replace('?', np.nan)
cancer_df = cancer_df.drop(
    ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
cancer_df = cancer_df.apply(pd.to_numeric)
cancer_df = cancer_df.fillna(cancer_df.mean())

# Splitting target and features
target_df = cancer_df['Biopsy']
input_df = cancer_df.drop(['Biopsy'], axis=1)
X = np.array(input_df).astype('float32')
y = np.array(target_df).astype('float32').reshape(-1, 1)

# Scaling features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

# Initializing and training the model
model = xgb.XGBClassifier(learning_rate=0.1, max_depth=50, n_estimators=100)
model.fit(X_train, y_train)

# Evaluating the model
result_train = model.score(X_train, y_train)
print("Training Accuracy:", result_train)

result_test = model.score(X_test, y_test)
print("Test Accuracy:", result_test)

#Visualizing model performance
#Generating predictions and metrics

#clasification report
y_predict = model.predict(X_test)
print(classification_report(y_test, y_predict))
#confusion matrix
cm = confusion_matrix(y_predict, y_test)
sns.heatmap(cm, annot = True)
plt.show()

#add loggers
#modularize