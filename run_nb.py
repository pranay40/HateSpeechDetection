from __future__ import division
import os
import numpy 
import pickle
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split  # Random split into training and test dataset.
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
#from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from build_feature_vector import *
from format_data import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

id_tweet_map = create_id_tweet_map()
id_class_map = create_id_class_map()

X, Y = TrainingData(id_tweet_map, id_class_map)

# Perform feature selection as before
X = numpy.asarray(X)
Y = numpy.asarray(Y).ravel()
X = SelectKBest(chi2, k=1200).fit_transform(X, Y)

# Use grid search to find the optimal value for alpha
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, Y)

# Print the best hyperparameter values and cross-validation accuracy
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Use the best hyperparameter values to train the final model
best_alpha = grid_search.best_params_['alpha']
final_classifier = MultinomialNB(alpha=best_alpha)

# Initialize lists to store evaluation metrics for each fold
precision_list, recall_list, f1_list = [], [], []

accuracy = 0
fold = 0

kf = KFold(n_splits=15, shuffle=True, random_state=42)
# Iterate through the folds
for train_idx, test_idx in kf.split(X):
    fold += 1
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    print(f"Fold {fold} - Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Print actual class distribution in each fold
    print(f"Actual class distribution in fold {fold}:")
    print(pd.Series(Y_test).value_counts())

    # Train the final model with the best hyperparameters
    final_classifier.fit(X_train, Y_train)

    # Make predictions on the test set
    predictions = final_classifier.predict(X_test)

    # Evaluate accuracy for this fold
    score = accuracy_score(Y_test, predictions)
    accuracy += score
    print(f"Score for fold {fold}: {score}")

    # Calculate and print precision, recall, and F1-score
    try:
        # Calculate and print precision, recall, and F1-score using 'micro' averaging
        precision = precision_score(Y_test, predictions, average='micro', zero_division=0)
        recall = recall_score(Y_test, predictions, average='micro')
        f1 = f1_score(Y_test, predictions, average='micro')

        print(f"Precision for fold {fold}: {precision}")
        print(f"Recall for fold {fold}: {recall}")
        print(f"F1-score for fold {fold}: {f1}")

        # Append values to the lists for later averaging
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        # Plot the confusion matrix
        # cm = confusion_matrix(Y_test, predictions)
        # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.title(f'Confusion matrix - Fold {fold}')
        # plt.colorbar()
        # plt.show()

    except Exception as e:
        print(f"Error in fold {fold}: {e}")

# print("Accuracy : ", round(accuracy / 10, 3))
print("Average Accuracy : {:.2%}".format(accuracy / 15))
print("Average Precision: {:.2%}".format(np.mean(precision_list)))
print("Average Recall: {:.2%}".format(np.mean(recall_list)))
print("Average F1-score: {:.2%}".format(np.mean(f1_list)))
