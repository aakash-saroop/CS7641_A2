import six
import sys
sys.modules['sklearn.externals.six'] = six

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve, auc
import mlrose

def preprocess_bank(df):
    df['Experience'] = df['Experience'].abs()
    return df

def run_nn_experiments(X_train, y_train, X_test, y_test, algorithms, max_iters):
    all_histories = []
    f1_scores = {alg: [] for alg in algorithms}
    
    # Define the neural network architecture
    hidden_nodes = [16, 8]
    activation = 'relu'
    
    for algorithm in algorithms:
        for max_iter in max_iters:
            print(f'\nRunning experiment with algorithm={algorithm} and max_iter={max_iter}')
            
            # Initialize neural network object and fit object
            nn_model = mlrose.NeuralNetwork(
                hidden_nodes=hidden_nodes, activation=activation,
                algorithm=algorithm, max_iters=max_iter,
                bias=True, is_classifier=True, learning_rate=0.01,
                early_stopping=True, clip_max=5, max_attempts=100, random_state=42
            )
            
            nn_model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = nn_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Test Accuracy: {accuracy:.4f}')
            
            # Evaluate predictions
            f1 = f1_score(y_test, y_pred)
            print(f'Prediction F1 Score: {f1:.4f}')
            f1_scores[algorithm].append(f1)
            
            # Store history for plotting later (if possible)
            # Note: mlrose doesn't store history in the same way as Keras, so we will just use the F1 scores.
            
            # Compute and plot confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
            disp.plot(cmap=plt.cm.Blues)
            plt.show()
    
    # Plot F1 score values for all experiments
    plt.figure(figsize=(14, 7))
    for algorithm in algorithms:
        plt.plot(max_iters, f1_scores[algorithm], label=f'F1 Score ({algorithm})')
    plt.title('F1 Score vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.show()

def bank():
    df = pd.read_csv('../data/bank.csv')
    print(df.head())
    df = preprocess_bank(df)

    X = df.drop(columns=['ID', 'ZIP Code', 'Personal Loan'])
    y = df['Personal Loan']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the range of algorithms and max_iters for experiments
    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg']
    max_iters = [50, 100, 200]

    # Run experiments with different algorithms and max_iters
    run_nn_experiments(X_train, y_train, X_test, y_test, algorithms, max_iters)

bank()
