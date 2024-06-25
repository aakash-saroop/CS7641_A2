import six
import sys
sys.modules['sklearn.externals.six'] = six

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import mlrose

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def preprocess_bank(df):
    df['Experience'] = df['Experience'].abs()
    return df

def run_a1_nn_experiment(X_train, y_train, X_test, y_test, learning_rate, epochs):
    # Define the model architecture
    model = Sequential([
        Dense(16, input_dim=X_train.shape[1], activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    end_time = time.time()
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Make predictions
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    # Evaluate predictions
    f1 = f1_score(y_test, y_pred)
    
    training_time = end_time - start_time
    
    return history, accuracy, f1, training_time

def run_a2_nn_experiment(X_train, y_train, X_test, y_test, algorithm, max_iters):
    hidden_nodes = [16, 8]
    activation = 'relu'
    
    # Initialize neural network object and fit object
    nn_model = mlrose.NeuralNetwork(
        hidden_nodes=hidden_nodes, activation=activation,
        algorithm=algorithm, max_iters=max_iters,
        bias=True, is_classifier=True, learning_rate=0.01,
        early_stopping=True, clip_max=5, max_attempts=100, random_state=42
    )
    
    start_time = time.time()
    nn_model.fit(X_train, y_train)
    end_time = time.time()
    
    # Evaluate the model
    y_pred = nn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Evaluate predictions
    f1 = f1_score(y_test, y_pred)
    
    training_time = end_time - start_time
    
    return accuracy, f1, training_time

def plot_learning_curves(history, algorithm, learning_rate, epochs):
    # Plot training & validation accuracy values
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['accuracy'], label=f'Train Acc ({algorithm}, lr={learning_rate}, epochs={epochs})')
    plt.plot(history.history['val_accuracy'], label=f'Val Acc ({algorithm}, lr={learning_rate}, epochs={epochs})')
    plt.title(f'Model Accuracy ({algorithm})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(14, 7))
    plt.plot(history.history['loss'], label=f'Train Loss ({algorithm}, lr={learning_rate}, epochs={epochs})')
    plt.plot(history.history['val_loss'], label=f'Val Loss ({algorithm}, lr={learning_rate}, epochs={epochs})')
    plt.title(f'Model Loss ({algorithm})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
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

    # Assignment 1 (A1) experiments
    learning_rate = 0.01
    epochs_list = [50, 100, 200]
    a1_results = []

    for epochs in epochs_list:
        print(f'\nRunning A1 experiment with learning rate={learning_rate} and epochs={epochs}')
        history, accuracy, f1, training_time = run_a1_nn_experiment(X_train, y_train, X_test, y_test, learning_rate, epochs)
        a1_results.append((learning_rate, epochs, accuracy, f1, training_time))
        plot_learning_curves(history, 'Backpropagation', learning_rate, epochs)
    
    # Assignment 2 (A2) experiments
    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg']
    max_iters = [50, 100, 200]
    a2_results = {alg: [] for alg in algorithms}
    
    for algorithm in algorithms:
        for max_iter in max_iters:
            print(f'\nRunning A2 experiment with algorithm={algorithm} and max_iter={max_iter}')
            accuracy, f1, training_time = run_a2_nn_experiment(X_train, y_train, X_test, y_test, algorithm, max_iter)
            a2_results[algorithm].append((max_iter, accuracy, f1, training_time))
    
    # Compare results
    plt.figure(figsize=(14, 7))
    for lr, epochs, accuracy, f1, training_time in a1_results:
        plt.scatter(epochs, accuracy, label=f'A1 Backprop (epochs={epochs}, acc={accuracy:.4f})')
    for algorithm in algorithms:
        for max_iter, accuracy, f1, training_time in a2_results[algorithm]:
            plt.scatter(max_iter, accuracy, label=f'A2 {algorithm} (iters={max_iter}, acc={accuracy:.4f})')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs/Iterations')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    
    plt.figure(figsize=(14, 7))
    for lr, epochs, accuracy, f1, training_time in a1_results:
        plt.scatter(epochs, f1, label=f'A1 Backprop (epochs={epochs}, f1={f1:.4f})')
    for algorithm in algorithms:
        for max_iter, accuracy, f1, training_time in a2_results[algorithm]:
            plt.scatter(max_iter, f1, label=f'A2 {algorithm} (iters={max_iter}, f1={f1:.4f})')
    plt.title('F1 Score Comparison')
    plt.xlabel('Epochs/Iterations')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    plt.show()
    
    plt.figure(figsize=(14, 7))
    for lr, epochs, accuracy, f1, training_time in a1_results:
        plt.scatter(epochs, training_time, label=f'A1 Backprop (epochs={epochs}, time={training_time:.2f}s)')
    for algorithm in algorithms:
        for max_iter, accuracy, f1, training_time in a2_results[algorithm]:
            plt.scatter(max_iter, training_time, label=f'A2 {algorithm} (iters={max_iter}, time={training_time:.2f}s)')
    plt.title('Training Time Comparison')
    plt.xlabel('Epochs/Iterations')
    plt.ylabel('Training Time (s)')
    plt.legend(loc='upper left')
    plt.show()

bank()
