#Joseph D'Onofrio - 21955039
#Thomas Flynn - 40034877
#Cristian Milatinov - 40098297

import os
import sys

from ModelStatsRecorder import ModelStatsRecorder

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

# change the current directory to the directory where the script file is running
os.chdir(os.path.dirname(os.path.abspath(__file__)))

dataPath = "../datasets/drug200.csv"
deliverables_path = "./deliverables/"
classes_name = ["drugA", "drugB", "drugC", "drugX", "drugY"]
iteration_count = 10

multinomialNB_recorder = ModelStatsRecorder("MultinomialNB")
decision_tree_recorder = ModelStatsRecorder("Decision Tree")
high_performance_decision_tree_recorder = ModelStatsRecorder("High Performance Decision Tree")
perceptron_recorder = ModelStatsRecorder("Perceptron")
base_mlp_recorder = ModelStatsRecorder("BaseMLP")
high_performance_mlp_recorder = ModelStatsRecorder("High Performance MLP")


def load_data_set():
    data = pd.read_csv(dataPath)
    return data


def plot_data_set(data):
    drug_rows = data["Drug"]
    frequencies = [0, 0, 0, 0, 0]

    for drug_row in drug_rows:
        if drug_row == classes_name[0]:
            frequencies[0] += 1
        elif drug_row == classes_name[1]:
            frequencies[1] += 1
        elif drug_row == classes_name[2]:
            frequencies[2] += 1
        elif drug_row == classes_name[3]:
            frequencies[3] += 1
        elif drug_row == classes_name[4]:
            frequencies[4] += 1
        else:
            print("%s is not defined in classes, error !", drug_row)

    fig = plt.figure()
    plt.bar(classes_name, frequencies)
    plt.show()
    fig.savefig(deliverables_path + 'Drug-distribution.pdf', dpi=fig.dpi)


# Convert all ordinal and nominal features in numerical format
def convert_ordinal_nominal_to_numerical(data):
    data["Sex"] = pd.factorize(data["Sex"])[0]
    data["Cholesterol"] = pd.factorize(data["Cholesterol"])[0]
    data["BP"] = pd.factorize(data["BP"])[0]
    data["Drug"] = pd.factorize(data["Drug"])[0]
    return data


# Preprocess the data set to get it ready for classification
def convert_to_matrix_and_split(data):
    # Convert pandes dataframe to Matrix
    data_array = pd.DataFrame.to_numpy(data)

    # split matrix into data set and target
    data_target = data_array[:, -1]  # for last column
    data_set = data_array[:, :-1]  # for all but last column

    # data.target: Vector of labels (0-4) indicating the category of each article
    return data_set, data_target


# Split the data set into a training set (80%) and a test set (20%)
def split_data_set(data_set, data_target):
    training_data, test_data, training_target, test_target = train_test_split(data_set, data_target, train_size=0.8,
                                                                              random_state=None)
    return training_data, test_data, training_target, test_target


# Train the Naive Bayes classifier using the training set
def train_naive_bayes(training_data, training_target):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(training_data, training_target)
    return nb_classifier


# Use the test set to create a prediction
def test_naive_bayes(nb_classifier, test_data):
    prediction = nb_classifier.predict(test_data)
    return prediction


# Train the Decision Tree classifier using the training set
def train_decision_tree(training_data, training_target):
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(training_data, training_target)
    return dt_classifier


# Use test set on Decision Tree to create a prediction
def test_decision_tree(dt_classifier, test_data):
    prediction = dt_classifier.predict(test_data)
    return prediction


def gridcv_using_decision_tree(training_data, training_target):
    param_grid = {'criterion': ["gini", "entropy"],
                  'max_depth': [2, 3],
                  'min_samples_split': [2, 3, 4]}
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid)
    grid.fit(training_data, training_target)
    return grid


def test_gridcv_using_decision_tree(high_preformance_classifier, test_data):
    prediction = high_preformance_classifier.predict(test_data)
    return prediction


def train_perceptron(training_data, training_target):
    p_classifier = Perceptron()
    p_classifier.fit(training_data, training_target)
    return p_classifier


def test_perceptron(p_classifier, test_data):
    prediction = p_classifier.predict(test_data)
    return prediction


def train_base_mlp(training_data, training_target):
    # Default hidden layers for MLPClassifier is one layer and 100 neuron
    mlp_classifier = MLPClassifier(activation='logistic', solver='sgd')
    mlp_classifier.fit(training_data, training_target)
    return mlp_classifier


def test_base_mlp(mlp_classifier, test_data):
    prediction = mlp_classifier.predict(test_data)
    return prediction


def gridcv_using_mlp(training_data, training_target):
    param_grid = {'activation': ["identity", "logistic", "tanh", "relu"],
                  'solver': ["adam", "sgd"],
                  'hidden_layer_sizes': [(10, 10, 10), (50, 50)]}
    grid = GridSearchCV(MLPClassifier(), param_grid=param_grid)
    grid.fit(training_data, training_target)
    return grid


def test_gridcv_using_mlp(high_preformance_mlp_classifier, test_data):
    prediction = high_preformance_mlp_classifier.predict(test_data)
    return prediction


# record and print statistics relating to a prediction
def recorder_and_print_stats(test_target, prediction, target_names, section_name, best_params=None,
                             model_recorder: ModelStatsRecorder = None):
    report = classification_report(test_target, prediction, target_names=target_names, output_dict=True,
                                   zero_division=1)

    if model_recorder is not None:
        model_recorder.accuracy_samples.append(report["accuracy"])
        model_recorder.macroaverage_f1_samples.append(report["macro avg"]["f1-score"])
        model_recorder.weighted_macroaverage_f1_samples.append(report["weighted avg"]["f1-score"])

    print("============= {} =============\n".format(section_name))
    if best_params is not None:
        print("Grid CV parameters:\n")
        print(best_params)
        print("")
    print("Confusion Matrix:\n")
    print(confusion_matrix(test_target, prediction))
    print("")

    print("Classification Report:\n")
    print(classification_report(test_target, prediction, target_names=target_names, zero_division=1))


def main():
    # Generate distribution graph and load & split dataset
    data = load_data_set()
    plot_data_set(data)
    data = convert_ordinal_nominal_to_numerical(data)
    data_set, data_target = convert_to_matrix_and_split(data)
    training_data, test_data, training_target, test_target = split_data_set(data_set, data_target)

    # Open output file and redirect stdout to it
    out_file = open(deliverables_path + 'drug-performance.txt', 'w')
    sys.stdout = out_file

    for i in range(iteration_count):
        # Train & predict using MultinomialNB
        nb_classifier = train_naive_bayes(training_data, training_target)
        prediction = test_naive_bayes(nb_classifier, test_data)
        recorder_and_print_stats(test_target, prediction, classes_name, "MultinomialNB, try {}".format(i),
                                 model_recorder=multinomialNB_recorder)

        # Train & predict using Decision Tree
        dt_classifier = train_decision_tree(training_data, training_target)
        prediction = test_decision_tree(dt_classifier, test_data)
        recorder_and_print_stats(test_target, prediction, classes_name, "Decision Tree, try {}".format(i),
                                 model_recorder=decision_tree_recorder)

        # Train & predict using high performance Decision Tree
        high_performance_classifier = gridcv_using_decision_tree(training_data, training_target)
        prediction = test_gridcv_using_decision_tree(high_performance_classifier, test_data)
        recorder_and_print_stats(test_target, prediction, classes_name,
                                 "High Performance Decision Tree, try {}".format(i),
                                 high_performance_classifier.best_params_, high_performance_decision_tree_recorder)

        # Train & predict using Perceptron
        p_classifier = train_perceptron(training_data, training_target)
        prediction = test_perceptron(p_classifier, test_data)
        recorder_and_print_stats(test_target, prediction, classes_name, "Perceptron, try {}".format(i),
                                 model_recorder=perceptron_recorder)

        # Train & predict using BaseMLP
        mlp_classifier = train_base_mlp(training_data, training_target)
        prediction = test_base_mlp(mlp_classifier, test_data)
        recorder_and_print_stats(test_target, prediction, classes_name, "BaseMLP, try {}".format(i),
                                 model_recorder=base_mlp_recorder)

        # Train & predict using High Performance MLP Classifier
        high_performance_mlp_classifier = gridcv_using_mlp(training_data, training_target)
        prediction = test_gridcv_using_mlp(high_performance_mlp_classifier, test_data)
        recorder_and_print_stats(test_target, prediction, classes_name,
                                 "High Performance MLP Classifier, try {}".format(i),
                                 high_performance_mlp_classifier.best_params_, high_performance_mlp_recorder)

    print("")

    print("============= Overall Performance Results =============\n")
    multinomialNB_recorder.print_performance_stats()
    decision_tree_recorder.print_performance_stats()
    high_performance_decision_tree_recorder.print_performance_stats()
    perceptron_recorder.print_performance_stats()
    base_mlp_recorder.print_performance_stats()
    high_performance_mlp_recorder.print_performance_stats()


if __name__ == "__main__":
    main()
