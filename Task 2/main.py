import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report

dataPath = "../datasets/drug200.csv"
deliverables_path = "./deliverables/"
classes_name = ["drugA", "drugB", "drugC", "drugX", "drugY"]


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

    if not os.path.isfile(deliverables_path + 'Drug-distribution.pdf'):
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
    mlp_classifier = MLPClassifier()
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


# Print statistics relating to a prediction
def print_stats(actual, prediction, target_names):
    print(confusion_matrix(actual, prediction))
    print(classification_report(actual, prediction, target_names=target_names))


def main():
    # Generate distribution graph and load & split dataset
    data = load_data_set()
    plot_data_set(data)
    data = convert_ordinal_nominal_to_numerical(data)
    data_set, data_target = convert_to_matrix_and_split(data)
    training_data, test_data, training_target, test_target = split_data_set(data_set, data_target)

    print("----------------------------------")
    print("Naive Bayes Model")
    print("----------------------------------")
    nb_classifier = train_naive_bayes(training_data, training_target)
    prediction = test_naive_bayes(nb_classifier, test_data)
    print_stats(prediction, test_target, classes_name)

    print("----------------------------------")
    print("Decision Tree")
    print("----------------------------------")
    dt_classifier = train_decision_tree(training_data, training_target)
    prediction = test_decision_tree(dt_classifier, test_data)
    print_stats(prediction, test_target, classes_name)

    print("----------------------------------")
    print("High Performance Decision Tree")
    print("----------------------------------")
    high_performance_classifier = gridcv_using_decision_tree(training_data, training_target)
    prediction = test_gridcv_using_decision_tree(high_performance_classifier, test_data)
    print(high_performance_classifier.best_params_)
    print_stats(prediction, test_target, classes_name)

    # TODO: Check warining -> UndefinedMetricWarning: Recall and F-score are ill-defined and being set to
    #  0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
    print("----------------------------------")
    print("Perceptron")
    print("----------------------------------")
    p_classifier = train_perceptron(training_data, training_target)
    prediction = test_perceptron(p_classifier, test_data)
    print_stats(prediction, test_target, classes_name)

    # TODO: Check warining -> UndefinedMetricWarning: Recall and F-score are ill-defined and being set to
    #  0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
    # TODO: Investigate -> ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and
    #  the optimization hasn't converged yet. warnings.warn(
    print("----------------------------------")
    print("BaseMLP")
    print("----------------------------------")
    mlp_classifier = train_base_mlp(training_data, training_target)
    prediction = test_base_mlp(mlp_classifier, test_data)
    print_stats(prediction, test_target, classes_name)

    # TODO: Check warining -> UndefinedMetricWarning: Recall and F-score are ill-defined and being set to
    #  0.0 in labels with no true samples. Use `zero_division` parameter to control this behavior.
    # TODO: Investigate -> ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and
    #  the optimization hasn't converged yet. warnings.warn(
    print("----------------------------------")
    print("High Performance MLP Classifier")
    print("----------------------------------")
    high_performance_mlp_classifier = gridcv_using_mlp(training_data, training_target)
    prediction = test_gridcv_using_mlp(high_performance_mlp_classifier, test_data)
    print(high_performance_mlp_classifier.best_params_)
    print_stats(prediction, test_target, classes_name)


if __name__ == "__main__":
    main()
