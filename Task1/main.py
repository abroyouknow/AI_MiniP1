#Joseph D'Onofrio - 21955039
#Thomas Flynn - 40034877
#Cristian Milatinov - 40098297

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

# change the current directory to the directory where the script file is running
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_path = "../datasets/BBC/"
deliverables_path = "./deliverables/"


# Generate bar graph distribution for all classes
def create_distribution_graph():
    classes = os.listdir(data_path)
    classes = list(filter(lambda d: os.path.isdir(data_path + d), classes))
    frequencies = []

    for classDir in classes:
        frequencies.append(len(os.listdir(data_path + classDir)))

    fig = plt.figure()
    plt.bar(classes, frequencies)
    plt.show()
    fig.savefig(deliverables_path + 'bbc-distribution.pdf', dpi=fig.dpi)

    return classes


# Preprocess the data set to get it ready for classification
def preprocess_data_set():
    data = load_files(data_path, encoding='latin-1')
    vectorizer = CountVectorizer(analyzer='word')

    # data_set: Matrix of dimension (M = number of articles x N = vocabulary size)
    # This is the number of times a word in vocab appears in a specific article
    data_set = vectorizer.fit_transform(data.data)

    # data.target: Vector of labels (0-4) indicating the category of each article
    return data_set, data.target, vectorizer


# Split the data set into a training set (80%) and a test set (20%)
def split_data_set(data_set, data_target):
    training_data, test_data, training_target, test_target = train_test_split(data_set, data_target, train_size=0.8, random_state=None)
    return training_data, test_data, training_target, test_target


# Train the Naive Bayes classifier using the training set
def train_naive_bayes(training_data, training_target, smoothing_value=1.0):
    nb_classifier = MultinomialNB(alpha=smoothing_value)
    nb_classifier.fit(training_data, training_target)
    return nb_classifier


# Use the test set to create a prediction
def test_naive_bayes(nb_classifier, test_data):
    prediction = nb_classifier.predict(test_data)
    return prediction


# Print statistics relating to a prediction
def print_stats(section_name, all_data, all_target, training_target, test_target, target_names, prediction, vectorizer):

    def calc_prior_probabilities():
        prior_prob = [0 for _ in target_names]
        for i in range(training_target.size):
            prior_prob[training_target[i]] = prior_prob[training_target[i]] + 1
        for i in range(len(prior_prob)):
            prior_prob[i] = prior_prob[i] / training_target.size
        return prior_prob

    def calc_class_word_tokens():
        summed_cols = all_data.sum(axis=1)
        num_words = [0 for _ in target_names]
        for i in range(summed_cols.shape[0]):
            num_words[all_target[i]] = num_words[all_target[i]] + summed_cols[i, 0]
        return num_words

    def calc_num_words_zero():
        num_words_zero = [0 for _ in target_names]
        for i in range(len(target_names)):
            rows_to_sum = []
            for k in range(all_target.size):
                if all_target[k] == i:
                    rows_to_sum.append(k)
            summed_rows = all_data[rows_to_sum, :].sum(axis=0)
            num_words_zero[i] = num_words_zero[i] + np.count_nonzero(summed_rows == 0)
        return num_words_zero

    def calc_num_words_one():
        rows_total = all_data.sum(axis=0)
        num_words_one = np.count_nonzero(rows_total == 1)
        return num_words_one

    def calc_word_log_prob(word, total_word_tokens):
        return math.log(vectorizer.vocabulary_[word] / total_word_tokens)

    print("============= {} =============\n".format(section_name))
    print("Confusion Matrix:\n")
    print(confusion_matrix(test_target, prediction))
    print("")

    print("Classification Report:\n")
    print(classification_report(test_target, prediction, target_names=target_names))

    print("Prior Probabilities:\n")
    prior_prob = calc_prior_probabilities()
    for i in range(len(target_names)):
        print("\t{:20} {:.5f}".format(target_names[i], prior_prob[i]))

    vocab_size = all_data.shape[1]
    print("\nVocabulary Size:\n\n\t{} words".format(vocab_size))

    print("\nTotal word-tokens per class:\n")
    class_word_tokens = calc_class_word_tokens()
    for i in range(len(target_names)):
        print("\t{:20} {:<10d}".format(target_names[i], class_word_tokens[i]))

    total_word_tokens = sum(class_word_tokens)
    print("\nTotal word-tokens in entire corpus:\n\n\t{} words".format(total_word_tokens))

    print("\nNumber of words with frequency of zero in each class:\n")
    num_words_zero = calc_num_words_zero()
    for i in range(len(target_names)):
        print("\t{:20} {:<8d} ({:5.2f} %)".format(target_names[i], num_words_zero[i], 100 * num_words_zero[i] / vocab_size))

    num_words_one = calc_num_words_one()
    print("\nNumber of words with frequency of one in entire corpus:\n\n\t{} words ({:5.2f} %)"
          .format(num_words_one, 100 * num_words_one / vocab_size))

    print("\nLog (base e) probability of our 2 favorite words:\n")
    cat_prob = calc_word_log_prob('cat', total_word_tokens)
    dog_prob = calc_word_log_prob('dog', total_word_tokens)
    print("\t{:10} {:.5f}".format("cat", cat_prob))
    print("\t{:10} {:.5f}".format("dog", dog_prob))
    print()


def main():
    # Generate distribution graph and load & split dataset
    list_of_classes = create_distribution_graph()
    all_data, all_target, vectorizer = preprocess_data_set()
    training_data, test_data, training_target, test_target = split_data_set(all_data, all_target)

    # Open output file and redirect stdout to it
    out_file = open(deliverables_path + 'bbc-performance.txt', 'w')
    sys.stdout = out_file

    # Train & predict try #1
    nb_classifier1 = train_naive_bayes(training_data, training_target)
    prediction1 = test_naive_bayes(nb_classifier1, test_data)
    print_stats("MultinomialNB, default values, try 1", all_data, all_target, training_target, test_target, list_of_classes, prediction1, vectorizer)

    # Train & predict try #2
    nb_classifier2 = train_naive_bayes(training_data, training_target)
    prediction2 = test_naive_bayes(nb_classifier2, test_data)
    print_stats("MultinomialNB, default values, try 2", all_data, all_target, training_target, test_target, list_of_classes, prediction2, vectorizer)

    # Train & predict try #3
    nb_classifier3 = train_naive_bayes(training_data, training_target, smoothing_value=0.0001)
    prediction3 = test_naive_bayes(nb_classifier3, test_data)
    print_stats("MultinomialNB, smoothing = 0.0001", all_data, all_target, training_target, test_target, list_of_classes, prediction3, vectorizer)

    # Train & predict try #4
    nb_classifier4 = train_naive_bayes(training_data, training_target, smoothing_value=0.9)
    prediction4 = test_naive_bayes(nb_classifier4, test_data)
    print_stats("MultinomialNB, smoothing = 0.9", all_data, all_target, training_target, test_target, list_of_classes, prediction4, vectorizer)


if __name__ == "__main__":
    main()
