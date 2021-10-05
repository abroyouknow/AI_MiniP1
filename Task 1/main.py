import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

dataPath = "../datasets/BBC/"
deliverablesPath = "./deliverables/"

# Generate bar graph distribution for all classes
def CreateDistributionGraph():
    classes = os.listdir(dataPath)
    classes = list(filter(lambda d: os.path.isdir(dataPath + d), classes))
    frequencies = []

    for classDir in classes:
        frequencies.append(len(os.listdir(dataPath + classDir)))

    if not os.path.isfile(deliverablesPath + 'BBC-distribution.pdf'):
        fig = plt.figure()
        plt.bar(classes, frequencies)
        plt.show()
        fig.savefig(deliverablesPath + 'BBC-distribution.pdf', dpi=fig.dpi)

    return classes

# Preprocess the data set to get it ready for classification
def PreprocessDataset():
    data = load_files(dataPath, encoding='latin-1')
    vectorizer = CountVectorizer(analyzer='word')

    # data_set: Matrix of dimension (M = number of articles x N = vocabulary size)
    # This is the number of times a word in vocab appears in a specific article
    data_set = vectorizer.fit_transform(data.data)

    # data.target: Vector of labels (0-4) indicating the category of each article
    return data_set, data.target, vectorizer


# Split the data set into a training set (80%) and a test set (20%)
def SplitDataSet(data_set, data_target):
    training_data, test_data, training_target, test_target = train_test_split(data_set, data_target, train_size=0.8, random_state=None)
    return training_data, test_data, training_target, test_target

# Train the Naive Bayes classifier using the training set
def TrainNaiveBayes(training_data, training_target):
    nb_classifier = MultinomialNB()
    nb_classifier.fit(training_data, training_target)
    return nb_classifier

# Use the test set to create a prediction
def TestNaiveBayes(nb_classifier, test_data):
    prediction = nb_classifier.predict(test_data)
    print(prediction)
    print(len(prediction))
    return prediction

# Print statistics relating to a prediction
def PrintStats(actual, prediction, target_names, vectorizer):
    print(confusion_matrix(actual, prediction))
    print(classification_report(actual, prediction, target_names=target_names))
    print("Num of Word: " + str(len(vectorizer.vocabulary_)))


def main():
    list_of_classes = CreateDistributionGraph()
    data_set, data_target, vectorizer = PreprocessDataset()
    training_data, test_data, training_target, test_target = SplitDataSet(data_set, data_target)
    nb_classifier = TrainNaiveBayes(training_data, training_target)
    prediction = TestNaiveBayes(nb_classifier, test_data)
    PrintStats(prediction, test_target, list_of_classes, vectorizer)


if __name__ == "__main__":
    main()
