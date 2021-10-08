import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

dataPath = "../datasets/drug200.csv"
deliverablesPath = "./deliverables/"
classes_name = ["drugA", "drugB", "drugC", "drugX", "drugY"]


def LoadDataSet():
    data = pd.read_csv(dataPath)
    return data


def PlotDataSet(data):
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

    if not os.path.isfile(deliverablesPath + 'Drug-distribution.pdf'):
        fig = plt.figure()
        plt.bar(classes_name, frequencies)
        plt.show()
        fig.savefig(deliverablesPath + 'Drug-distribution.pdf', dpi=fig.dpi)


# Convert all ordinal and nominal features in numerical format
def ConvertOrdinalNominalToNumerical(data):
    data["Sex"] = pd.factorize(data["Sex"])[0]
    data["Cholesterol"] = pd.factorize(data["Cholesterol"])[0]
    data["BP"] = pd.factorize(data["BP"])[0]
    data["Drug"] = pd.factorize(data["Drug"])[0]
    return data


# Preprocess the data set to get it ready for classification
def ConvertToMatrixAndSplit(data):
    # Convert pandes dataframe to Matrix
    data_matrix = pd.DataFrame.to_numpy(data)

    # split matrix into data set and target
    data_set = data_matrix[:, -1]  # for last column
    data_target = data_matrix[:, :-1]  # for all but last column

    # data.target: Vector of labels (0-4) indicating the category of each article
    return data_set, data_target,


# Split the data set into a training set (80%) and a test set (20%)
def SplitDataSet(data_set, data_target):
    training_data, test_data, training_target, test_target = train_test_split(data_set, data_target, train_size=0.8,
                                                                              random_state=None)
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
def PrintStats(actual, prediction, target_names):
    print(confusion_matrix(actual, prediction))
    print(classification_report(actual, prediction, target_names=target_names))


def main():
    data = LoadDataSet()
    PlotDataSet(data)
    data = ConvertOrdinalNominalToNumerical(data)
    data_set, data_target = ConvertToMatrixAndSplit(data)
    training_data, test_data, training_target, test_target = SplitDataSet(data_set, data_target)

    # TODO: Check Data_set, data_target compared to training_data, test_data, training_target, test_target. There's
    #  an error
    print("----------------------------------")
    print("Naive Bayes Model")
    print("----------------------------------")
    nb_classifier = TrainNaiveBayes(training_data, training_target)
    prediction = TestNaiveBayes(nb_classifier, test_data)
    PrintStats(prediction, test_target, classes_name)



if __name__ == "__main__":
    main()
