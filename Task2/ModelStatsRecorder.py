import statistics


class ModelStatsRecorder:
    def __init__(self, section_name):
        self.section_name = section_name
        self.accuracy_samples = []
        self.macroaverage_f1_samples = []
        self.weighted_macroaverage_f1_samples = []

    def print_performance_stats(self):
        print("============= {} =============\n".format(self.section_name))
        print("Average accuracy: {}\n".format(statistics.mean(self.accuracy_samples)))
        print("Average macro-average F1: {}\n".format(statistics.mean(self.macroaverage_f1_samples)))
        print("Average weighted macro-average F1: {}\n".format(statistics.mean(self.weighted_macroaverage_f1_samples)))
        print("")

        print("Standard Deviation accuracy: {}\n".format(statistics.stdev(self.accuracy_samples)))
        print("Standard Deviation macro-average F1: {}\n".format(statistics.stdev(self.macroaverage_f1_samples)))
        print("Standard Deviation weighted macro-average F1: {}\n"
              .format(statistics.stdev(self.weighted_macroaverage_f1_samples)))
