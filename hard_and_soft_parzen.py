# Coded by Philippe Schoeb
# October 12th 2023

import numpy as np

winequality = np.genfromtxt("winequality.txt")


def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.labels = train_labels
        self.inputs = train_inputs

    def compute_predictions(self, test_data):
        number_random = 0
        labels_predicted = []
        for i in range(len(test_data)):
            candidate = [0] * len(self.label_list)  # Where we keep our scores
            for j in range(len(self.labels)):
                dist = np.linalg.norm(test_data[i] - self.inputs[j])
                if dist <= self.h:
                    index = -1  # We search for the good index
                    label = self.labels[j]
                    for k in range(len(self.label_list)):
                        if self.label_list[k] == label:
                            index = k
                            break
                    candidate[index] += 1  # Add to the scores

            if np.sum(candidate) == 0:  # Random label
                labels_predicted.append(int(draw_rand_label(test_data[i], self.label_list)))
                number_random += 1
            else:
                labels_predicted.append(int(self.label_list[np.argmax(candidate)]))
        print("Number of randomized choices : " + str(number_random))
        labels_predicted = np.array(labels_predicted)
        return labels_predicted

class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.inputs = train_inputs
        self.labels = train_labels

    def compute_predictions(self, test_data):
        normalisation = float(1)/((2*np.pi)**(11/2)*self.sigma**11)
        labels_predicted = []
        for i in range(len(test_data)):
            candidate = [0] * len(self.label_list)
            for j in range(len(self.labels)):
                dist = np.linalg.norm(test_data[i] - self.inputs[j])
                kernel = normalisation * np.exp(-(dist**2)/(2*(self.sigma**2)))
                if kernel < 0:
                    print("Kernel < 0, problem. Kernel = " + str(kernel))
                index = -1
                label = self.labels[j]
                for k in range(len(self.label_list)):
                    if self.label_list[k] == label:
                        index = k
                        break
                candidate[index] += kernel
            labels_predicted.append(int(self.label_list[np.argmax(candidate)]))
        labels_predicted = np.array(labels_predicted)
        return labels_predicted

def split_dataset(wineQuality):
    train = []
    valid = []
    test = []
    for i in range(len(wineQuality)):
        if i % 5 == 0 or i % 5 == 1 or i % 5 == 2:
            train.append(wineQuality[i])
        elif i % 5 == 3:
            valid.append(wineQuality[i])
        else:
            test.append(wineQuality[i])
    train = np.array(train)
    valid = np.array(valid)
    test = np.array(test)
    return train, valid, test

class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        HardParzen.__init__(HardParzen, h)
        HardParzen.train(HardParzen, self.x_train, self.y_train)
        predictions = HardParzen.compute_predictions(HardParzen, self.x_val)
        number_errors = 0
        for i in range(len(predictions)):
            if int(predictions[i]) != int(self.y_val[i]):
                number_errors += 1
        error = float(number_errors)/len(predictions)
        return error

    def soft_parzen(self, sigma):
        SoftRBFParzen.__init__(SoftRBFParzen, sigma)
        SoftRBFParzen.train(SoftRBFParzen, self.x_train, self.y_train)
        predictions = SoftRBFParzen.compute_predictions(SoftRBFParzen, self.x_val)
        number_errors = 0
        for i in range(len(predictions)):
            if int(predictions[i]) != int(self.y_val[i]):
                number_errors += 1
        error = float(number_errors) / len(predictions)
        return error

def get_test_errors(wineQuality):
    train, valid, test = split_dataset(wineQuality)
    ErrorRate.__init__(ErrorRate, train[:, :11], train[:, 11], valid[:, :11], valid[:, 11])
    valeurs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 3, 10, 20]
    error1 = [0] * len(valeurs)
    error2 = [0] * len(valeurs)
    for i in range(len(valeurs)):
        h = valeurs[i]
        sigma = valeurs[i]
        error1[i] = ErrorRate.hard_parzen(ErrorRate, h)
        error2[i] = ErrorRate.soft_parzen(ErrorRate, sigma)
        print("Error rate with h = " + str(h) + ", Hard : " + str(error1[i]))
        print("Error rate with sigma = " + str(sigma) + ", Soft : " + str(error2[i]))
    index1 = np.argmin(error1)
    index2 = np.argmin(error2)
    best_h = valeurs[index1]
    best_sigma = valeurs[index2]
    ErrorRate.__init__(ErrorRate, train[:, :11], train[:, 11], test[:, :11], test[:, 11])
    error_hard = ErrorRate.hard_parzen(ErrorRate, best_h)
    error_soft = ErrorRate.soft_parzen(ErrorRate, best_sigma)
    print("For hard with test set, h = " + str(best_h) + ", error_rate = " + str(error_hard))
    print("For soft with test set, sigma = " + str(best_sigma) + ", error_rate = " + str(error_soft))
    errors = np.array([error_hard, error_soft])
    return errors

get_test_errors(winequality)
