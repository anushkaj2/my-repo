# Code from scratch
# Logistic Regression with K FOld Cross Validation and Confusion Matrix to compute accuracy

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv(r"/home/ibab/datasets/binary_logistic_regression_data_simulated_for_ML.csv")       # Load data
    df.insert(0, "x_0", 1)      # Add a column of 1s as the bias term
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    return df, x, y

def sigmoid(z):
    return 1/(1+np.exp(-z))     # Logistic function

def cost_function(x_train, y_train, theta):
    f = sum(
        [((y_train.iloc[i] * np.log(sigmoid(sum(theta[j] * x_train.iloc[i, j] for j in range(len(theta))))) +
           (1 - y_train.iloc[i]) * np.log(1 - sigmoid(sum(theta[j] * x_train.iloc[i, j] for j in range(len(theta)))))))
         for i in range(len(x_train))])     # Cost function
    return f

def logistic_regression_theta(x_train, y_train):
    theta = np.zeros(x_train.shape[1])      # Initialising theta values with 0s
    log_l_theta_prev = cost_function(x_train, y_train, theta)
    learning_rate = 0.000001       # Learning rate: Size of steps taken to arrive at convergence
    delta = 1000000       # Delta: When difference between 2 consecutive cost functions is minimised, convergence is reached
    p_d = np.zeros(theta.shape)

    while delta > 0.1:
        for j in range(len(theta)):     # Theta update: This loop calculates the partial derivative, and updates theta
            p_d[j] = sum((y_train.iloc[i] - sigmoid(sum(theta[k] * x_train.iloc[i, k] for k in range(len(theta))))) *
                         x_train.iloc[i, j] for i in range(len(x_train)))
            theta[j] = theta[j] + (learning_rate * p_d[j])
        log_l_theta_after = cost_function(x_train, y_train, theta)
        delta = abs(log_l_theta_prev - log_l_theta_after)       # Updates delta
        log_l_theta_prev = log_l_theta_after
    return theta

def logistic_regression_predict(theta, x_testing):
    y_predicted = [ (sigmoid(sum(theta[k] * x_testing.iloc[i, k] for k in range(len(theta))))) for i in range(len(x_testing)) ]     # Computes the predicted y probabilities
    return y_predicted

def confusion_matrix(y, y_hat, threshold):
    y_predicted = []    # Classifying predicted probabilities to 0 or 1 based on threshold
    for val in y_hat:
        if val < threshold:
            y_predicted.append(0)
        else:
            y_predicted.append(1)
    tp = 0     # Values for confusion matrix
    fp = 0
    fn = 0
    tn = 0
    y = list(y)
    for i in range(len(y_hat)):     # Calculates the confusion matrix
        if y[i] == y_predicted[i]:
            if y[i] == 1:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if y[i] == 1:
                fn = fn + 1
            else:
                fp = fp + 1
    matrix = np.array([[tp, fp], [fn, tn]])
    return matrix

def metrics(confusion_m):
    accuracy = (confusion_m[0,0] + confusion_m[1,1])/(confusion_m[0,0] + confusion_m[0,1] + confusion_m[1,0] + confusion_m[1,1])        # Calculates accuracy sore
    return accuracy

def logistic_regression_accuracy(y, y_predicted):
    threshold = 0.5     # Threshold to calculate confusion matrix
    confusion_m = confusion_matrix(y, y_predicted, threshold)  # Confusion matrix
    accuracy = metrics(confusion_m)
    print(accuracy)
    return accuracy

def k_fold_cross_validation(k, df):
    accuracy_list = []
    fold_size = len(df) // k        # Determines fold sizes based on k

    for i in range(1, k + 1):
        start = (i - 1) * fold_size         # Start index 
        end = i * fold_size         # End index 
        x_train = pd.concat([df.iloc[end:, 0:-1], df.iloc[:start, 0:-1]])
        y_train = pd.concat([df.iloc[end:, -1], df.iloc[:start, -1]])
        x_test = df.iloc[start:end, 0:-1]
        y_test = df.iloc[start:end, -1]
        theta = logistic_regression_theta(x_train, y_train)     # Thetas are obtained from the logistic regression function
        y_predicted = logistic_regression_predict(theta, x_test)        # Predicted probabilities of y are obtained
        accuracy = logistic_regression_accuracy(y_test, y_predicted)        # Predicted probabilities are passed into this function to calculate accuracy
        accuracy_list.append(accuracy)      # All the k accuracies are appended to a list
    return accuracy_list

def main():
    df, x, y = load_data()      # Load data
    k = 10      # k folds
    cv = k_fold_cross_validation(k, df)     # Performs k fold cross validation
    cv_mean = np.mean(cv)       # Mean accuracy is calculated
    cv_stddev = np.std(cv)      # Standard deviation of the k accuracies is calculated
    print(f"Mean accuracy of the logistic regression model with K folds is {cv_mean} and the standard deviation is {cv_stddev}")

if __name__ == "__main__":
    main()
