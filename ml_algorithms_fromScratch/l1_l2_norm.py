# Code from scratch
# Use L1 and L2 norm to prevent Overfitting in Batch Gradient Descent

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv(r"/home/ibab/datasets/simulated_data_multiple_linear_regression_for_ML.csv")
    df.insert(0, "x_0", 1)
    x = df.iloc[:, 0:-2]
    y = df.iloc[:, -2]
    return df, x, y

def cost_function(x_train, y_train, theta, lmbda):
    total_cost = 0
    for i in range(len(x_train)):
        predicted_value = sum(theta[j] * x_train.iloc[i, j] for j in range(len(theta)))     # Predicted value
        squared_error = (predicted_value - y_train.iloc[i]) ** 2        # Sum of squared errors: Difference between train value and predicted value squared
        total_cost = total_cost + squared_error
    total_cost = 0.5 * total_cost
    return total_cost

def l1_norm(x_train, y_train):
    theta = [0, 0, 0, 0, 0, 0]
    lmbda = 0.1     # regularization parameter

    cost_function_prev = cost_function(x_train, y_train, theta, lmbda)

    learning_rate = 0.0000001       # Learning rate: Size of steps taken to arrive at convergence
    delta = 100000000000        # Delta: When difference between 2 consecutive cost functions is minimised, convergence is reached
    p_d = [0, 0, 0, 0, 0, 0]

    while delta >= 1:
        for j in range(len(theta)):     # Theta updates
            p_d[j] = sum([(theta[0] * x_train.iloc[i, 0] +
                           theta[1] * x_train.iloc[i, 1] +
                           theta[2] * x_train.iloc[i, 2] +
                           theta[3] * x_train.iloc[i, 3] +
                           theta[4] * x_train.iloc[i, 4] +
                           theta[5] * x_train.iloc[i, 5] -
                           y_train.iloc[i]) * x_train.iloc[i, j] + lmbda
                          for i in range(len(x_train))])
            theta[j] = theta[j] - (learning_rate * p_d[j])
        cost_function_after = cost_function(x_train, y_train, theta, lmbda)
        delta = abs(cost_function_prev - cost_function_after)
        cost_function_prev = cost_function_after
    return theta

def l2_norm(x_train, y_train):
    theta = [0, 0, 0, 0, 0, 0]
    lmbda = 0.1     # Regularization parameter

    cost_function_prev = cost_function(x_train, y_train, theta, lmbda)

    learning_rate = 0.0000001       # Learning rate: Size of steps taken to arrive at convergence
    delta = 100000000000        # Delta: When difference between 2 consecutive cost functions is minimised, convergence is reached
    p_d = [0, 0, 0, 0, 0, 0]

    while delta >= 1:
        for j in range(len(theta)):     # Updates theta
            p_d[j] = sum([(theta[0] * x_train.iloc[i, 0] +
                           theta[1] * x_train.iloc[i, 1] +
                           theta[2] * x_train.iloc[i, 2] +
                           theta[3] * x_train.iloc[i, 3] +
                           theta[4] * x_train.iloc[i, 4] +
                           theta[5] * x_train.iloc[i, 5] -
                           y_train.iloc[i]) * x_train.iloc[i, j]
                          for i in range(len(x_train))])
            theta[j] = theta[j]*(1-2*learning_rate*lmbda) - (learning_rate * p_d[j])
        cost_function_after = cost_function(x_train, y_train, theta, lmbda)
        delta = abs(cost_function_prev - cost_function_after)
        cost_function_prev = cost_function_after
    return theta

def r2_score(y_ground_truth, y_pred):
    ss_residuals = sum((y_ground_truth - y_pred) ** 2)      # Sum of squares of residuals
    ss_total = sum((y_ground_truth - np.mean(y_ground_truth)) ** 2)     # Total sum of squares
    r_squared = 1 - (ss_residuals / ss_total)       # We use R squared as the performance metric
    return r_squared

def main():
    df, x, y = load_data()      # Load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)       # Split data into test and train sets
    theta_l1 = l1_norm(x_train, y_train)        # Obtain thetas from l1 norm
    theta_l2 = l2_norm(x_train, y_train)        # Obtain thetas from l2 norm
    y_pred_l1 = np.dot(x_test, theta_l1)      # Predicted y values for l1 norm
    y_pred_l2 = np.dot(x_test, theta_l2)  # Predicted y values for l2 norm
    r2_l1 = r2_score(y_test, y_pred_l1)       # Calculate r squared score for l1 norm
    r2_l2 = r2_score(y_test, y_pred_l2)       # Calculate r squared score for l2 norm
    print("R^2 score of the batch gradient descent model using l1 norm:", r2_l1)
    print("R^2 score of the batch gradient descent model using l2 norm:", r2_l2)

if __name__ == "__main__":
    main()
