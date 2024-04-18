import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv(r"/home/ibab/datasets/simulated_data_multiple_linear_regression_for_ML.csv")       # Load file
    df.insert(0, "x_0", 1)      # Insert 1s in the first column, to create the bias term
    x = df.iloc[:, 0:-2]
    y = df.iloc[:, -2]
    return df, x, y

def cost_function(x_train, y_train, theta):
    total_cost = 0
    for i in range(len(x_train)):
        predicted_value = sum(theta[j] * x_train.iloc[i, j] for j in range(len(theta)))     # Predicted value
        squared_error = (predicted_value - y_train.iloc[i]) ** 2        # Sum of squared errors: Difference between train value and predicted value squared
        total_cost = total_cost + squared_error
    total_cost = 0.5 * total_cost
    return total_cost

def batch_gradient_descent(x_train, y_train):
    theta = [0] * x_train.shape[1]      # Initialising theta values with 0s
    cost_function_prev = cost_function(x_train, y_train, theta)
    learning_rate = 0.0000001       # Learning rate: Size of steps taken to arrive at convergence
    delta = 100000000000        # Delta: When difference between 2 consecutive cost functions is minimised, convergence is reached

    while delta > 1:
        p_d = [0] * len(theta)
        for i in range(len(theta)):     # Theta update: This loop calculates the partial derivative, and updates theta
            p_d_sum = 0
            for j in range(len(y_train)):
                h_theta_x = sum(theta[k] * x_train.iloc[j, k] for k in range(len(theta)))
                p_d_sum = p_d_sum + (h_theta_x - y_train.iloc[j]) * x_train.iloc[j, i]
            p_d[i] = p_d_sum / len(y_train)
            theta[i] = theta[i] - learning_rate * p_d[i]
        cost_function_after = cost_function(x_train, y_train, theta)
        delta = abs(cost_function_prev - cost_function_after)       # Updates delta
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
    theta = batch_gradient_descent(x_train, y_train)        # Obtain thetas
    y_pred = np.dot(x_test, theta)      # Predicted y values
    r2 = r2_score(y_test, y_pred)       # CAlculate r squared score
    print("R^2 score of the batch gradient descent model:", r2)

if __name__ == "__main__":
    main()

