import random;
import math;
import numpy as np;
from matplotlib import pyplot as plt
import pandas as pd 
import seaborn as sns 
 
    
# now we want to flip the label, and we store the booleans for whether we flip or not
# http://www.hongliangjie.com/wp-content/uploads/2011/10/logistic.pdf
# https://stats.stackexchange.com/questions/250937/which-loss-function-is-correct-for-logistic-regressionhttps://stats.stackexchange.com/questions/250937/which-loss-function-is-correct-for-logistic-regression

# FIXED TEST SAMPLE
test_samples = []
means = [[1,0], [-1,0],[1/24, 5/24], [-1/24, 5/24], [1/24, -1/24], [-1/24, 1/24]]
for i in range(500) :
    mean = random.choices(means, weights=(1, 1, 1, 1, 2, 2), k=1)
    mean = mean[0]
    test_samples.append(np.random.multivariate_normal(mean, np.eye(2)))

test_labels = []          
for i in range(500) :
    sample = test_samples[i]
    if ( sample[0] > 0):
        test_labels.append(1)
    else :
        test_labels.append(-1)

# graph plotted for logistic not corrected so need to test the functions and verify correct
                 
def logistic_loss(z):
    return math.log(1+ math.exp(-1*z))

def inverse_sigmoid(z):
    return math.log(z/(1-z))

def huberized_loss(z):
    # technically should be one but we are using 0.5 instead for now
    tau = 0.1
    if (z <= -inverse_sigmoid(tau)):
        return -tau*z - math.log(1-tau)-tau*inverse_sigmoid(tau)
    else :
        return math.log(1+ math.exp(-1*z))
        
def partially_huberized_loss(z):
    # technically should be one but we are using 0.5 instead for now
    tau = 1.1
    if (z <= inverse_sigmoid(1/tau)):
        return -tau/(1+math.exp(-z)) + math.log(tau) + 1
    else :
        return math.log(1+ math.exp(-1*z))
    
# what is above is presumably correct since the graph previously drawn was correct 
        
# because now we are in labels {+1, -1}, hence we take the product
# now we will be working with gradient descent and minimizing a different type
# of loss because we have the labels we have explained above 
def calculate_logistic_gradient(labels, samples, w):
    # we iterate over all the samples in the samples
    # note w must be give as a column vector and should be numpy array
    gradient = np.zeros((2,))
    # because you need to apply this on all of the 1000 random samples 
    for i in range(1000):
        # taking transpose of samples[i] because they are given as row vectors
        # print(labels[i]*w.T.dot(samples[i])) # prints correctly 0.0
        # print(1 + math.exp(-labels[i]*w.T.dot(samples[i]))) # prints correctly two
        gradient += math.exp(-labels[i]*w.T.dot(samples[i]))/(1 + math.exp(-labels[i]*w.T.dot(samples[i])))*(-labels[i]*samples[i])
    return gradient
    
def calculate_huber_gradient(labels, samples, w):
    # we iterate over all the samples in the samples
    # note w must be give as a column vector and should be numpy array
    tau = 0.5
    gradient = np.zeros((2,))
    # because you need to apply this on all of the 1000 random samples 
    for i in range(1000):
        # taking transpose of samples[i] because they are given as row vectors
        # print(labels[i]*w.T.dot(samples[i])) # prints correctly 0.0
        # print(1 + math.exp(-labels[i]*w.T.dot(samples[i]))) # prints correctly two
        if(labels[i]*w.T.dot(samples[i]) <= -1*inverse_sigmoid(tau)):
            gradient += math.exp(-labels[i]*w.T.dot(samples[i]))/(1 + math.exp(-labels[i]*w.T.dot(samples[i])))*(-labels[i]*samples[i])
        else:
            gradient += -tau*labels[i]*samples[i]
    return gradient
    
def calculate_partial_huber_gradient(labels, samples, w):
    # we iterate over all the samples in the samples
    # note w must be give as a column vector and should be numpy array
    tau = 1.5
    gradient = np.zeros((2,))
    # because you need to apply this on all of the 1000 random samples 
    for i in range(1000):
        # taking transpose of samples[i] because they are given as row vectors
        # print(labels[i]*w.T.dot(samples[i])) # prints correctly 0.0
        # print(1 + math.exp(-labels[i]*w.T.dot(samples[i]))) # prints correctly two
        if(labels[i]*w.T.dot(samples[i]) <= inverse_sigmoid(1/tau)):
            gradient += math.exp(-labels[i]*w.T.dot(samples[i]))/(1 + math.exp(-labels[i]*w.T.dot(samples[i])))*(-labels[i]*samples[i])
        else:
            gradient += tau*labels[i]*samples[i]*math.exp(-labels[i]*w.T.dot(samples[i]))
    return gradient

def learning_by_gradient_descent_logistic(labels, samples, w, gamma):
    loss = 0
    for i in range(1000):
        loss += logistic_loss(labels[i]*w.T.dot(samples[i]))    
        
    grad = calculate_logistic_gradient(labels, samples, w)
    w -= gamma * grad
    return loss, w

def learning_by_gradient_descent_huber(labels, samples, w, gamma):
    loss = 0
    for i in range(1000):
        loss += huberized_loss(labels[i]*w.T.dot(samples[i]))    
        
    grad = calculate_huber_gradient(labels, samples, w)
    w -= gamma * grad
    return loss, w

def learning_by_gradient_descent_partial_huber(labels, samples, w, gamma):
    loss = 0
    for i in range(1000):
        loss += partially_huberized_loss(labels[i]*w.T.dot(samples[i]))    
        
    grad = calculate_partial_huber_gradient(labels, samples, w)
    w -= gamma * grad
    return loss, w

def generate_training_samples():
    
    random_samples = []
    for i in range(1000) :
        mean = random.choices(means, weights=(1, 1, 1, 1, 2, 2), k=1)
        mean = mean[0]
        random_samples.append(np.random.multivariate_normal(mean, np.eye(2)))
    
    labels = []
    flip = [True, False];

    for i in range(1000) :
        do_we_flip = random.choices(flip, weights=(45, 55), k=1)
        sample = random_samples[i]
        
        if(do_we_flip == True) :
            if ( sample[0] > 0):
                labels.append(-1)
            else :
                labels.append(1)
        else :
            if ( sample[0] > 0):
                labels.append(1)
            else :
                labels.append(-1)
                
    return random_samples, labels

def optimal_weight_logistic(samples, labels, max_iter, threshold, gamma):
    
    losses_logistic = []
    # the initial optimal weight vector is set to be a vector of length two of zeroes
    w_logistic = np.zeros((2,))
    for iter in range(max_iter):
        # get loss and update w.
        loss, w_logistic = learning_by_gradient_descent_logistic(labels, samples, w_logistic, gamma)
        # converge criterion
        losses_logistic.append(loss)
        # do we actually want this condition below to test convergence ? but either way it runs max
        # max_iter times
        if len(losses_logistic) > 1 and np.abs(losses_logistic[-1] - losses_logistic[-2]) < threshold:
            break
    
    return w_logistic;

def optimal_weight_huber(samples, labels, max_iter, threshold, gamma):
    
    losses_huber = []
    # the initial optimal weight vector is set to be a vector of length two of zeroes
    w_huber = np.zeros((2,))
    for iter in range(max_iter):
        # get loss and update w.
        loss, w_huber = learning_by_gradient_descent_huber(labels, samples, w_huber, gamma)
        # converge criterion
        losses_huber.append(loss)
        # do we actually want this condition below to test convergence ? but either way it runs max
        # max_iter times
        if len(losses_huber) > 1 and np.abs(losses_huber[-1] - losses_huber[-2]) < threshold:
            break
    
    return w_huber;
    
def optimal_weight_partial_huber(samples, labels, max_iter, threshold, gamma):
    
    losses_partial_huber = []
    # the initial optimal weight vector is set to be a vector of length two of zeroes
    w_partial_huber = np.zeros((2,))
    for iter in range(max_iter):
        # get loss and update w.
        loss, w_partial_huber = learning_by_gradient_descent_partial_huber(labels, samples, w_partial_huber, gamma)
        # converge criterion
        losses_partial_huber.append(loss)
        # do we actually want this condition below to test convergence ? but either way it runs max
        # max_iter times
        if len(losses_partial_huber) > 1 and np.abs(losses_partial_huber[-1] - losses_partial_huber[-2]) < threshold:
            break
    
    return w_partial_huber;

def test_accuracy_logistic(samples, labels, max_iter, threshold, gamma):
    w_logistic = optimal_weight_logistic(samples, labels, max_iter, threshold, gamma)
    predicted_labels = []
    # for the test cases the label and then check against the true label 
    for i in range(500) :
        sample = test_samples[i];
        product = w_logistic.T.dot(sample);
        if product >= 0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(-1)
    
    number_correctly_predicted = 0;
    for i in range(500) :
        if test_labels[i] == predicted_labels[i] :
            number_correctly_predicted += 1;
            
    return number_correctly_predicted/500

def test_accuracy_huber(samples, labels, max_iter, threshold, gamma):
    w_huber = optimal_weight_huber(samples, labels, max_iter, threshold, gamma)
    predicted_labels = []
    # for the test cases the label and then check against the true label 
    for i in range(500) :
        sample = test_samples[i];
        product = w_huber.T.dot(sample);
        if product >= 0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(-1)
    
    number_correctly_predicted = 0;
    for i in range(500) :
        if test_labels[i] == predicted_labels[i] :
            number_correctly_predicted += 1;
            
    return number_correctly_predicted/500

def test_accuracy_partial_huber(samples, labels, max_iter, threshold, gamma):
    w_partial_huber = optimal_weight_partial_huber(samples, labels, max_iter, threshold, gamma)
    predicted_labels = []
    # for the test cases the label and then check against the true label 
    for i in range(500) :
        sample = test_samples[i];
        product = w_partial_huber.T.dot(sample);
        if product >= 0:
            predicted_labels.append(1)
        else:
            predicted_labels.append(-1)
    
    number_correctly_predicted = 0;
    for i in range(500) :
        if test_labels[i] == predicted_labels[i] :
            number_correctly_predicted += 1;
            
    return number_correctly_predicted/500

def plot_graph() :
    # init parameters
    max_iter = 100 # in order to reduce the compute time
    threshold = 1e-2 # so it will be 0.1
    gamma = 0.1 # for a quicker convergence 
    
    # we perform 500 trials of the same experiment
    # and the accuracies need to be stored in this matrix 
    accuracies_logistic = []
    accuracies_huber = []
    accuracies_partial_huber = []
    
    # should be 500 but we will try with smaller number of samples to see if it works with this many
    for k in range(10):
        # generate the random samples and labels
        samples, labels = generate_training_samples();
        accuracy_logistic = test_accuracy_logistic(samples, labels, max_iter, threshold, gamma)
        accuracies_logistic.append(accuracy_logistic)
        accuracy_huber = test_accuracy_huber(samples, labels, max_iter, threshold, gamma)
        accuracies_huber.append(accuracy_huber)
        accuracy_partial_huber = test_accuracy_partial_huber(samples, labels, max_iter, threshold, gamma)
        accuracies_partial_huber.append(accuracy_partial_huber)
        
    # now we have 500 test accuracies for the logistic regression and we want to plot a box plot
    df = pd.DataFrame(list(zip(accuracies_logistic, accuracies_huber,accuracies_partial_huber)))
    # print(df) # actually prints and the columns are different but all values are really close to 1 
    print(df)
    boxplot = df.boxplot(grid=False) 
    boxplot.plot()
    plt.show()

plot_graph() 
