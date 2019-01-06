
import csv
import sys
import math
import numpy as np

#Command line inputs 
train_input_path = sys.argv[1]
test_input_path = sys.argv[2]
train_output_path = sys.argv[3]
test_output_path = sys.argv[4]
metrics_out_path = sys.argv[5]
num_epoch = int(sys.argv[6])
hidden_units = int(sys.argv[7])
init_flag = int(sys.argv[8])
learning_rate = float(sys.argv[9])

def ParseInitFlag(flag):
    if flag == 1:
        return "random"
    elif flag == 2:
        return "zero"
    else:
        return "NaN"

print("Sucessfully parsed arguments. Training neural network on", train_input_path, "with", hidden_units, "hidden units \n", ParseInitFlag(init_flag), "parameter initialization, and learning rate of", learning_rate, ".")




#Read in the training data:
with open(train_input_path, 'r') as f:
    reader = csv.reader(f, delimiter = ',')
    row_arrays = [np.asarray(row, dtype = float) for row in reader]
    training_dat = np.stack(row_arrays)

#Read in the test data:
with open(test_input_path, 'r') as f:
    reader = csv.reader(f, delimiter = ',')
    row_arrays = [np.asarray(row, dtype = float) for row in reader]
    test_dat = np.stack(row_arrays)


#Implements the sigmoid activation function of a vector
def SigmoidVec(input):
    e = np.exp(np.multiply(-1, input))
    denom = np.add(1, e)
    return np.divide(1, denom)


#do the sigmoid forward and add a bias term of 1.0
def SigmoidForward(a):
    return np.append(1.0, SigmoidVec(a))

#Backward pass of sigmoid function
def SigmoidBackward(a, b, g_b):
    one_minb = np.subtract(np.ones((len(b))), b)
    prod = np.multiply(b, one_minb).reshape(1,len(b))
    return np.multiply(g_b.reshape(1, len(b)), prod)

#Forward Softmax
def SoftmaxForward(b):
    e = np.exp(b - np.max(b))
    return e / e.sum()

#Backward pass of Softmax
def SoftmaxBackward(a, yh, g_yh):
    dydb = np.subtract(np.diag(yh), np.dot(yh.reshape(10,1), yh.reshape(1,10)))
    return np.dot(g_yh.reshape(1,10), dydb)

#Forward Pass of Linear Layer
def LinearForward(a, w):
    return np.dot(w, a)

#Backward Pass of Linear Layer
def LinearBackward(a, w, b, g_b):
    g_w = np.dot(np.transpose(g_b), a.reshape(1, len(a)))
    g_a = np.dot(np.transpose(w), np.transpose(g_b))
    return g_w, g_a

#Forward Pass of Cross-Entropy
def CrossEntropyForward(y, yh):
    yt = -np.transpose(y)
    return np.dot(yt, np.log(yh))

#Backward Pass of Cross Entropy
def CrossEntropyBackward(y, yh, b, gb):
    return -(np.dot(gb, np.divide(y, yh)))


#Forward calculation step of the neural network. We use two sets of parameters al and B, with z as the sigmoid activation layer, and a final y_pred
#determined by a softmax layer. The algorithm
#Returns a hashmap containing the value of each state of the neural network computation.
def NNForward(x, y, al, B):
    first_linear = LinearForward(x, al)
    z_hidden = SigmoidForward(first_linear)
    hidden_linear = LinearForward(z_hidden, B)
    y_pred = SoftmaxForward(b)
    J = CrossEntropyForward(y, y_pred)
    o = {"x": x, "a" : first_linear, "z" : z_hidden, "b": hidden_linear, "yh": y_pred, "J": J}
    return o


#Backpropagation step of the neural network. gJ is the derivative of the objective function J with respect to itself (dJ/dJ = 1).
def NNBackward(x, y, al, B, o):
    gJ = 1
    yh = o["yh"]
    b = o["b"]
    J = o["J"]
    z = o["z"]
    a = o["a"]

    #First, the backprop for the objective function to get the derivative w.r.t. the predicted y values.
    gyh = CrossEntropyBackward(y, yh, J, gJ)

    #Next, the backprop from the predicted y values to the outputs of the second linear hidden layer (linear combination with B params)
    gb = SoftmaxBackward(b, yh, gyh)

    #From this we can get both the derivatives w.r.t the parameters B and the sigmoid layer z
    gB, gz = LinearBackward(z, B, b, gb)

    #Next, backprop from the sigmoid layer to get the parameters for the output of the first linear layer
    ga = SigmoidBackward(a, z, gz)[0][1:]
    ga = ga.reshape(1, len(ga))

    #Finally, get the derivative w.r.t. the first linear parameters alpha (gal)
    gal, gx = LinearBackward(x, al, a, ga)

    #return the derivative w.r.t. both parameters alpha and beta.
    return gal, gB



#The Finite difference method was implemented for debugging the backpropagation algorithm. It is more or less
#A brute force approach to computing the derivative.
def FiniteDiffA(x, y, al, B):
    ep = 1e-5

    #First alpha
    alphadiff = np.zeros((hidden_units, len(x)))
    agrad = np.zeros((hidden_units, len(x)))

    #Add a small amount ep to a single alpha parameter, then subtract, and compute the difference.
    for i in range(hidden_units):
        for j in range(len(x)):
            alphadiff[i][j] = ep
            aplus = LinearForward(x, np.add(al, alphadiff))
            aminus = LinearForward(x, np.subtract(al, alphadiff))

            v = np.subtract(aplus, aminus)


            agrad[i][j] = np.sum(v)

    return agrad

def FiniteDiffYh(y, yh):
    yhgrad = np.zeros((len(yh)))

    #Add a small amount to a single parameter, then subtract, and compute the difference.
    for i in range(len(yh)):
        yh_diff = np.zeros((len(yh)))
        yh_diff[i] = 1e-5
        yhplus = CrossEntropyForward(y, np.add(yh, yh_diff))
        yhminus = CrossEntropyForward(y, np.subtract(yh, yh_diff))
        yhgrad[i] = np.sum(np.subtract(yhplus, yhminus)) / 2e-5
    return yhgrad

def FiniteDiffb(y, b):
    bgrad = np.zeros(len(b))
    for i in range(len(b)):
        b_diff = np.zeros((len(b)))
        b_diff[i] = 1e-5
        bplus = CrossEntropyForward(y, SoftmaxForward(np.add(b, b_diff)))
        bminus = CrossEntropyForward(y, SoftmaxForward(np.subtract(b, b_diff)))
        bgrad[i] = np.multiply(np.sum(np.subtract(bplus, bminus)), 2e5)
    return bgrad



#Compute the cross entropy from a training dataset, a test dataset, parameters alpha and beta, over a set number of epochs.
def ComputeCrossEntropy(train, test, alpha, beta, epoch):
    #Initialize arrays
    cent = np.array([])
    test_cent = np.array([])

    #Append the cross-entropy of each row in the training dataset to cent
    for row in train:
        y = np.zeros((10))
        y[int(row[0])] = 1.0
        x = np.append(1.0, row[1:])
        o = NNForward(x, y, alpha, beta)
        cent = np.append(cent, o["J"])

    #Append the cross-entropy of each row in the test dataset to test_cent
    for test_row in test:
        test_y = np.zeros((10))
        test_y[int(test_row[0])] = 1.0
        test_x = np.append(1.0, test_row[1:])
        test_o = NNForward(test_x, test_y, alpha, beta)
        test_cent = np.append(test_cent, test_o["J"])

    #Compute the means and output in a nicely formatted string
    current_epoch_string = "epoch=" + str(epoch) + " crossentropy(train): " + str(np.mean(cent)) + "\n"
    current_epoch_string += "epoch=" + str(epoch) + " crossentropy(test): " + str(np.mean(test_cent)) + "\n"

    return current_epoch_string


#Perform 'epochs' number of epochs of stochastic gradient descent (SGD) to compute the optimized alpha and beta parameters for a given
#training dataset. Then, compute the cross-entropy and error on the training and test datasets, and print.
def SGD(train_data, epochs, test_data):
    #Initialize parameters

    if init_flag == 1:
        alpha = np.random.rand(hidden_units, len(train_data[0]))
        alpha = np.multiply(0.2, np.subtract(0.5, alpha))

        beta = np.random.rand(10, (hidden_units + 1))
        beta = np.multiply(0.2, np.subtract(0.5, beta))
    elif init_flag == 2:
        alpha = np.zeros((hidden_units, len(train_data[0])))
        beta = np.zeros((10, hidden_units + 1))
    else:
        print("Error: Incorrect init_flag input")

    #Initialize metrics string
    metrics_string = ""

    for i in range(epochs):

        #First, do the training
        for row in train_data:
            y = np.zeros((10))
            y[int(row[0])] = 1.0
            x = np.append(1.0, row[1:])
            o = NNForward(x, y, alpha, beta)

            ga, gb = NNBackward(x, y, alpha, beta, o)

            alpha = np.subtract(alpha, np.multiply(learning_rate, ga))
            beta = np.subtract(beta, np.multiply(learning_rate, gb))

        metrics_string += ComputeCrossEntropy(train_data, test_data, alpha, beta, (i+1))
        print("Finished Epoch #:", (i + 1), "out of", epochs, "...")

    return alpha, beta, metrics_string



trained_alpha, trained_beta, metrics = SGD(training_dat, num_epoch, test_dat)

print("Network training complete. Evaluating model on training data and on ", test_input_path)


#PredictClass takes a given dataset and parameters alpha and beta, and returns a series of predictions,
#defined as the most likely class for each entry in the dataset, for all entries in the dataset.
def PredictClass(data, alpha, beta):
    error = [0.0, 0.0]
    output_string = ""

    for row in data:
        y = int(row[0])
        x = np.append(1.0, row[1:])

        o = NNForward(x, y, alpha, beta)

        prediction = np.argmax(o["yh"])
        output_string += str(prediction) + "\n"

        if prediction == y:
            error[0] += 1.0
        else:
            error[1] += 1.0
    error_tot = error[0] + error[1]
    err_out = str(error[1]/error_tot)

    return output_string, err_out


train_labels, train_err = PredictClass(training_dat, trained_alpha, trained_beta)
test_labels, test_err = PredictClass(test_dat, trained_alpha, trained_beta)


##Pretty-printing the outputs
print("Done! \n Printing training data labels to", train_output_path, "\n Printing test data labels to", test_output_path, "\n Printing metrics to", metrics_out_path)

with open(train_output_path, 'w') as f:
    f.write(train_labels)

with open(test_output_path, 'w') as f:
    f.write(test_labels)

#Now format the error string

error_string = "error(train): " + str(train_err) + "\n"
error_string += "error(test): " + str(test_err) + "\n"

metrics += error_string

with open(metrics_out_path, 'w') as file:
    file.write(metrics)
