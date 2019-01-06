Neural Network Optical Character Recognizer

This project takes inputs as .csv files, representing hand-drawn characters
from the Optical Character Recognition (OCR) dataset, with the class described as
an integer representing a given a-z letter, and binary feature vectors representing
black-and-white pixels.

The model is a single-hidden-layer neural net with a sigmoid activation function for the
hidden layer, and a softmax on the output layer. The objective function used
is average cross-entropy over the training dataset, and is minimized by stochastic 
gradient descent (SGD). 

The program neuralnet.py takes 9 command line arguments and is run in the following way:

$python neuralnet.py <1> <2> <3> <4> <5> <6> <7> <8> <9>

Where:

<1> -- path to the training input .csv file 

<2> -- path to the test input .csv file 

<3> -- path to the output plaintext file to which the predictions on the 
training dataset will be written

<4> -- path to the output plaintext file to which the predictions on the 
test dataset will be written 

<5> -- path of the output plaintext file to which the metrics, such as 
cross entropy and error, will be written 

<6> -- integer specifying the number of times the backpropagation step loops
through the training data.

<7> -- positive integer specifying the number of hidden units

<8> -- init_flag -- integer taking the value 1 or 2 that specifies initialization method
for the network parameters. If init_flag == 1, weights are initialized randomly. If init_flag == 2,
weights are initialized to zero.

<9> -- float value specifying the learning rate for stochastic gradient descent.