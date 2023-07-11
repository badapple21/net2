from mnist import MNIST
import random 
import math
import time
import os 

mndata = MNIST('samples')

images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# import matplotlib.pyplot as plt
# import numpy as np
# index = 0
# image = np.array(images[index]).reshape(28, 28)
# plt.imshow(image)
# plt.show()

def sigmoid(x):
   return  1/(1+math.exp(x * -1))

def fake_desigmoid(x):
   return (x * (1-x))

class neuron:
    def __init__(self, activation, layer, row, weights, bias):
        self.activation = activation
        self.layer = layer
        self.row = row
        self.weights = weights
        self.bias = bias

class input_neuron(neuron):
    def __init__(self, activation, row):
        super().__init__(activation, 0, row, [], 0)

class output_neuron(neuron):
    def __init__(self, activation, row, weights, bias):
        super().__init__(0, 3, row, weights, bias)

class hidden_neuron(neuron):
    def __init__(self, activation, layer, row, weights, bias):
        super().__init__(activation, layer, row, weights, bias)

def create_net(input_neurons, hidden_layers, output_neurons):
    net = []

    #inits the input layer neurons
    net.append([])
    for i in range(input_neurons):
        net[0].append(input_neuron(0, i))
    #inits the hidden layers of the net and adds them to the net list
    for i in range(1, len(hidden_layers)+1):
        net.append([])
        for j in range(hidden_layers[i-1]):
            net[i].append(hidden_neuron(0, i, j, [random.randint(40, 60)/100 for i in range(len(net[i-1]))], random.randint(-100, 100)/10))
    #inits the output layer neurons
    net.append([])
    for i in range(output_neurons):
        net[-1].append(output_neuron(0, i, [random.randint(40, 60)/100 for i in range(len(net[-2]))], random.randint(-100, 100)/100))

    return net

#loads the selected image into the net 
def load_image(image, net):
    for i in range(len(net[0])):
        net[0][i].activation = image[i]

    return net
    

def run_net(image, net):
    net = load_image(image, net)

    # activates the hidden layers
    for i in range(1, len(net)):
        net = activate_layer(net, i)
   
    
    # grabs the activation of each output neuron and adds it to the output list
    output = []
    for neuron in net[-1]:
        output.append(neuron.activation)

    return output, net

def activate_layer(net, layer):
    #goes through each neuron in the current layer
    for i in range(len(net[layer])):
        #goes through each neuron in the previous layer 
        for j in range(len(net[layer-1])):
            #multiplies the activation of the neuron in the previous layer by the weight of the connection and adds it to the current neurons activation   
            net[layer][i].activation += (net[layer-1][j].activation * net[layer][i].weights[j])
        # takes the activation of the current neuron and adds the bias 
        net[layer][i].activation += net[layer][i].bias
        
        #takes the activation of the current neuron and puts it through the sigmoid "squishifitcaion" function to get a number between 0 and 1 
        net[layer][i].activation = sigmoid(net[layer][i].activation)

    return net

def get_correct_result(image_label):
    output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    output[image_label] = 1.0

    return output

def get_max(data):
    max = 0
    max_index = -1
    for i in range(len(data)):
        if(data[i]>max):
            max_index = i
            max = data[i]        

    return max_index    

#calculates the error in each neuron of the output layer and returns as a list of errors 
def get_output_layer_error(output, correct_output):
    error = []

    #loops through the output and compares the given result to the correct result and adds that error to the error array 
    for i in range(len(output)):
        error.append(correct_output[i]-output[i])

    return error


#gets the error of all the nodes in a given layer assuming the layer after has already had the error calculated and returns as a list of errors 
def get_hidden_layer_error(hidden_layer_num, error_next_layer, net):
    errors = []
    
    #sets up int i that loops through the number of neurons in the chosen layer
    for i in range(len(net[hidden_layer_num])):

        total_error=0                                                                 
        max_weight = 0

        #loops through the neurons in the next layer
        for j in range(len(error_next_layer)):
            max_weight = 0

            #gets the sum of all the weights for the neuron in the next layer 
            for weight in net[hidden_layer_num+1][j].weights:
                max_weight+=weight
            
            #calculates the given neurons responsibility in the next layers error and adds that to the total error of the neuron 
            total_error+=(net[hidden_layer_num+1][j].weights[i]/max_weight)*error_next_layer[j]

        errors.append(total_error)
    
    return errors

#function tha that returns the bias gradients for an entire layer given the learning rate the calculated error and the outputs of the net 
def get_layer_bias_gradients(lr, errors, outputs):
    #sets up the list that the gradients will be added to 
    gradients = []

    #loops through all the neurons and calculates that amount they should change by
    for i in range(len(errors)):
        gradients.append(lr*errors[i]*fake_desigmoid(outputs[i]))

    #returns the list 
    return gradients



#returns the weight deltas for the entire layer in a list given the learning rate errors outputs of the layer and the inputs of the layer 
def get_layer_weights_deltas(lr, errors, outputs, inputs):
    # sets up the array that the deltas will be added to 
    layer_deltas = []

    #loops through all the neurons in the layer 
    for i in range(len(errors)):
        #sets up the array that all the deltas for neuron will be added to
        neuron_deltas = []

        # loops through all the weights of the neuron 
        for j in range(len(inputs)):
            # calculates the delta of each weight and adds them to the list for that specific neuron 
            neuron_deltas.append(lr*errors[i]*fake_desigmoid(outputs[i])*inputs[j])

        layer_deltas.append(neuron_deltas) 
    
    gradients = get_layer_bias_gradients(lr, errors, outputs)
    return layer_deltas, gradients


def get_layer_inputs(net, layer_num):
    layer_inputs  = []
    for i in range(len(net[layer_num])):
        layer_inputs.append(net[layer_num][i].activation)

    return layer_inputs

def adjust_weights(deltas, layer_num, net):
    for i in range(len(deltas)):
        for j in range(len(deltas[i])):
            net[layer_num][i].weights[j] += deltas[i][j]

    return net


def adjust_bias(gradients, layer_num, net):
    for i in range(len(gradients)):
        net[layer_num][i].bias += gradients[i]

    return net



def train(input_data, correct_output, net, lr):
    net_output, net = run_net(input_data, net)
    layer_errors = []
    layer_inputs = []
    for i in range(len(net)-1):
        if(i==0):
            layer_errors.append(get_output_layer_error(net_output, correct_output))
        else:
            layer_errors.append(get_hidden_layer_error(len(net)-(i+1), layer_errors[i-1], net))

        layer_inputs.append(get_layer_inputs(net, len(net)-(i+2)))
        layer_weight_deltas, bias_gradients = get_layer_weights_deltas(lr, layer_errors[i], layer_inputs[i-1], layer_inputs[i])
        net = adjust_weights(layer_weight_deltas, len(net)-(i+1), net)
        net = adjust_bias(bias_gradients, len(net)-(i+1), net)


    return net
    

net = create_net(784, [64], 10)
net_copy = net
start = int(time.time())
for i in range(len(images)):
        net = train(images[i], get_correct_result(labels[i]), net, .1)
        print(f"training: {i}/{len(images)} done, total time: {int((int(time.time())-start)/60)}, est time remaining:  {int((((int(time.time())-start)/(i+1))*(len(images)-i))/60)}")


correct = 0
start = time.time()
for i in range(len(test_images)):
    good, bad = run_net(test_images[i], net)
    if(get_max(good)==test_labels[i]):
        correct+=1
    
    print(f"testing: {i}/{len(test_images)}, {correct}/{i} correct so far {(correct/(i+1))*100}% accuracy total time: {int((int(time.time())-start)/60)}, est time remaining:  {int((((int(time.time())-start)/(i+1))*(len(test_images)-i))/60)}")

if(net==net_copy):
    print("No Training Done Net matches Net Copy")