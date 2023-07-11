from mnist import MNIST
import random 
import math

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
            net[i].append(hidden_neuron(0, i, j, [random.randint(40, 60)/100 for i in range(len(net[i-1]))], random.randint(-100, 100)/100))
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
    output[image_label+1] = 1.0

    return output

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

def get_layer_bias_gradients(lr, errors, outputs):
    gradients = []
    for i in range(len(errors)):
        gradients.append(lr*errors[i]*fake_desigmoid(outputs[i]))

    return gradients


def get_layer_weights_deltas(lr, errors, outputs, inputs):
    layer_deltas = []
    for i in range(len(errors)):
        neuron_deltas = []
        for j in range(len(inputs)):
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
    output_layer_errors = get_output_layer_error(net_output, correct_output)
    layer_input = get_layer_inputs(net, 1)
    layer_weight_deltas, bias_gradients = get_layer_weights_deltas(lr, output_layer_errors, net_output, layer_input)
    net = adjust_weights(layer_weight_deltas, 2, net)
    net = adjust_bias(bias_gradients, 2, net)

    hidden_layer_errors = get_hidden_layer_error(1, output_layer_errors, net)
    hidden_layer_inputs = get_layer_inputs(net, 0)
    layer_weight_deltas, bias_gradients = get_layer_weights_deltas(lr, hidden_layer_errors, layer_input, hidden_layer_inputs)
    net = adjust_weights(layer_weight_deltas, 1, net)
    net = adjust_bias(bias_gradients, 1, net)

    return net
    
training_data = [
    {
        "inputs": [0, 0],
        "targets": [0]
    }, 
     {
        "inputs": [1, 0],
        "targets": [1]
    }, 
     {
        "inputs": [0, 1],
        "targets": [1]
    }, 
     {
        "inputs": [1, 1],
        "targets": [0]
    }, 
]

net = create_net(2, [10], 1)

for i in range(500000):
        data = random.choice(training_data)   
        net = train(data["inputs"], data["targets"], net, .1)



for data in training_data:
    good, bad = run_net(data["inputs"], net)
    print(good)


    