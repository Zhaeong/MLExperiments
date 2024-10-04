from engine import Value, draw_dot
import random


class Neuron:
    def __init__(self, num_inputs):
        weights_list = []
        for i in range(num_inputs):
            weights_list.append(Value(random.uniform(-1,1)))

        self.w = weights_list
        self.b = Value(random.uniform(-1,1))

    # call all elements of x with all elements of weights
    def __call__(self, x):
        # w * x + b
        activation = self.b 
        for wi, xi in zip(self.w, x):
            val = wi * xi
            activation += val

        # above can be all done in single line like thus, but less readable
        # act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
        
        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
    
class NeuronLayer:
    def __init__(self, num_inputs, num_neurons):
        neuron_list = []
        for i in range(num_neurons):
            neuron_list.append(Neuron(num_inputs))
        self.neurons = neuron_list

    def __call__(self, x):
        # call forward pass of all neurons
        outputs = []
        for n in self.neurons:
            outputs.append(n(x))
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
        
class MultiLayerPerceptron:
    # num_neurons_list defines num neurons of each layer
    def __init__(self, num_inputs, num_neurons_list):
        # total layers including inputs 
        layers = [num_inputs] + num_neurons_list

        layer_list = []
        # iterate over consequtive sizes and create layers for them
        for i in range(len(num_neurons_list)):
            print("Layer inputs:" + str(layers[i]) + " neurons:" + str(layers[i+1]))
            layer_list.append(NeuronLayer(layers[i], layers[i+1]))
        self.layers = layer_list

    def __call__(self, x):
        print("layer inputs")
        for layer in self.layers:
            # note x is updated to be the output of this layer
            # so each layer is called with the output of previous layer
            # this is for the forward pass
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


if __name__ == "__main__":
    print("Start")
    #n = Neuron(2)
    #x = [Value(2.0), Value(3.0)]
    #print(n(x))

    #L = NeuronLayer(2, 3)
    #print(L(x))

    # 3 dimensional input
    num_inputs = [Value(2.0), Value(3.0), Value(-1.0)]

    MLP = MultiLayerPerceptron(3, [4,4,1])


    #print(MLP(num_inputs))
    #draw_dot(MLP(num_inputs)[0])

    xs = [
            [2.0,3.0,-1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets

    # current predictions of model based on inputs
    ypred = [MLP(x) for x in xs]

    print("ypred")
    print(ypred)

    # loss is single number that measures how well the neural net is performing
    # we'll want to minimize the loss

    #subtracting prediction with the desired targets
    difference_list = []
    for ygt, yout in zip(ys, ypred):
        diff = (yout - ygt) ** 2
        difference_list.append(diff)


    loss = sum(difference_list)

    loss.backward()

    # draw_dot(loss)

    print(loss)

    # negative sign because we want to reduce the loss
    for params in MLP.parameters():
        params.data += -0.1 * params.grad


    exit()

    # training loop
    steps = 10
    for k in range(steps):
        # forward
        ypred = [MLP(x) for x in xs]
    
        # get loss
        difference_list = []
        for ygt, yout in zip(ys, ypred):
            diff = (yout - ygt) ** 2
            difference_list.append(diff)

        loss = sum(difference_list)

        #zero grad
        for params in MLP.parameters():
            params.grad = 0.0

        # backwards pass which fills in the gradients
        loss.backward()

        # update (gradient descent)
        # negative sign because we want to reduce the loss
        for params in MLP.parameters():
            params.data += -0.1 * params.grad

        #print(k, loss.data)

    print(ypred)

