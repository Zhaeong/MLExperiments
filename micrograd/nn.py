from engine import Value
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
        return 0.0
    
if __name__ == "__main__":
    print("Start")
    a = Neuron(4)
    print(a.w)
    print(

