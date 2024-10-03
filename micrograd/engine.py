from graphviz import Digraph
import math
# Graph visualizer
def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes = set()
    edges = set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':"LR"}) # LR = left to right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name=uid, label = "{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')

        if n._op:
            # this means the Value in the graph is from some operation, create an op node
            dot.node(name=uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    dot.render('graph', view=True)

def manualDerivative():
    h = 0.0001

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label='e'
    d = e+c; d.label='d'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L1 = L.data

    draw_dot(L)
    # a is incremented by small value h
    a = Value(2.0 + h, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label='e'
    d = e+c; d.label='d'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L2 = L.data 

    # this is calculating the derivative of L with respect to a
    # measuring the change in value (rise/run)
    print((L2-L1)/h)

def manual2DNeuron():
    # inputs of neuron
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights of neuron
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias of neuron
    # chosen to make the other values easier to look at
    b = Value(6.88137, label='b')

    x1w1 = x1*w1; x1w1.label='x1*w1' 
    x2w2 = x2*w2; x2w2.label='x2*w2' 

    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'

    # the cell body without the activation function
    n = x1w1x2w2 + b; n.label='n'

    # tanh is a hyperbolic function for activation
    o = n.tanh(); o.label = 'output'

    # back propagation manually

    # manual setting of o's gradients
    # output's gradient is 1, d_o/d_o = 1
    o.grad = 1.0

    # o = tanh(n)
    # what is local derivative of o with respect to n:
    # see wiki for derivative of tanh
    # d_o/d_n = 1 - tanh(n)**2
    #       = 1 - o**2

    n.grad = 1 - o.data**2

    # a '+' derivative is just the same as the previous
    # n = x1w1x2w2 + b
    # what is derivative of n with respect to x1w1x2w2
    # d_n/d_x1w1x2w2
    
    # derivative =  (f(x+h) - f(x)) / h
    # = (x1w1x2w2 + h + b) - (x1w1x2w2 + b) / h
    # = x1w1x2w2 + h + b - x1w1x2w2 - b / h 
    # = h/h 
    # = 1

    # so in + case the derivative d_n/d_x1w1x2w2 is 1
    # then via chain rule
    # then the derivative of d_o/d_x1w1x2w2 = d_o/d_n * d_n/d_x1w1x2w2
    x1w1x2w2.grad = n.grad * 1
    b.grad = n.grad * 1

    # x1w1x2w2 = x1w1 + x2w2
    # this is also '+' so it becomes similar
    # d_o/d_x1w1 = d_o/d_n * d_n/d_x1w1x2w2 * d_x1w1x2w2/d_x1w1 
    x1w1.grad = n.grad * 1 * 1 # or x1w1x2w2.grad * 1
    x2w2.grad = n.grad * 1 * 1

    # x1w1 = x1 * w1 
    # so this is *, we need to find the derivative
    # d_x1w1/d_x1 
    # derivative =  (f(x+h) - f(x)) / h
    # = ((x1 + h) * w1) - (x1 * w1) / h 
    # = (x1 * w1 + h * w1) - x1*w1 / h 
    # = h * w1 / h 
    # = w1  
    
    # so d_x1w1/d_x1 = w1 
    # and d_x1w1/d_w1 = x1
    # then d_o/d_x1 = d_o/d_n * d_n/d_x1w1x2w2 * d_x1w1x2w2/d_x1w1 * d_x1
    x1.grad = n.grad * 1 * 1 * w1.data # or x1w1.grad * w1.data
    w1.grad = n.grad * 1 * 1 * x1.data # or x1w1.grad * x1.data


    x2.grad = n.grad * 1 * 1 * w2.data
    w2.grad = n.grad * 1 * 1 * x2.data


    # a '*' derivative is just the other one

    draw_dot(o)
    return o


def backprod2DNeuron():
    # inputs of neuron
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights of neuron
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias of neuron
    # chosen to make the other values easier to look at
    b = Value(6.88137, label='b')

    x1w1 = x1*w1; x1w1.label='x1*w1' 
    x2w2 = x2*w2; x2w2.label='x2*w2' 

    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'

    # the cell body without the activation function
    n = x1w1x2w2 + b; n.label='n'

    # tanh is a hyperbolic function for activation
    o = n.tanh(); o.label = 'output'

    # back propagation by calling _backward function

    # manual setting of o's gradients
    # output's gradient is 1, d_o/d_o = 1
    o.grad = 1.0

    o._backward()

    n._backward()

    x1w1x2w2._backward()
    b._backward()

    x1w1._backward()
    x2w2._backward()


    draw_dot(o)

def auto2DNeuron():
    # inputs of neuron
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # weights of neuron
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # bias of neuron
    # chosen to make the other values easier to look at
    b = Value(6.88137, label='b')

    x1w1 = x1*w1; x1w1.label='x1*w1' 
    x2w2 = x2*w2; x2w2.label='x2*w2' 

    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'

    # the cell body without the activation function
    n = x1w1x2w2 + b; n.label='n'

    # tanh is a hyperbolic function for activation
    o = n.tanh(); o.label = 'output'

    o.backward()

    draw_dot(o)



class Value:

    # If just initializing Value, it will be empty, however when opertion is done
    # it will add the operands into the returned value as children
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        # The derivative of the output with respect to this Value 
        # Calculated via chain rule
        # i.e. the effect it has on the output
        # initialized to 0 because it has no effect initially
        self.grad = 0.0
        
        # A function to store how to chain the output gradients
        self._backward = lambda: None

        self._prev = set(_children)
        self._op = _op
        self.label = label
    
    # To print out a nicer looking value
    def __repr__(self):
        return f"Value: (data={self.data})"

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # ------------------
    # ----Operations----
    # Returns another Value object of the done operation
    # with the parameters added as previous
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        data_value = self.data + other.data
        out = Value(data_value, (self, other), '+')

        # calculating the derivative of the inputs with respect to output
        # out = self + other
        # d_out/d_self = 1
        # Then multiply with output's gradient due to chain rule
        # Then add to the existing gradient
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        data_value = self.data * other.data
        out = Value(data_value, (self, other), '*')

        # derivatives of inputs with respect to output
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # https://en.wikipedia.org/wiki/Hyperbolic_functions
    def tanh(self):
        # n is the neuron prior to activation function
        n = self.data
        t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        out = Value(t, (self,), 'tanh')

        # derivatives of inputs with respect to output
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    #------------------




if __name__ == "__main__":
    print("Start")

    #manualDerivative()
    #manual2DNeuron()
    #backprod2DNeuron()
    auto2DNeuron()
