import torch
import torch.nn.functional as torchF

import matplotlib.pyplot as plt

words = open('names.txt').read().splitlines()

print(len(words))

# all the characters in the names text file
data_characters = sorted(list(set(''.join(words))))

string_to_index = {}
string_to_index['.'] = 0
index = 1
for c in data_characters:
    string_to_index[c] = index
    index+=1

# reverse mapping
index_to_string = {}
for i in string_to_index:
    index_to_string[string_to_index[i]] = i

print(index_to_string)  

# for training, this is the expected next char from the previous characters
# we want to create neural network that takes the inputs and predicts the labels
inputs = []
labels = []

# context length: how many char do we need to predict the next one
block_size = 3

for w in words:
    #print(w)

    # 0 is the index for beginning or end char '.'
    # so start context with list of dots
    context = [0] * block_size
    for char in w + '.':
        char_index = string_to_index[char]
        inputs.append(context)
        labels.append(char_index)

        inputString = ''
        for i in context:
            inputString += index_to_string[i]
        #print("input:", inputString, context, "Prediction:", index_to_string[char_index], char_index)

        # increment to next char, crop list and append next
        context = context[1:] + [char_index]

        # e.g.
        # input: .em [0, 5, 13] Prediction: m [13]


inputs = torch.tensor(inputs)
labels = torch.tensor(labels)

# a mapping of character to 2D vectors
CharLookupTable = torch.randn((27, 2))
def training():

    # forward pass
    neurons = 100

    # hidden layer
    # 6 inputs because 2 dimensional embedding * 3 context length
    weights_1 = torch.randn((6, neurons))
    biases_1 = torch.randn(100)

    # now create output layer
    weights_2 = torch.randn(neurons, 27)
    biases_2 = torch.randn(27)

    parameters = [CharLookupTable, weights_1, biases_1, weights_2, biases_2]
    print("Num Params:", sum(p.nelement() for p in parameters)) 
    for p in parameters:
        p.requires_grad = True

    for i in range(100):

        # minibatch, so sample just a random number of input -> labels
        examples_used = 32
        ix = torch.randint(0, inputs.shape[0], (examples_used,))

        embedding = CharLookupTable[inputs[ix]]
        emb_cat = embedding.view(-1, 6)
        hidden_states = torch.tanh(emb_cat @ weights_1 + biases_1)
        logits = hidden_states @ weights_2 + biases_2
    
        loss = torchF.cross_entropy(logits, labels[ix])
    
        print(loss.item())
    
        # back propagation
        for p in parameters:
            p.grad = None
        loss.backward()
    
        # update
        learning_rate = 0.1
        for p in parameters:
            p.data += -learning_rate * p.grad
    
    return parameters

def evaluate_loss(parameters):

    # indices based on parameters = [CharLookupTable, weights_1, biases_1, weights_2, biases_2]
    embedding = parameters[0][inputs]
    emb_cat = embedding.view(-1, 6)
    hidden_states = torch.tanh(emb_cat @ parameters[1] + parameters[2])
    logits = hidden_states @ parameters[3] + parameters[4]
    
    loss = torchF.cross_entropy(logits, labels)
    
    print("Full dataset loss:", loss.item())

if __name__ == "__main__":
    print("Start")
    
    params = training()

    evaluate_loss(params)
    exit()


    # first arg is row, which is the same as list,
    # second is columns, which is the indexes in the list
    # 2, 3:
    # [[0,0,0],
    #  [0,0,0]]
    #tester = torch.ones(32, 3)

    # for embedding the 27 characters into 2 dimensions
    LookupTable = torch.randn((27, 2))

    # we want to convert the inputs into embeddings
    # pytorch allows for multi indexing via lists
    # e.g. 
    # LookupTable[1,2,3] = [[a,b],[c,d],[e,f]
    # inputs[1][2] this is 5, which corresponds to char 'e'

    # so LookupTable[5] = LookupTable[inputs][1,2]

    embedding = LookupTable[inputs]
    # shape = [32,3,2]
    # 32 examples, 3 context integers, each integer has 2 dimensions

    # we can concatenate 3,2 into 6
    # can also do emb_cat = embedding.view(embedding.shape[0], 6)
    emb_cat = embedding.view(-1, 6)

    
    neurons = 100
    # 6 inputs because 2 dimensional embedding * 3 context length
    weights_1 = torch.randn((6, neurons))
    biases_1 = torch.randn(100)

    # @ is matrix multiply
    hidden_states = torch.tanh(emb_cat @ weights_1 + biases_1)
    print(hidden_states)
    print(hidden_states.shape)

    # now create output layer
    weights_2 = torch.randn(neurons, 27)
    biases_2 = torch.randn(27)


    # for 32 examples
    # embedding shape = (32, 6)
    # hidden layer = (6, 100)
    # states = (32, 100)
    # (32, 6) X (6, 100) = (32, 100)
    # each example has an activation in the hidden state

    # then output needs to compess the 100 neurons back into 27 likelihoods
    # (32, 100) X (100, 27) = (32, 27)

    # log counts
    logits = hidden_states @ weights_2 + biases_2

    # now exponetiate and divide to get probabilities
    # (known as softmax)

    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    print(probs)

    # using labels we can see the probability of neuro network of predicting the correct next char
    # for 32 examples
    #prediction = probs[torch.arange(32), labels]
    #print(prediction)

    # negative log likelihood, want to minimize this
    # loss_prev = -prediction.log().mean()
    loss = torchF.cross_entropy(logits, labels)
    print(loss)

    
    parameters = [LookupTable, weights_1, biases_1, weights_2, biases_2]
    print("Num Params:", sum(p.nelement() for p in parameters)) 



    



