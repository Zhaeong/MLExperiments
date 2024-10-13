import torch
import torch.nn.functional as torchF

words = open('names.txt').read().splitlines()
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

# training set for bigram
bigram_start = []
bigram_end = []

# bigram counts
for w in words:
    #add start and end characters
    characters = ['.'] + list(w) + ['.']
    # iterate over two characters at a time
    for ch1, ch2 in zip(characters, characters[1:]):
        idx1 = string_to_index[ch1]
        idx2 = string_to_index[ch2]

        bigram_start.append(idx1)
        bigram_end.append(idx2)

# convert to tensors
# when the input is idx in bigram_start, the desired output is the idx in bigram_end
bigram_start = torch.tensor(bigram_start)

# also know as 'labels' during training, or ground truth
bigram_end = torch.tensor(bigram_end)

g = torch.Generator().manual_seed(2147483647)

def gradient_descent():
    print('start grad')   
    # initilize random weights
    weights = torch.randn((27,27), generator=g, requires_grad=True)

    for i in range(100):
        # forward pass
        bigram_start_encoded = torchF.one_hot(bigram_start, num_classes=27).float()
        logits = bigram_start_encoded @ weights

        # softmax
        counts = logits.exp()
        probabilities = counts / counts.sum(1, keepdim=True)
    
        # loss
        # note we are calculating the loss based on only the first 5 inputs
        # num_inputs = 5
        num_inputs = bigram_start.nelement()
        loss = -probabilities[torch.arange(num_inputs), bigram_end].log().mean()

        print("loss: ", loss.item())

        # backwards pass
        # set gradients to 0
        weights.grad = None
        loss.backward()

        # update
        # to reduce loss, we subtract based on gradients 
        learning_rate = 50 
        weights.data += -learning_rate * weights.grad
if __name__ == "__main__":


    gradient_descent()

    exit()

    # Use one-hot encoding to turn the inputs (integers) into vectors which turns the integer dimension into 1
    # and the rest into 0s, depending on number of classes
    # e.g. we have 27 classes, so integer 3 becomes [0, 0, 0, 1, 0, .... 0], with vector size 27

    # note, also cast to float, since we want inputs to be floats for NNs
    bigram_start_encoded = torchF.one_hot(bigram_start, num_classes=27).float()
    print(bigram_start_encoded)

    # randomize initial weights to NN
    # 1 neuron with 27 weights
    # weights = torch.randn((27,1))
    # 27 neurons with 27 weights
    # 1 layer of neuron net with 27 neurons

    weights = torch.randn((27,27), generator=g)
    print(weights)

    # multiply the input with the weights
    # this gets the activation (weights) of each neuron on the input, so a 5x27 matrix
    # (inputs , classes (27)) @ (27, 27) = (inputs x 27)
    # so for 3 inputs
    # (3, 27) x (27, 27) = (3, 27)
    # so it's each neuron's weights on each input
    # activations at (3, 18) means 18's neuron's weights on the third input
    # these are logits = log-counts
    logits = bigram_start_encoded @ weights

    print(logits)

    # these are random negative and positive numbers, in the end we want the weights to 
    # be probabilities of next characters

    # exponetiating them turns them positive
    # e ^ x
    logits_exp = logits.exp()
    # note: log(e^x) = x which is why we called logits log counts
    # roughly equivalent to previously use probability matrix of counts
    counts = logits_exp
    print(logits_exp)

    # probabilities are just the counts normalized
    probabilities = counts / counts.sum(1, keepdim=True)

    # so each row in matrix is the probability of the next char
    # note: exponetiating and then normalizing is called a 'softmax'
    # softmax takes random inputs and then outputs a probability distribution
    # essentially first making all numbers positive (exp), and then dividing them all by the sum
    print(probabilities)

    # this is the probabilities of the next characters for input 1
    print(probabilities[1].sum())


    # we can get the negative log likelihood for inputs
    input_one = bigram_start[0].item()
    label_one = bigram_end[0].item()

    char_input = index_to_string[input_one]
    char_label = index_to_string[label_one]

    input_one_probs = probabilities[0]

    print(f'bigram {char_input}, {char_label}')
    print(f'bigram {input_one}, {label_one}')
    print(f'probabilities:')
    print(input_one_probs)
    input_one_prob_correct = input_one_probs[label_one]
    print('probability of correct label:',  input_one_prob_correct.item())

    # calculate negative log likelihood 
    input_one_neg_log_likelihood = -torch.log(input_one_prob_correct)
    print('neg log likelihood for input 0:', input_one_neg_log_likelihood.item())

    # the loss is just the average of the negative log likelihood for all input, label pairs
    # we can index each input in probability with the labels
    
    # gives a list from 0...number
    input_list = torch.arange(5)

    # this gives us the probabilies of each label for each input in input list
    label_probs = probabilities[input_list, bigram_end]
    print(label_probs)

    # average the negative log of probs to get the loss
    loss = -torch.log(label_probs).mean()
    print(loss)

    # single line:
    # loss = -probabilities[torch.arange(5), bigram_end].log().mean()


    # since we can calculate loss, we can initiate back propagation to tune the weights
    



