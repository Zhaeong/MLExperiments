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


if __name__ == "__main__":
    print("Start")

    inputs = []
    labels = []

    for w in words[:5]:
        print(w)

    tester = torch.ones(32, 3)
    print(tester)
