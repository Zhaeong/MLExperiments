import torch
import matplotlib.pyplot as plt

#print(torch.__version__)
#print(torch.cuda.is_available())

words = open('names.txt').read().splitlines()


if __name__ == "__main__":
    print(words[:10])


    # bigram counts
    Bigrams = {}
    for w in words:
        #add start and end characters
        characters = ['.'] + list(w) + ['.']
        # iterate over two characters at a time
        for ch1, ch2 in zip(characters, characters[1:]):
            bigram = (ch1,ch2)

            # Get the bigram in the dictionary
            # if it doesn't exist, default to 0, and then add 1
            Bigrams[bigram] = Bigrams.get(bigram, 0) + 1
    
    # Sort the dictionary by the counts
    sorted_bigrams = dict(sorted(Bigrams.items(), key = lambda kv: -kv[1]))

    # Now use a 2D array to represent the bigram counts
    for i in sorted_bigrams:
        print(i, sorted_bigrams[i])


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

    print(string_to_index)
    print(index_to_string)

    Counts = torch.zeros((27,27), dtype=torch.int32)

    for w in words:
        #add start and end characters
        characters = ['.'] + list(w) + ['.']
        # iterate over two characters at a time
        for ch1, ch2 in zip(characters, characters[1:]):
            start_char = string_to_index[ch1]
            end_char = string_to_index[ch2]

            #Counts[start_char][end_char] += 1
            Counts[start_char, end_char] += 1

    print(Counts)
 

    plt.figure(figsize=(16,16))
    plt.imshow(Counts, cmap='Blues')
    for i in range(27):
        for j in range(27):
            char_str = index_to_string[i] + index_to_string[j]
            plt.text(j, i, char_str, ha="center", va="bottom", color='gray')
            plt.text(j, i, Counts[i,j].item(), ha="center", va="top", color='gray')
    plt.axis('off')

    
    plt.show()
