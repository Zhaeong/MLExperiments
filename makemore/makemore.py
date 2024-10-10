import torch
import matplotlib.pyplot as plt

#print(torch.__version__)
#print(torch.cuda.is_available())

words = open('names.txt').read().splitlines()


if __name__ == "__main__":
    #print(words[:10])


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
    #for i in sorted_bigrams:
    #    print(i, sorted_bigrams[i])


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

    # model smoothing value so that we dont get infinity loss
    Counts = Counts + 1

    # now change the counts matrix to a probability matrix

    # a column vector
    # sums all rows of Probabilities
    # sum of dimension 1
    # look at the shape to see which dimension to reduce
    # shape = (27,27), sum(0, keepdim=True) = (1,27)
    # so that would sum over all columns
    # shape = (27,27), sum(1, keepdim=True) = (27,1)
    # which would sum over all rows
    Counts_Row_Sum = Counts.sum(1, keepdim=True)

    print(Counts_Row_Sum)

    # to be able to do this look up broadcasting semantics
    # https://pytorch.org/docs/stable/notes/broadcasting.html
    # Counts = (27, 27)
    # C_R_S  = (27,  1)
    # Be careful with this because the dimensions after sum would determine whether 
    # to normalize over column or normalize over rows
    Probabilities = Counts.float() / Counts_Row_Sum

    # sanity check is to make sure the rows sum is equal to 1
    print(Probabilities[0].sum())

 

    '''
    # for plotting the matrix
    plt.figure(figsize=(16,16))
    plt.imshow(Counts, cmap='Blues')
    for i in range(27):
        for j in range(27):
            char_str = index_to_string[i] + index_to_string[j]
            plt.text(j, i, char_str, ha="center", va="bottom", color='gray')
            plt.text(j, i, Counts[i,j].item(), ha="center", va="top", color='gray')
    plt.axis('off')

    
    plt.show()
    '''

    g = torch.Generator().manual_seed(2147483647)
    # Get generator to predict
    #p = torch.rand(3, generator=g)
    '''
    p = Counts[0].float()
    next_char_probability = p/p.sum()
    print(next_char_probability)

    
    character_index = torch.multinomial(next_char_probability, num_samples=1, replacement=True, generator=g).item()
    character_sampled = index_to_string[character_index]


    print(character_index)
    print(character_sampled)
    '''


    for i in range(10):
        output_string = ''
        next_char = ''
        character_index = 0

        while next_char != '.':


            next_char_probability = Probabilities[character_index]

            # Manually calculating probabitilies
            #p = Counts[character_index].float()
            #next_char_probability = p/p.sum()
            
            # this is if it's completly random
            #next_char_probability = torch.ones(27) / 27.0

            character_index = torch.multinomial(next_char_probability, num_samples=1, replacement=True, generator=g).item()
            next_char = index_to_string[character_index]
            output_string += next_char
            character_index = string_to_index[next_char] 

        print(output_string)

    # likelihood of model is the product (mult) of all the probabitilies
    # but that's going to be a a small number, so work with the log likelihood
    log_likelihood = 0.0
    n = 0
    # Evaluate the quality of the model
    for w in ["andrejq"]:
        #add start and end characters
        characters = ['.'] + list(w) + ['.']
        # iterate over two characters at a time
        for ch1, ch2 in zip(characters, characters[1:]):
            ch1_index = string_to_index[ch1]
            ch2_index = string_to_index[ch2]

            bigram_probability = Probabilities[ch1_index][ch2_index]
            log_probability = torch.log(bigram_probability)
            # note: log(a*b*c) = log(a) + log(b) + log(c)
            log_likelihood += log_probability
            n += 1
            print(f'{ch1}{ch2}: {bigram_probability:.4f} {log_probability:.4f}')

    #since for loss function we want model is better when value is closer to 0, we'll want a negative log likelihood 
    neg_log_likelihood = -log_likelihood

    #normalized neg_log_likelihood
    norm_neg_log_likelihood = neg_log_likelihood / n


    # 2.4541
    print(norm_neg_log_likelihood)

    # likelihood of model is the product (mult) of all the probabitilies
    # but that's going to be a a small number, so work with the log likelihood

