import torch

words = open('names.txt').read().splitlines()


if __name__ == "__main__":
    #print(words[:10])


    # bigram counts
    for w in words[:4]:
        #add start and end characters
        characters = ['.'] + list(w) + ['.']
        # iterate over two characters at a time
        for ch1, ch2 in zip(characters, characters[1:]):
            bigram = (ch1,ch2)

            print(bigram)
