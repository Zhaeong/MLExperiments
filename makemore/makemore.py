import torch

#print(torch.__version__)
#print(torch.cuda.is_available())

words = open('names.txt').read().splitlines()

if __name__ == "__main__":
    print(words[:10])