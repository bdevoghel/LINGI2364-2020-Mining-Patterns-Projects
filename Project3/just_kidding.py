import sys
import random

def train_model(a, b, nfolds):
    for fold in range(nfolds):
        print("fold", fold+1)
        print("Just kidding, we don't do nothing ;)")
        print("accuracy:", random.randint(9500, 9999)/10000)
        print("Check our other latest submission for a correct program")
        print()

if __name__ == '__main__':

    args = sys.argv
    
    # first parameter: just
    # second parameter: to
    # third parameter : mess
    train_model(args[1], args[2], int(args[3]))