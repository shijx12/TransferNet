import pickle
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt', required=True)
    parser.add_argument('--pt', required=True)
    args = parser.parse_args()
    
    glove = {}
    for line in open(args.txt, encoding='latin-1'):
        w, *vector = line.strip().split(' ')
        vector = list(map(float, vector))
        vector = np.asarray(vector)
        glove[w] = vector

    with open(args.pt, 'wb') as f:
        pickle.dump(glove, f)

if __name__ == '__main__':
    main()
