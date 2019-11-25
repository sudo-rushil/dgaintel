from tensorflow.keras.models import load_model
import numpy as np
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
saved_model_path = os.path.join(dir_path, 'domain_classifier_model.h5')

model = load_model(saved_model_path)
char2idx = {'-': 0, '.': 1, '0': 2, '1': 3, '2': 4, '3': 5, 
            '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11, 
            '_': 12, 'a': 13, 'b': 14, 'c': 15, 'd': 16, 'e': 17, 
            'f': 18, 'g': 19, 'h': 20, 'i': 21, 'j': 22, 'k': 23, 
            'l': 24, 'm': 25, 'n': 26, 'o': 27, 'p': 28, 'q': 29, 
            'r': 30, 's': 31, 't': 32, 'u': 33, 'v': 34, 'w': 35, 
            'x': 36, 'y': 37, 'z': 38}

def get_prediction(domain_name, model=model, mapping=char2idx):
  domain_name = domain_name.lower()

  name_vec = []
  for c in domain_name:
    if c not in mapping: return -1
    name_vec.append(mapping[c])

  vec = np.zeros((1, 82))
  vec[0, :len(domain_name)] = name_vec

  prediction = model(vec).numpy().sum()

  return prediction

def main():
  p = get_prediction('microsoft.com')
  print('\nmicrosoft.com has a probability of {} of being DGA'.format(p))
  p = get_prediction('vlurgpeddygdy.com')
  print('vlurgpeddygdy.com has a probability of {} of being DGA'.format(p))

if __name__ == '__main__':
  main()