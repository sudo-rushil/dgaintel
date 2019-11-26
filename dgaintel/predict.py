from tensorflow.keras.models import load_model
import numpy as np
import os

'''
Loads DGA prediction model from saved keras file in module directory.
'''
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

'''
Loads any input as a list of domain names; provides inheritable support for polymorphism.
Input: single domain (str), list of domains (list), domains in .txt file (FileObj)
Output: list of domains (list)
'''
def _inputs(domains):
  lpath = os.path.splitext(domains)

  if lpath[1] == ".txt":
    path = os.path.join(os.getcwd(), domains)

    with open(path, 'r') as dfile:
      domain_list = dfile.readlines()
      domain_list = [domain.strip('\n').lower() for domain in domain_list]

    return domain_list

  elif type(domains) == list:
    return [domain.lower() for domain in domains]

  else:
    return [domains.lower()]

'''
Wrapper to return printable/writable strings.
Input: domain_name (str), prob (float) (optional)
Output: prediction (str)
'''
def _get_prediction(domain_name, prob=None):
  if not prob:
    prob = _get_prob([domain_name])

  if prob >= 0.5:
    return '{} is DGA with probability {}\n'.format(domain_name, prob)
  else:
    return '{} is genuine with probability {}\n'.format(domain_name, prob)

'''
Core inference function; calls model on vectorized batch of domain names.
Input: list of domains (list)
Output: len(domains) == 1: single probability value
        raw=False: list of tuples of format (domain_name, probability)
        raw=True: np.ndarray of probabilities
'''
def get_prob(domains, raw=False, internal=False):
  vec = np.zeros((len(domains), 82))

  for i, domain in enumerate(domains):
    for j, char in enumerate(domain):
      if char not in char2idx: return -1
      vec[i, j] = char2idx[char]


  prob = model(vec).numpy()
  prob = prob.transpose()[0]

  if not internal:
    if prob.shape[0] == 1:
      return prob.sum()

    if raw:
      return prob

  return list(zip(domains, list(prob)))

'''
Wrapper for printing out/writing full predictions on a domain or set of domains; can be called directly
Input: domain (str), list of domains (list), domains in .txt file (FileObj)
Output: show to stdout
        show=False: list of prediction strings (list)
        to_file=<filename>.txt: writes new file at <filename>.txt with predictions 
'''
def get_prediction(domains, to_file=None, show=True):
  raw_probs = get_prob(_inputs(domains), internal=True)
  preds = [_get_prediction(domain, prob=prob) for domain, prob in raw_probs]

  if to_file:
    assert os.path.splitext(to_file)[1] == ".txt"

    with open(os.path.join(os.getcwd(), to_file), 'w') as outfile:
      outfile.writelines(preds)
    return None

  if show:
    for pred in preds:
      print(pred.strip('\n'))
    return None

  return preds

def main():
  get_prediction(['microsoft.com', 'squarespace.com', 'hsfkjdshfjasdhfk.com', 'fdkhakshfda.com', 'foilfencersarebad.com', 'discojjfdsf.com', 'fasddafhkj.com', 'wikipedai.com'])
  
if __name__ == '__main__':
  main()
