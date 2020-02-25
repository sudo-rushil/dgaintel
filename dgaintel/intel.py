import os
import re
import numpy as np
import dgaintel.predict

class Intel:
    def __init__(self, whitelist=[]):
        '''
        Whitelist: list of TLDs that the model should always clear.
        '''
        self.whiteset = set(whitelist)
        self.pattern = r'[a-z]+\.[a-z]+$'

    def _get_prediction(self, domain, prob):
        if prob >= 0.5:
            return '{} is DGA with probability {}\n'.format(domain, prob)

        return '{} is genuine with probability {}\n'.format(domain, prob)

    def get_prob(self, domains, raw=False, internal=False):
        if not isinstance(domains, list):
            domains = dgaintel.predict._inputs(domains)

        vec = np.zeros((len(domains), 82))
        white = set()

        for i, domain in enumerate(domains):
            matches = re.findall(self.pattern, domain.strip('/'))
            assert len(matches) == 1, f'Input error: {domain} is an invalid domain'

            if matches[0] in self.whiteset: white.add(i); continue

            for j, char in enumerate(matches[0]):
                if char not in dgaintel.predict.CHAR2IDX: return -1
                vec[i, j] = dgaintel.predict.CHAR2IDX[char]


        prob = dgaintel.predict.MODEL(vec).numpy()
        prob = prob.transpose()[0]

        for i in range(len(prob)):
            if i in white:
                prob[i] = 0

        if not internal:
            if prob.shape[0] == 1:
                return prob.sum()

            if raw:
                return prob

        return list(zip(domains, prob))

    def get_prediction(self, domains, to_file=None, show=True):
        '''
        Wrapper for printing out/writing full predictions on a domain or set of domains
        Input: domain (str), list of domains (list), domains in .txt file (FileObj)
        Output: show to stdout
            show=False: list of prediction strings (list)
            to_file=<filename>.txt: writes new file at <filename>.txt with predictions
        '''
        raw_probs = self.get_prob(domains, internal=True)
        preds = [self._get_prediction(domain, prob) for domain, prob in raw_probs]

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
