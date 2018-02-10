from collections import *
from random import random
import pprint
import operator
import math
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der

def train_char_lm(fname, order=4, add_k=1):
  ''' Trains a language model.

  Skeleton code was borrowed from 
  http://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139

  Inputs:
    fname: Path to a text corpus.
    order: The length of the n-grams.
    add_k: k value for add-k smoothing. NOT YET IMPLMENTED

  Returns:
    A dictionary mapping from n-grams of length n to a list of tuples.
    Each tuple consists of a possible net character and its probability.
  '''

  # TODO: Add your implementation of add-k smoothing.

  data = open(fname).read()
  list_of_chars = set(list(data))
  lm = defaultdict(Counter)
  pad = "~" * order
  data = pad + data
  for i in range(len(data)-order):
    history, char = data[i:i+order], data[i+order]
    lm[history][char]+=1
  
  def add_k_func(counter):
    for char in list_of_chars:
      counter[char] = counter[char] + add_k
      counter['OOV'] = add_k
    return counter

  def normalize(counter):
    s = float(sum(counter.values()))
    return [(c,cnt/s) for c,cnt in counter.items()]
  lm = {hist:add_k_func(chars) for hist, chars in lm.items()}
  outlm = {hist:normalize(chars) for hist, chars in lm.items()}
  return outlm


def generate_letter(lm, history, order):
  ''' Randomly chooses the next letter using the language model.
  
  Inputs:
    lm: The output from calling train_char_lm.
    history: A sequence of text at least 'order' long.
    order: The length of the n-grams in the language model.
    
  Returns: 
    A letter
  '''
  
  history = history[-order:]
  dist = lm[history]
  x = random()
  for c,v in dist:
    x = x - v
    if x <= 0: return c
    
    
def generate_text(lm, order, nletters=500):
  '''Generates a bunch of random text based on the language model.
  
  Inputs:
  lm: The output from calling train_char_lm.
  history: A sequence of previous text.
  order: The length of the n-grams in the language model.
  
  Returns: 
    A letter  
  '''
  history = "~" * order
  out = []
  for i in range(nletters):
    c = generate_letter(lm, history, order)
    history = history[-order:] + c
    out.append(c)
  return "".join(out)


def interpolated_perplexity(test_filename, lms, lambdas):
  '''Computes the perplexity of a text file given the language model.
  
  Inputs:
    test_filename: path to text file
    lm: The output from calling train_char_lm.
    order: The length of the n-grams in the language model.
  
  Assumptions:
    Assumes probabilities have been smoothed
    Assumes out of vocabulary words have been handled somehow

  lms: A list of language models, outputted by calling train_char_lm. Assumes 
     the language model with the longest history is first in the list and the language
     model with the shortest history (i.e. 0 ) is last
     Constraints on the lms list suggest the longest history (ie the highest order lm) 
     has order n-1, where n is the length of the list
    
  '''
  data = open(test_filename).read()
  trained_lexicon = list(map(lambda x: x[0], lm[list(lm.keys())[0]]))
  trained_lexicon.remove('OOV')

  N = len(data)
  pad = "~" * order
  data = pad + data
  prob = 0
  for i in range(len(data) - order):
    history, char = data[i:i+order], data[i + order]
    curr_prob = calculate_prob_with_backoff(char, history, lms, lambdas)
    prob += curr_prob
  prob = -1 / N * prob
  perplex = math.exp(prob)
  return perplex

def perplexity(test_filename, lm, order=4):
  '''Computes the perplexity of a text file given the language model.
  
  Inputs:
    test_filename: path to text file
    lm: The output from calling train_char_lm.
    order: The length of the n-grams in the language model.
  
  Assumptions:
    Assumes probabilities have been smoothed
    Assumes out of vocabulary words have been handled somehow
  '''
  data = open(test_filename).read()
  trained_lexicon = list(map(lambda x: x[0], lm[list(lm.keys())[0]]))
  if 'OOV' in trained_lexicon:
    trained_lexicon.remove('OOV')

  N = len(data)
  pad = "~" * order
  data = pad + data
  prob = 0
  for i in range(len(data) - order):
    history, char = data[i:i+order], data[i+order]
    if history in lm:
      possibilities = lm[history]
      char_to_prob = dict(possibilities)
      if char not in trained_lexicon:
        char = 'OOV'
      curr_prob = math.log(char_to_prob[char]) 
    else:
      curr_prob = math.log(1/(len(trained_lexicon) + 1))
    prob += curr_prob
  prob = -1 / N * prob
  perplex = math.exp(prob)
  return perplex

def calculate_prob_with_backoff(char, history, lms, lambdas):
  '''Uses interpolation to compute the probability of char given a series of 
     language models trained with different length n-grams.

   Inputs:
     char: Character to compute the probability of.
     history: A sequence of previous text.
     lms: A list of language models, outputted by calling train_char_lm. Assumes 
     the language model with the longest history is first in the list and the language
     model with the shortest history (i.e. 0 ) is last
     Constraints on the lms list suggest the longest history (ie the highest order lm) 
     has order n-1, where n is the length of the list
     lambdas: A list of weights for each lambda model. These should sum to 1.
    
  Returns:
    Probability of char appearing next in the sequence.
  '''
  first_lm = lms[0]
  trained_lexicon = list(map(lambda x: x[0], first_lm[list(first_lm.keys())[0]]))
  if 'OOV' in trained_lexicon:
    trained_lexicon.remove('OOV')

  prob = 0
  curr_order = len(lms) - 1
  for curr_lm, curr_lambda in zip(lms, lambdas): 
    curr_history = history[len(history) - curr_order: ]
    if curr_history in curr_lm:
      possibilities = curr_lm[curr_history]
      char_to_prob = dict(possibilities)
      if char not in trained_lexicon:
        char = 'OOV'
      curr_prob = curr_lambda * char_to_prob[char] 
    else:
      curr_prob = curr_lambda * len(trained_lexicon)
    prob += curr_prob
    curr_order -= 1

  return prob

def interpolated_perplexity_opt(lambdas, lms, test_filename):
  '''Computes the perplexity of a text file given the language model.
  
  Inputs:
    test_filename: path to text file
    lm: The output from calling train_char_lm.
    order: The length of the n-grams in the language model.
  
  Assumptions:
    Assumes probabilities have been smoothed
    Assumes out of vocabulary words have been handled somehow

  lms: A list of language models, outputted by calling train_char_lm. Assumes 
     the language model with the longest history is first in the list and the language
     model with the shortest history (i.e. 0 ) is last
     Constraints on the lms list suggest the longest history (ie the highest order lm) 
     has order n-1, where n is the length of the list
  lambdas: A 1 dimensional ndarray where each value is a weight for a language model where
  the first corresponds to the highest order model
  '''
  lambdas = lambdas.tolist()
  data = open(test_filename).read()
  first_lm = lms[0]
  trained_lexicon = list(map(lambda x: x[0], first_lm[list(first_lm.keys())[0]]))
  if 'OOV' in trained_lexicon:
    trained_lexicon.remove('OOV')

  N = len(data)
  pad = "~" * (len(lms) - 1)
  data = pad + data
  prob = 0
  for i in range(len(data) - order):
    history, char = data[i:i+order], data[i+order]
    curr_prob = calculate_prob_with_backoff(char, history, lms, lambdas)
    prob += curr_prob
  prob = -1 / N * prob
  perplex = math.exp(prob)
  return perplex

def print_probs(lm, history):
    probs = sorted(lm[history],key=lambda x:(-x[1],x[0]))
    pp = pprint.PrettyPrinter()
    pp.pprint(probs)

def set_lambdas(lms, dev_filename):
  '''Returns a list of lambda values that weight the contribution of each n-gram model

  This can either be done heuristically or by using a development set.

  Inputs:
    lms: A list of language models, outputted by calling train_char_lm.
    dev_filename: Path to a development text file to optionally use for tuning the lmabdas. 

  Returns:
    Probability of char appearing next in the sequence.
  '''
  init_val = 1/len(lms)
  init_vals = init_val * np.ones(len(lms))
  bons = []
  for i in range(len(lms)):
    bons.append((0, 1))
  
  cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
  lambdas = minimize(interpolated_perplexity_opt, init_vals, args=(lms, dev_filename), 
  bounds=bons, constraints=cons, options={'maxiter': 1000, 'disp': True})

  lambdas.x = lambdas
  lambdas = lambdas.tolist()
  return lambdas 

if __name__ == '__main__':
  print('Training language model')
  lm_0 = train_char_lm("shakespeare_input.txt", order=0, add_k = 1)
  lm_1 = train_char_lm("shakespeare_input.txt", order=1, add_k = 1)
  lm_2 = train_char_lm("shakespeare_input.txt", order=2, add_k = 1)
  lm_3 = train_char_lm("shakespeare_input.txt", order=3, add_k = 1)
  lm_4 = train_char_lm("shakespeare_input.txt", order=4, add_k = 1)

  unsmoothed_lm = train_char_lm("shakespeare_input.txt", order=3, add_k = 0)
  print("Shakespeare Perplexity 3 - order and unsmoothed: " +
  str(perplexity('shakespeare_input.txt', unsmoothed_lm, order = 3)))

  lms = [lm_3, lm_2, lm_1, lm_0]
  print("Shakespeare Perplexity 0 - order: " + 
  str(perplexity('shakespeare_input.txt', lm_0, order = 0)))
  print("Shakespeare Perplexity 1 - order: " + 
  str(perplexity('shakespeare_input.txt', lm_1, order = 1)))
  print("Shakespeare Perplexity 2 - order: " +
  str(perplexity('shakespeare_input.txt', lm_2, order = 2)))
  print("Shakespeare Perplexity 3 - order: " +
  str(perplexity('shakespeare_input.txt', lm_3, order = 3)))
  print("Shakespeare Perplexity 4 - order: " +
  str(perplexity('shakespeare_input.txt', lm_4, order = 4)))

  print("War and Peace Perplexity 0 - order: " + 
  str(perplexity('warpeace_input.txt', lm_0, order = 0)))
  print("War and Peace Perplexity 1 - order: " + 
  str(perplexity('warpeace_input.txt', lm_1, order = 1)))
  print("War and Peace Perplexity 2 - order: " +
  str(perplexity('warpeace_input.txt', lm_2, order = 2)))
  print("War and Peace Perplexity 3 - order: " +
  str(perplexity('warpeace_input.txt', lm_3, order = 3)))

  best_lambdas = set_lambdas(lms, "warpeace_input.txt")
  print(best_lambdas)