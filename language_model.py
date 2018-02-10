from collections import *
from random import random
import pprint
import operator
import math

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


def interpolated_perplexity(test_filename, lms, lambdas, order=4):
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
  trained_lexicon = map(lambda x: x[0], lm[lm.keys()[0]])
  trained_lexicon.remove('OOV')

  N = len(data)
  pad = "~" * order
  data = pad + data
  prob = 0
  for i in range(len(data) - order):
    history, char = data[i:i+order], data[i]
    curr_prob = calculate_prob_with_backoff(char, history, lms, lambdas)
    prob += curr_prob
  prob = math.exp(prob)
  perplex = 1 / prob
  perplex = math.pow(perplex, 1/N)
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
  trained_lexicon = map(lambda x: x[0], lm[lm.keys()[0]])
  trained_lexicon.remove('OOV')

  N = len(data)
  pad = "~" * order
  data = pad + data
  prob = 0
  for i in range(len(data) - order):
    history, char = data[i:i+order], data[i]
    if history in lm:
      possibilities = lm[history]
      char_to_prob = dict(possibilities)
      if char not in trained_lexicon:
        char = 'OOV'
      curr_prob = math.log(char_to_prob[char]) 
    else:
      curr_prob = math.log(1/(len(trained_lexicon) + 1))
    prob += curr_prob
  prob = math.exp(prob)
  perplex = 1 / prob
  perplex = math.pow(perplex, 1/N)
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
  trained_lexicon = map(lambda x: x[0], first_lm[first_lm.keys()[0]])
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
  # TODO: YOUR CODE HERE
  pass

if __name__ == '__main__':
  print('Training language model')
  lm = train_char_lm("shakespeare_input.txt", order=0)
  print_probs(lm, ' ')
  print(len(lm))
  sum = 0
  print(len(lm['']))
  for char, prob in lm['']:
    sum += prob
  print(type(lm['']))
  print(sum)