from nltk.corpus import conll2002
sent_a = conll2002.iob_sents('esp.testa')
sent_b = conll2002.iob_sents('esp.testb')
for sent in sent_a:
  for word_tuple in sent:
    if (len(word_tuple) != 3):
      print("Weird Tuple: " + str(word_tuple))

for sent in sent_b:
  for word_tuple in sent:
    if (len(word_tuple) != 3):
      print("Weird Tuple: " + str(word_tuple))

print("done tuple check")