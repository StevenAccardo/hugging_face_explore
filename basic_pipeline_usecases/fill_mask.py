from transformers import pipeline

# No model was supplied, defaulted to distilbert/distilroberta-base
# Fill mask will try to predict the word that goes in the place of the mask token
unmasker = pipeline('fill-mask')
# top_k denotes how many results you want back
# <mask> is the mask token. Changes depending on the model used
results = unmasker('This course will teach you all about <mask> models.', top_k=2)
print(results)

# The score is the probablility that the guessed answer is correct. This is not a true probability, but the softmax divided by the logits.

# 2 guesses due to the top_k value
# [
#   {'score': 0.19620104134082794, 'token': 30412, 'token_str': ' mathematical', 'sequence': 'This course will teach you all about mathematical models.'},
#   {'score': 0.040527429431676865, 'token': 38163, 'token_str': ' computational', 'sequence': 'This course will teach you all about computational models.'}
# ]