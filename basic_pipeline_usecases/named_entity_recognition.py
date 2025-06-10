from transformers import pipeline

# No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english
# Named entitty recognition is used to classify named entitities in text into predefined categories.
ner = pipeline('ner', grouped_entities=True)
result = ner('My name is Sylvain and I work at Hugging Face in Brooklyn.')

print(result)
# [
#   {'entity_group': 'PER', 'score': np.float32(0.9981694), 'word': 'Sylvain', 'start': 11, 'end': 18}, 
#   {'entity_group': 'ORG', 'score': np.float32(0.9796019), 'word': 'Hugging Face', 'start': 33, 'end': 45}, 
#   {'entity_group': 'LOC', 'score': np.float32(0.9932106), 'word': 'Brooklyn', 'start': 49, 'end': 57}
# ]