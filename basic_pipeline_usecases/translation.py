from transformers import pipeline

# model explicity chosen
# Transformers original task
# encoder and decoder arch
translator = pipeline('translation', model='Helsinki-NLP/opus-mt-fr-en')
result = translator('Ce cours est produit par Hugging Face.')

print(result)

# [{'translation_text': 'This course is produced by Hugging Face.'}]