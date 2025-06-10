from transformers import pipeline

# No model was supplied, defaulted to facebook/bart-large-mnli
# Zero shot classification is a technique that allows you to classify text into a category without training a model on that category.
classifier = pipeline('zero-shot-classification')
results = classifier(
    'This is a course about the Transformers library',
    candidate_labels=['education', 'politics', 'business'],
)

print(results)
# {
#   'sequence': 'This is a course about the Transformers library',
#  'labels': ['education', 'business', 'politics'],
#  'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]
# }