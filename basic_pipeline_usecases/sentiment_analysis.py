from transformers import pipeline

# No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english
# Encoder only architecture
# Sentiment analysis takes a string and returns a label and a score to determine the sentiment of the input text, whether postive or negative.
classifier = pipeline('sentiment-analysis')

result = classifier(
  [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
   ]
)

print(result)
[
  {'label': 'POSITIVE', 'score': 0.9598050713539124},
  {'label': 'NEGATIVE', 'score': 0.9994558691978455}
]