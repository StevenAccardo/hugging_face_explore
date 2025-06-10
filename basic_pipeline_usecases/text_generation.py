from transformers import pipeline
# No model was supplied, defaulted to openai-community/gpt2
# Text generation is a technique that allows you to generate text based on a prompt.
# This uses the decoder-only architecture, which excels at predicting the next token in a sequence.
generator = pipeline('text-generation')
result = generator('In this course, we will teach you how to')
print(result)
# [
#   {
#     'generated_text': 'In this course, we will teach you how to use the Transformers library to build your own models. We will start with a simple model and then build up to a more complex model. We will also cover how to use the library to finetune existing models.'
#   }
# ]


# Text generation with a model selected
generator = pipeline("text-generation", model="distilgpt2")
result2 = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(result2)
# [
#   {
#     'generated_text': 'In this course, we will teach you how to get the most out of a project.\n\n\n\nThis course will help you understand how to find a good solution to the problem.\nWe will teach you how to get the most out of a project.\nThis course will help you understand how to get the most out of a project.\nWe will teach you how to get the most out of a project.\nThis course will help you understand how to get the most out of a project.\nWe will teach you how to get the most out of a project.\nThis course will help you understand how to get the most out of a project.\nWe will teach you how to get the most out of a project.\nWe will teach you how to get the most out of a project.\nWe will teach you how to get the most out of a project.\nWe will teach you how to get the most out of a project.\nWith this course, we will teach you how to get the most out of a project.\nWe will teach you how to get the most out of a project.\nYou can learn how to get the most out of a project.\nYou can learn how to get the most out of a project.\nWe will teach you how to get the most'
#   }
# ]