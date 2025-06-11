from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
# This model is instantiated with random weights, and while useable it will output gibberish.
# This would be the route you want if you wanted to train the model from scratch
model = BertModel(config)

print(config)
# BertConfig {
#   'attention_probs_dropout_prob': 0.1,
#   'classifier_dropout': null,
#   'hidden_act': 'gelu',
#   'hidden_dropout_prob': 0.1,
#   'hidden_size': 768,
#   'initializer_range': 0.02,
#   'intermediate_size': 3072,
#   'layer_norm_eps': 1e-12,
#   'max_position_embeddings': 512,
#   'model_type': 'bert',
#   'num_attention_heads': 12,
#   'num_hidden_layers': 12,
#   'pad_token_id': 0,
#   'position_embedding_type': 'absolute',
#   'transformers_version': '4.52.4',
#   'type_vocab_size': 2,
#   'use_cache': true,
#   'vocab_size': 30522
# }