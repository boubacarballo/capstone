from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/nli-deberta-v3-large')
\
label_mapping = ['contradiction', 'entailment', 'neutral']
labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]


print(type(scores))
print(scores)