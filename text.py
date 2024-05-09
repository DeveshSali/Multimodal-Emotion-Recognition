from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("wesleyacheng/twitter-emotion-classification-with-bert")
model = AutoModelForSequenceClassification.from_pretrained("wesleyacheng/twitter-emotion-classification-with-bert")

from transformers import pipeline
import torch

twitter_emotion_classifier = pipeline(task='text-classification', model=model, tokenizer=tokenizer)

tweet = """
I hate you!
"""

jeet = twitter_emotion_classifier(tweet)
print(jeet)