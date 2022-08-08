from datasets import load_dataset, load_metric
import datasets
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import pandas as pd
from scipy.special import softmax

def transform_labels(label):
  num = label['Primary']
  return {'labels' : num}

def tokenize_data(example):
    return tokenizer(example['text'], truncation=True, padding=True, max_length=256)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def classify(tweet, model_loc):
  model = AutoModelForSequenceClassification.from_pretrained(model_loc, local_files_only=True)

  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

  tweet_token = [tokenizer(tweet, truncation=True, padding=True, max_length=256)]

  training_args = TrainingArguments(
      output_dir=model_loc,
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=8,
      num_train_epochs=1,
      weight_decay=0.01,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
  )

  metric = load_metric("accuracy")

  data = datasets.Dataset.from_pandas(pd.DataFrame(data=tweet_token))

  pred = trainer.predict(test_dataset = data)
  logits = pred[0][0]
  probabilities = softmax(logits)
  print ("Probabilities = ", probabilities)

  return probabilities