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

if __name__ == '__main__': #For training a model
  #Data Loading
  train_files = ['BERT Codebook/Codebook (1).csv', 'BERT Codebook/Codebook (2).csv', 'BERT Codebook/Codebook (3).csv', 
    'BERT Codebook/Codebook (4).csv', 'BERT Codebook/Codebook (5).csv', 'BERT Codebook/Codebook (6).csv', 
    'BERT Codebook/Codebook (7).csv', 'BERT Codebook/Codebook (8).csv', 'BERT Codebook/Codebook (9).csv']
  test_files = ['BERT Codebook/Codebook (10).csv', 'BERT Codebook/Codebook (11).csv', 'BERT Codebook/Codebook (12).csv']
  dataset = load_dataset('csv', data_files={'train': train_files, 'test': test_files}) #, encoding = "ISO-8859-1")

  print (dataset)

  #Preprocessing
  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

  remove_columns = ['Primary','ID', 'text']
  dataset = dataset.map(tokenize_data, batched=True)
  dataset = dataset.map(transform_labels, remove_columns=remove_columns)

  print (dataset)

  # print (tokenizer("I am a boy", truncation=True, padding=True, max_length=256))
  classify("Covid is bad", "./bert_model")

  #Training
  # training_args = TrainingArguments("test_trainer", num_train_epochs=3)

  train_dataset = dataset['train'].shuffle(seed=10).select(range(8000))
  eval_dataset = dataset['train'].shuffle(seed=10).select(range(8000,9000))

  model_exist = True #Change to False to train model

  if not model_exist:
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    training_args = TrainingArguments(
        output_dir="./bert_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

  else:
    model = AutoModelForSequenceClassification.from_pretrained('./bert_model', local_files_only=True)

    training_args = TrainingArguments(
        output_dir="./bert_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
    )

    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
    )

  metric = load_metric("accuracy")

  eval_results = trainer.predict(test_dataset=eval_dataset)
  logits = eval_results[0]
  probabilities = softmax(logits)
  print (probabilities)

  # trainer.save_model('./bert_model')