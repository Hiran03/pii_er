# PII NER Assignment Skeleton

This repo is a skeleton for a token-level NER model that tags PII in STT-style transcripts.

## Setup

```bash
pip install -r requirements.txt
```
## Synthetic Data Generation (GPT-4o)

**Prompt**

I need you to create a dataset for a **PII Entity Recognition for Noisy STT Transcripts** task.

You will:

● Detect the following entity types:  
`CREDIT_CARD`, `PHONE`, `EMAIL`, `PERSON_NAME`, `DATE`, `CITY`, `LOCATION`

● Mark **PII = true** for:  
`CREDIT_CARD`, `PHONE`, `EMAIL`, `PERSON_NAME`, `DATE`

● Mark **PII = false** for:  
`CITY`, `LOCATION`

● Output entity spans as **character offsets** on the original transcript.

---
Generate your own train (500-1000 examples) and dev (100-200) sets reflecting noisy
STT patterns. 

Few-shot examples



{"id": "utt_0001", "text": "my credit card number is 4242 4242 4242 4242 and my email is ramesh dot sharma at gmail dot com", "entities": [{"start": 26, "end": 49, "label": "CREDIT_CARD"}, {"start": 66, "end": 80, "label": "PERSON_NAME"}, {"start": 84, "end": 104, "label": "EMAIL"}]}
{"id": "utt_0002", "text": "call me on nine eight seven six five four three two one zero i live in chennai", "entities": [{"start": 11, "end": 63, "label": "PHONE"}, {"start": 75, "end": 82, "label": "CITY"}]}

**Response**

❓ Before I generate the dataset, choose:
1. Dataset size

Pick one:

A) 500 train + 100 dev
B) 800 train + 150 dev
C) 1000 train + 200 dev
D) Custom (tell me exact numbers)
2. Noise level for transcripts

Choose one:

A) Mild noise

"gmail dot com", "at the rate", “six nine zero”

lowercase, missing punctuation

common ASR mistakes: “to / too / two”, “their / there”

B) Medium noise

mis-hearings: “jawn” for "john", “september twenty first” → "september twenty first"

numbers in words

filler words, hesitations

dropped small words

C) Heavy noise

phonetic spellings (“phoen ex” for "phoenix")

extra words, repeated words

wrong word boundaries

distorted email syntax ("g mail dot calm")

mixed number formats (digits + words)


Then I will generate:

✔ Fully valid JSONL
✔ Exact character offsets
✔ No hallucinated entities
✔ 100% entity-label compliance:

PII = CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE

non-PII = CITY, LOCATION
✔ Natural ASR/STT-like noisy text
✔ Separate train.jsonl and dev.jsonl files

**Prompt**
Size B Noise B

## Train
A different model was chosen because it best performs in this job
```bash
python src/train.py \
  --model_name microsoft/xtremedistil-l6-h384-uncased \
  --train data/train.jsonl --dev data/dev.jsonl \
  --out_dir out
```

## Predict

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

## Evaluate

```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

## Results


**Training**

Some weights of BertForTokenClassification were not initialized from the model checkpoint at 
microsoft/xtremedistil-l6-h384-uncased and are newly initialized:
['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

Epoch 1/50: ... average loss: 2.3778  
Epoch 2/50: ... average loss: 1.5500  
Epoch 3/50: ... average loss: 1.3777  
...
Epoch 50/50: ... average loss: 0.0079

Saved model + tokenizer to out


**Per-entity metrics:**

| Entity / Group | Precision | Recall | F1 Score |
|----------------|-----------|--------|----------|
| CITY           | 0.997     | 1.000  | 0.998    |
| CREDIT_CARD    | 0.000     | 0.000  | 0.000    |
| DATE           | 0.993     | 1.000  | 0.997    |
| EMAIL          | 0.150     | 0.270  | 0.193    |
| LOCATION       | 1.000     | 1.000  | 1.000    |
| PERSON_NAME    | 0.587     | 0.560  | 0.573    |
| PHONE          | 0.371     | 0.783  | 0.504    |
| **PII (overall)**     | **0.586** | **0.723** | **0.647** |
| **Non-PII (overall)** | **0.998** | **1.000** | **0.999** |



Macro-F1: 0.609


**Latency over 50 runs (batch_size=1):**

  p50: 1.30 ms
  
  p95: 1.38 ms




