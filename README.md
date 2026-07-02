# NLP Sentiment Analysis with Hugging Face
ProSensia AI/ML Bootcamp | Week 3 Day 4

## what i did
used a pre-trained distilbert model from hugging face to classify 1000 customer reviews as POSITIVE or NEGATIVE. cleaned raw messy text with regex first then ran inference.

## files
- nlp_sentiment_wrapper.ipynb
- customer_reviews_raw.csv
- reviews_with_sentiment.csv
- requirements.txt
- README.md

## how to run
pip install -r requirements.txt
jupyter notebook nlp_sentiment_wrapper.ipynb

## model used
distilbert-base-uncased-finetuned-sst-2-english
- pre-trained on SST-2 sentiment dataset
- no fine-tuning done, used as-is
- runs on cpu (device=-1)

## text cleaning steps
1. remove html tags with regex
2. remove html entities like &amp;
3. remove non-ascii characters (emojis etc)
4. remove extra whitespace
5. lowercase and strip

## results
- total reviews: 1000
- positive: 518
- negative: 482
- avg confidence: 0.8785

## why pretrained > from scratch
training a sentiment model from scratch needs:
- millions of labeled examples
- days/weeks of gpu training
- huge cost
distilbert already has this knowledge, we just call pipeline() and done
