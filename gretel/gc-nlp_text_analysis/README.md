# Working Safely with Sensitive Free Text Using Gretel Cloud and NLP

Using Gretel.ai's [NER and NLP features](https://gretel.ai/platform/data-cataloghttps://gretel.ai/platform/data-catalog), we analyze and label chat logs looking for PII and other potentially sensitive information. After labeling the dataset, we build a transformation pipeline that will redact and replace any sensitive strings from chat messages.

At the end of the notebook we'll have a dataset that is safe to share without compromising a user's personal information.