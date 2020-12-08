# Work Safely with Free Text Using Gretel

Using Gretel.ai's [NER and NLP features](https://gretel.ai/platform/data-cataloghttps://gretel.ai/platform/data-catalog), we analyze and label a set of email dumps looking for PII and other potentially sensitive information. After labeling the dataset, we build a transformation pipeline that will redact and replace any sensitive strings from the email messages.

At the end of the notebook we'll have a dataset that is safe to share and analyze without compromising a user's personal information.