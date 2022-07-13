# Gretel Synthetics Configuration Templates

The templates in this directory contain configurations to train synthetic models on data with a variety of characteristics and to enforce various levels of privacy protections.

Templates can be downloaded and modified for use with Gretel synthetics, or imported directly from the Gretel CLI via:

`gretel models create --config synthetics/[template_name] ...`


 | template_name      | description |
 | ----------- |  ----------- |
 |`default`| Use the `gretel-synthetics` LSTM model. This model works for a variety of synthetic data tasks including time-series, tabular, and text data. Generally useful for a few thousand records and upward. Dataset generally has a mix of categorical, continuous, and numerical values.  |
 |`complex-or-free-text` | Best for highly complex datasets, or those containing plain text such as tweets and conversations.  |
 |`differential-privacy` | Differential privacy helps prevent unintended memorization of secrets in the training data, by limiting the amount that any training example, or small set of training examples, can affect the model.|
 |`high-dimensionality`| Use for datasets with more than 20 columns and/or 50,000 rows. Works well for largely numeric data. Avoid if dataset contains free text fields.  |
 |`high-accuracy`| Useful for optimizing data for downstream ML tasks, at possible cost of higher compute. |
 |`low-record-count`| For datasets that have fewer than 1000 rows and/or 4 columns, and a mix of categorical, numerical, and continuous values.  |
