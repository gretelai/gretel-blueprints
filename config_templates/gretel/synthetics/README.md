# Gretel Synthetics Configuration Templates

The templates in this directory can be used for a variety training data characteristics.  Here is a short outline of the potential options:


 | Template name      | Description |
 | ----------- |  ----------- |
 |`default`| General purpose configuration, which uses our default settings from Gretel's open source package. Works for a variety of datasets. Generally useful for a few thousand records and upward. Dataset generally has a mix of categorical, continuous, and numerical values.      |
 |`complex-or-free-text` | Best for large and highly complex datasets, or those  containing plain text such as tweets and conversations.
 |`differential-privacy` | Differential privacy helps prevent unintended memorization of secrets in the training data, by limiting the amount that any training example, or small set of training examples can affect the model.|
 |`high-record-count`| When working with datasets with extremely high record counts, potentially in tens of millions and above. Records still have a mix of data types.|
 |`low-record-count`| For datasets that have record counts in the hundreds and a mix of categorical, numerical, and continuous values.    |
 |`numeric-data`| When working with data that is mostly numeric (integers, floating point data, etc). Typical examples are time series, financial data, and GPS coordinates.|
