# Gretel Synthetics Configuration Templates

The templates in this directory contain configurations to train synthetic models on data with a variety of characteristics and to enforce various levels of privacy protections.

Templates can be downloaded and modified for use with Gretel synthetics, or imported directly from the Gretel CLI via:

`gretel models create --config synthetics/[template_name] ...`


 | template_name      | description |
 | ----------- |  ----------- |
 |`default`| Use the `gretel-synthetics` LSTM model. This model works for a variety of synthetic data tasks including time-series, tabular, and text data. Generally useful for a few thousand records and upward. Dataset generally has a mix of categorical, continuous, and numerical values.  |
 |`complex-or-free-text` | Best for highly complex datasets, or those containing plain text such as tweets and conversations.  |
 |`high-dimensionality`| Use for datasets with more than 20 columns and/or 50,000 rows. Works well for largely numeric data. Avoid if dataset contains free text fields.  |
 |`high-dimensionality-high-record-count`| Use for datasets with more than 250,000 rows and 20 columns. Works well for largely numeric data. Avoid if dataset contains free text fields.  |
 |`high-accuracy`| Useful for optimizing data for downstream ML tasks, at possible cost of higher compute. |
 |`low-record-count`| For datasets that have fewer than 1000 rows and/or 4 columns, and a mix of categorical, numerical, and continuous values.  |
 |`time-series`| Specialized model for time-series datasets. Modify this config with values that fit your dataset.  |
 |`natural-language`| Useful for single-column natural language datasets such as reviews, tweets, and conversations. Dataset must be single-column. | 
 |`tabular-lstm-evaluate`| Use this blueprint for generating and evaluating synthetic data performance on ML models (classification/regression). You can change the configuration to use the best synthetic model for your use case. | 