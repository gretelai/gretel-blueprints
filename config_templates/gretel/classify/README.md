# Gretel Classify Configuration Templates

Templates in this directory contain configurations to classify data with a variety of characteristics and to enforce various levels of privacy protections.

Templates can be downloaded and modified for use with Gretel classify APIs, or imported directly from the Gretel CLI via:

`gretel models create --config classify/[template_name] ...`


 | template_name      | description |
 | ----------- |  ----------- |
 |`classify-pii-nlp-on`| This template contains a simple policy to identify commonly found personally identifiable information (PII). NLP is turned on to improve detection of person name and location data in unstructured text fields. This is the default config file provided in the Gretel console.|
 |`classify-pii-nlp-off`| This template contains the same policy as `classify-pii-nlp-on` but has NLP turned off to improve performance. Use this template for detecting common PII fields in large datasets.|
 |`classify-pii-regex`| An example of using label_predictors and regular expressions to define custom predictors.|
 |`classify-all-entities`| Label all supported info types in your data. Warning! Performance could be slow for this model, depending on the amount of data being processed.|
