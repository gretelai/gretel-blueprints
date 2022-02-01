# Gretel Transforms Configuration Templates

Templates in this directory contain configurations to transform data with a variety of characteristics and to enforce various levels of privacy protections.

Templates can be downloaded and modified for use with Gretel transform APIs, or imported directly from the Gretel CLI via:

`gretel models create --config transform/[template_name] ...`


 | template_name      | description |
 | ----------- |  ----------- |
 |`default`| This template contains a simple policy to replace personally identifiable information (PII) with a fake value or redact with a pre-defined character. NLP is turned on to improve detection of person names and locations. This is the default config file provided in the Gretel console.|
 |`redact-pii-nlp-off`| This template contains the same policy as `redact-pii-nlp-on` (from `default` template), but has NLP turned off to improve performance. Use this template for de-identifying common PII fields in large datasets.|
 |`redact-pii-regex`| An example of using label_predictors and regular expressions to define custom predictors.|