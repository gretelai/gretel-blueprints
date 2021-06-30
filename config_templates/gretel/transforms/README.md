# Gretel Transforms Configuration Templates

Templates in this directory contain configurations to transform data with a variety of characteristics and to enforce various levels of privacy protections.

Templates can be downloaded and modified for use with Gretel transform APIs, or imported directly from the Gretel CLI via:

`gretel models create --config transforms/[template_name] ...`


 | template_name      | description |
 | ----------- |  ----------- |
 |`redact-sensitive-pii`| This template contains a simple policy to replace personally identifiable information (PII) with a fake value or secure hash.|
