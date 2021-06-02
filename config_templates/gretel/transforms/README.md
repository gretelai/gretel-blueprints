# Gretel Transforms Configuration Templates

The templates in this directory can be used for a variety of potential use-cases.  Here is a short outline of the potential options:
* `value-level-pii`: this config is useful for working with unstructured data, for example one containing free-form text, etc.
  * **NOTE** This kind of transform requires labeling of each record, so it will be slower than field-level transform only.
* `field-level-pii`: this config is useful for working with structured data, where each field contains the same type of information.
  * It's much faster than value-level one, because it transforms each record without labeling it.
* `chaining-policies-example`: this is an example of how to chain policies to create more complex transforms.

# Backwards Compatability with Gretel Beta SDKs

TODO