# Gretel Synthetics Configuration Templates

The templates in this directory can be used for a variety training data characteristics.  Here is a short outline of the potential options:

 - `default`: General purpose configuration, which uses our default settings from Gretel's open source package. Works for a variety of datasets. Generally useful for a few thousand records and upward. Dataset generally has a mix of categorical, continuous, and numerical values.
 - `low-record-count`: For datasets that have record counts in the hundreds and a mix of categorical, numerical, and continuous values.
 - `high-record-count`: When working with datasets with extremely high record counts, potentially in tens of millions and above. Records still have a mix of data types.
 - `numeric-data`: When working with data that is mostly numerical (integers, floating point data, etc). Typical examples are time series data.
   - **NOTE**: If any data is one-hot encoded, we recommend encoding those fields back to a single field then splitting it back into one-hot encoded fields after generation is done.
   - **NOTE**: For floating point numbers, we recommend reducing the precision to as low as it can go without disrupting the downstream use case.
- `high-field-count`: For highly dimensional or sparse data with many fields / columns.
  - **NOTE**: We recommend removing columns that are unnecessary for the downstream use case (such as ML modeling, etc) and also applying consolidation of one-hot encoded fields like mentioned above.
- `complex-or-free-text`: If you have higly complex data such as free text data, chat logs, etc.
  - **NOTE**: For some free-text scenarios, you also may want to set `reset_states: False`


# Backwards Compatability with Gretel Beta SDKs

To make use of these templates with our Beta Python SDKs, you may use the values in the `param` objects as your configuration dictionary that gets passed into the main synthetics class.  Additionally, you may utilize a helper we have in our `gretel-client` to fetch the entire dictionary you would need:

```python
from gretel_client import get_synthetics_config 
    
config_template = get_synthetics_config("low-record-count")

# modify config_template dict if need be

model = SyntheticDataBundle(
    synthetic_config=config_template,
    # ...
)
```