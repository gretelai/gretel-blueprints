A frequent question customers ask is whether synthetic data is of high enough quality to train downstream machine learning tasks. Classifiers, for example, require highly accurate data before they can be usefully deployed. 

The term "downstream" has multiple connotations. It could refer to a step in the system that happens after data has been processed and transformed. It can also refer to the machine learning step itself. 

### Notebook Details
This notebook uses Gretel's ACTGAN model to generate synthetic data from a highly dimensional dataset and then verifies its quality when training a downstream classifier using the open source AutoML PyCaret library.

[Read our blog post](https://gretel.ai/blog/downstream-ml-classification-with-gretel-actgan-and-pycaret) or run the notebook to try it out yourself.