# Anonymize production data for use in dev environments

Data seeded in development, test and other pre-production environments often don't have parity with production data. This difference in quality can make it difficult to track down bugs during development, and often leads to bugs that only occur in production.

In this blueprint, we take a production dataset containing sensitive, personally identifying details and generate a fake, anonymized copy of that dataset. The resulting dataset has the same shape, and can be loaded into pre-production databases, but isn't identifiable back to any customer.

Using Gretel's [Data Catalog](https://gretel.ai/platform/data-catalog) and [Transformation](https://gretel.ai/platform/transform) tools we walk-through a notebook that analyzes a source dataset and automatically generate a data pipeline that will transform a production dataset. While this demonstration ran as a notebook, this same pipeline can be deployed into a variety of different data stacks.

## Steps

1. Click "Transform" from the project's navigation bar.
1. Copy the Project URI key from the Console.
1. Select the "Anonymize production data for use in dev environments" blueprint. This will launch a Jupyter notebook in Google Colab.
1. Follow the steps outlined in the notebook to generate a fake, anonymized version of this project's dataset.
