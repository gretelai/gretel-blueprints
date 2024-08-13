![Generate synthetic tabular, text and time series data](https://blueprints.gretel.cloud/use_cases/images/navigator-ft-hero.png "Generate synthetic tabular, text and time series data")

**Navigator Fine Tuning** is the latest advancement in our suite of synthetic data solutions. It builds upon the recent general availability of [Gretel Navigator](https://console.gretel.ai/navigator), enabling you to generate data not only from a prompt, but also from fine-tuning the underlying model on your domain-specific real-world datasets to generate the highest quality synthetic data.

One of the standout features of Navigator Fine Tuning is its support for multiple tabular data modalities within a single model. This means you can now generate datasets that maintain correlations across:
- Numeric Data: Continuous or discrete numbers
- Categorical Data: Categories or labels
- Free Text: Unstructured text entries and long-form natural language such as email messages or notes in medical treatment summaries
- Time Series: Sequential time-stamped data
- JSON Data: Complex nested structures

All these data types can coexist within a single dataset, maintaining correlations not just within individual rows, but also across events spanning multiple rows, making Navigator  an exceptionally powerful tool for time series data generation.

Try it in the Console, or if you prefer code, give the [SDK notebook](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/demo/navigator-fine-tuning-intro-tutorial.ipynb) a spin. 
