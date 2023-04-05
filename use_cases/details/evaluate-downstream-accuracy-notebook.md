Customers frequently ask whether synthetic data is of high enough quality to train downstream machine learning tasks. Classification and regression models, for example, require highly accurate data before they can be usefully deployed. 

In these notebooks, learn how to use Gretel Evaluate tasks to validate synthetic data utility on classification and regression models. 

### Try it in the Gretel Console
Select the Gretel Synthetic model you want to train and generate synthetic data (the default is LSTM which generally works well for any tabular dataset), and under the Evaluate options, add the name of the `target` (or label) column of the dataset. Check out the ["Synthesize data + evaluate ML performance"](https://console.gretel.ai/use_cases/cards/use-case-downstream-accuracy/projects) flow to get started.

### Notebooks

For more advanced examples, these notebooks use Gretel's synthetic model to generate synthetic data from publicly available datasets and then verifies its quality when training downstream models using Gretel’s Evaluate model, which consumes the open-source AutoML PyCaret library for both classification and regression.

[Read our blog post](https://gretel.ai/blog/downstream-ml-evaluation-and-quality-scores) or try out the notebooks for yourself, either by [downloading them from Github](https://github.com/gretelai/gretel-blueprints/tree/main/docs/notebooks) or using one of the Google Colab links below.
