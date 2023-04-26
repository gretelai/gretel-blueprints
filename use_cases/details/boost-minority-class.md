![boost minority class banner](https://blueprints.gretel.cloud/use_cases/images/boost-minority-class-hero.png "Boost Minority Class Banner")

Real world datasets can often be sparse, meaning they contain limited instances from a particular target class. Training downstream ML models on such datasets is challenging since the model struggles to accurately learn the boundaries between majority and minority class samples.

### Notebook Details

This notebook utilizes Gretel ACTGAN to train a model on a dataset of [credit card transactions from September 2013]("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset only contains a few instances of the positive class (fraudulent transactions). Once the model has been trained, the notebook conditionally generates additional samples of the minority class that can be used to augment the original dataset.

The notebook also provides Gretel Data Quality (SQS) metrics and visualizes the samples. Note that the synthetic samples are generated with Gretel's Privacy Filters and therefore provide privacy protection compared to other resampling techniques.

*Related*: Read about [smart seeding](https://gretel.ai/blog/gretel-smart-seeding-is-auto-complete-for-your-data) and [conditional data generation](https://gretel.ai/blog/conditional-data-generation-in-4-lines-of-code) on our blog.