Imbalanced datasets are a common problem in machine learning. For example, if the training data has an imbalanced demographic, the resulting AI models can exhibit representational biases.

Conditional data generation (sometimes called seeding or prompting) is a technique where a generative model is instructed to generate data according to some pre-specified conditioning--such as a topic, sentiment, or one or more field values in a tabular dataset--to balance out the biases.

### Notebook Details

This notebook utilizes a new model that allows you to take partial values from your training dataset and use them as input for synthesis. You do this by specifying one or more seed columns. For instance, if you have data with 5,000 rows, and choose columns A, B, C, the model will be trained with those columns as "smart seeds."

See more [smart seeding](https://gretel.ai/blog/gretel-smart-seeding-is-auto-complete-for-your-data) and [conditional data generation](https://gretel.ai/blog/conditional-data-generation-in-4-lines-of-code) examples on our blog.
