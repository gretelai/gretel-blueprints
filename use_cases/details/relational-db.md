![Generate relational data](https://blueprints.gretel.cloud/use_cases/images/relational-data-generation.png "Generate multi-table synthetic data")

Gretel Relational takes you beyond synthesizing and transforming one table at a time, and enables generating entire databases, all while maintaining referential integrity. This early preview will give you a taste of what's coming in the Gretel Console and CLI/SDK. [We'd love to hear your feedback!](https://dqq4jigtkl1.typeform.com/to/Gibb8awJ)

To connect to your remote database, just 

### Notebook Details

In the notebooks below, we've created two practical examples using a telecommunications database and Gretel Relational. See the database schema below.

![telecom db diagram](https://blueprints.gretel.cloud/use_cases/images/telecom-db-small.png "Telecom Database Diagram")

In the first example, we'll synthesize all 5 tables in the database, while in the second, we will transform the data to remove PII (Personally Identifiable Information). You can also transform and then synthesize a dataset; stay tuned for a blog post and notebook on that soon.

We'll demonstrate how easy it is to connect to a database, run Gretel Relational, view quality metrics, and output the data back into a database. We will also test that referential integrity exists both before and after data is synthesized/transformed.

[Read our blog](https://gretel.ai/blog/generate-synthetic-databases-with-gretel-relational) to learn more about Gretel Relational, check out [our docs](https://docs.gretel.ai/reference/relational), or use the colab notebooks below to see it in action. 
