Gretel Relational takes you beyond synthesizing and transforming one table at a time, and enables generating entire databases, all while maintaining referential integrity. We support connections to several relational databases such as MS SQL Server, Snowflake, BigQuery, Oracle, MySQL, PostgreSQL, with more connectors coming soon. 

To get started in the Console, go to "Workflows" in the sidebar, and select the "New Workflow" button. Choose your project and the model you want to use (ACGTAN for synthesizing or Transform v2 for redacting PII). Next, use the Connetion Creation 
Wizard to create a connection to your remote database. And then follow the easy steps to finish creating your workflow. 

Want to see how Gretel Relational works? Check out one of our notebooks below to go through the process step by step and compare the data before and after synthesis.

To connect to your remote database, just 

### Notebook Details

In the notebooks below, we've created two practical examples using a telecommunications database and Gretel Relational. See the database schema below.

![telecom db diagram](https://blueprints.gretel.cloud/use_cases/images/telecom-db-small.png "Telecom Database Diagram")

In the first example, we'll synthesize all 5 tables in the database, while in the second, we will transform the data to remove PII (Personally Identifiable Information). You can also transform and then synthesize a dataset; stay tuned for a blog post and notebook on that soon.

We'll demonstrate how easy it is to connect to a database, run Gretel Relational, view quality metrics, and output the data back into a database. We will also test that referential integrity exists both before and after data is synthesized/transformed.

[Read our blog](https://gretel.ai/blog/generate-synthetic-databases-with-gretel-relational) to learn more about Gretel Relational, check out [our docs](https://docs.gretel.ai/reference/relational), or use the colab notebooks below to see it in action. 
