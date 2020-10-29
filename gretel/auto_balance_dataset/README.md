# Auto-Balance Dataset

In this blueprint, we will use Gretel-Synthetics to produce a balanced, privacy preserving version of your dataset. This blueprint can be used to support fair AI as well as generally any imbalanced dataset. Information Systems (IS) utilizing Artificial Intelligence (AI) are now ubiquitous in our culture. They are often responsible for critical decisions such as who to hire and at what salary, who to give a loan or insurance policy to, and who is at risk for cancer or heart decease. Fair AI strives to eliminate IS discrimination against demographic groups. This blueprint can help you achieve fair AI by eliminating the bias in your data. All it takes is one pass through the data, and bias will be completely removed from as many fields as you like. Correlations and distributions in non-bias fields will, as always, transfer from your training data to your synthetic data.


## Objective
In this blueprint, we will remove bias by training a generative synthetic data model to create a balanced dataset. The blueprint supports two different modes for balancing your data. The first (mode="full"), is the scenario where you'd like to generate a complete synthetic dataset with bias removed. The second (mode="additive"), is the scenario where you only want to generate synthetic samples, such that when added to the original set will remove bias. The blueprint takes data from an existing Gretel project and first shows graphically the field distributions to help you narrow in on fields containing bias. After choosing the fields you would like to balance, a synthetic data model is then trained and data is generated as needed to balance your dataset. At the conclusion of the blueprint, the option is given to generate a full Synthetic Performance Report. An example report created after balancing the columns "gender", "race" and "income_bracket" in the Kaggle US Adult Income dataset is located [here](https://gretel-public-website.s3-us-west-2.amazonaws.com/blueprints/data_balancing/Auto_Balance_Performance_Report.html).


## Steps
1. Click "Transform" on the project NavBar.
2. Copy your Project URI key from the Console.
3. Select the "Auto-Balance Dataset" notebook.
4. If using Colab, click Runtime->Change project runtime and change to "GPU".
5. Step through the notebook cells, answering questions and adjusting params as needed.
6. Once your new data is generated, choose either the cell to save to a CSV or to a new Gretel project.
7. Finally, if desired, generate a full Synthetic Performance Report.
