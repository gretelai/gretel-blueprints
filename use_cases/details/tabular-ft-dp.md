![Generate differentially-private synthetic numeric, categorical, and text data](https://blueprints.gretel.cloud/use_cases/images/tabular-ft-dp.png "Generate differentially-private synthetic numeric, categorical, and text data")

With Differential Privacy for Tabular Fine-Tuning, you can now get mathematical guarantees related to privacy without having to separate your dataset out by data type. Tabular Fine-Tuning supports datasets containing numerical, categorical, free text, and event-driven fields.

Fine-tuning with differential privacy can be done on the record level, or in the case of event-driven data, on the group level.

When Tabular Fine-Tuning is run with differential privacy, you can expect it to take roughly twice as long to train as when run without. We recommend your dataset have at least a few thousand records to ensure that the model is still able to learn about the dataset without learning too much about individual records.
