![Generate differentially-private synthetic numeric, categorical, and text data](https://blueprints.gretel.cloud/use_cases/images/navigator-ft-dp-hero.png "Generate differentially-private synthetic numeric, categorical, and text data")

With Differential Privacy for Navigator Fine Tuning, you can now get mathematical guarantees related to privacy without having to separate your dataset out by data type. Navigator Fine Tuning supports datasets containing numerical, categorical, free text, and event-driven fields.

Fine-tuning with differential privacy can be done on the record level, or in the case of event-driven data, on the group level. You can read more about how differential privacy works in our [blog](https://gretel.ai/blog/generate-complex-synthetic-tabular-data-with-navigator-fine-tuning-differential-privacy).

You can try it out below, or in the [SDK](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/demo/navft-dp/navft-dp-sample.ipynb). For an advanced example using event-driven data, check out this [notebook](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/demo/navft-dp/navft-dp-experiments.ipynb).

When Navigator Fine Tuning is run with differential privacy, you can expect it to take roughly twice as long to train as when run without. We recommend your dataset have at least a few thousand records to ensure that the model is still able to learn about the dataset without learning too much about individual records.