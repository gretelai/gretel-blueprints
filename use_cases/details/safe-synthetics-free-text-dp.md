![Redact & synthesize free text with maximum privacy](https://blueprints.gretel.cloud/use_cases/images/text-ft-dp.png "Redact & synthesize free text with maximum privacy")

This blueprint helps you create a synthetic version of your free text dataset with maximum privacy.

First, the blueprint helps redact and replace any personally identifiable information (PII) to ensure the model has no chance to learn it. Then, it runs Text Fine-Tuning to generate synthetic records, obfuscating quasi-identifiers like gender and age. While synthesizing, differential privacy is applied in order to achieve mathematical guarantees of privacy.

You can try using the sample dataset, or upload your own.

Prefer coding? Check out the [SDK notebook](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/safe-synthetics/free-text-transform-synthesize-dp.ipynb) example.