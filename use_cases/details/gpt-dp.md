![Create free text data with privacy guarantees](https://blueprints.gretel.cloud/use_cases/images/gpt-dp.png "Create free text data with privacy guarantees")

Unlock the potential of your text data while ensuring privacy by applying [differentially private fine-tuning using Text Fine-Tuning](https://gretel.ai/blog/generate-differentially-private-synthetic-text-with-gretel-gpt). This method allows you to create a version of your free text data that maintains the integrity of sensitive information while still providing high-quality outputs.

We recommend having a dataset of at least 10,000 samples to ensure reasonable quality. Note that differential privacy requires more epochs, which leads to longer training times compared to running without differential privacy.

Prefer coding? Check out the [SDK notebook](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/generate_differentially_private_synthetic_text.ipynb) example.
