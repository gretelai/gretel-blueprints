![RAG Evaluation with Synthetic Data](https://blueprints.gretel.cloud/use_cases/images/data-designer.png "RAG Evaluation with Synthetic Data")

Create tailored evaluation datasets for your Retrieval-Augmented Generation systems with Gretel Data Designer. This blueprint helps you generate domain-specific reference documents, queries, and ground truth answers that match your real-world use cases and evaluation needs.

Unlike generic benchmarks, these custom datasets let you test how your RAG system handles your specific information domains, question types, and edge cases. Control document complexity, create multi-hop reasoning challenges, and measure performance across retrieval accuracy and answer quality metrics. Generate everything from simple factual queries to complex scenarios requiring information synthesis across multiple documents.

[Check out our blog post here](https://gretel.ai/blog/build-high-quality-datasets-for-ai-using-gretel-navigator) or [read the documentation](https://docs.gretel.ai/create-synthetic-data/gretel-data-designer-beta) to learn more.

Here are some more notebooks if you'd like to try out other use cases:
- [Multi-turn conversation datasets](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/demo/navigator/multi-turn-chat/navigator-data-designer-sdk-multi-turn-conversation.ipynb)
- [Start from seed data](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/demo/navigator/navigator-data-designer-sdk-sample-to-dataset.ipynb)
- [Text to SQL pairs](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/demo/navigator/text-to-code/navigator-data-designer-sdk-text-to-sql.ipynb)