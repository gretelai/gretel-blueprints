![Generate text to code datasets for training coding assistants](https://blueprints.gretel.cloud/use_cases/images/data-designer.png "Generate text to code datasets for training coding assistants")

Gretel Data Designer helps you create high-quality synthetic datasets that pair natural language instructions with corresponding code implementations. These instruction-code pairs are essential for training and fine-tuning coding assistants that can accurately translate user requests into executable code. With Data Designer, you can generate thousands of diverse, domain-specific examples without the manual effort typically required.

This blueprint showcases how to create synthetic datasets for code generation in both Python and SQL contexts. You'll learn how to control complexity levels, specify industry domains, and incorporate relevant programming concepts—ensuring your training data covers the full spectrum of scenarios your model needs to handle. Each generated example includes natural language instructions, relevant context (like database schemas for SQL), and properly validated code solutions that follow best practices. The notebooks also demonstrate built-in validation and evaluation to ensure code quality and correctness.

Whether you're building a general-purpose coding assistant or domain-specific tools for healthcare, finance, or technology sectors, these blueprints provide a flexible framework for generating the high-quality training data needed to improve your models' ability to translate natural language into working code.

[Check out our blog post here](https://gretel.ai/blog/build-high-quality-datasets-for-ai-using-gretel-navigator) or [read the documentation](https://docs.gretel.ai/create-synthetic-data/gretel-data-designer-beta) to learn more.

Interested in other use cases? Here are a few more notebooks:
- [RAG evaluation](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/data-designer/rag-examples/generate-rag-evaluation-dataset.ipynb)
- [Start from seed data](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/data-designer/data-designer-101/3-seeding-with-a-dataset.ipynb)
- [Multi-turn conversation datasets](https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/data-designer/multi-turn-chat/multi-turn-conversation.ipynb)