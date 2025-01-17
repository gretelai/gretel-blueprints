import json
import time
from contextlib import contextmanager

import numpy as np
import plotly.graph_objects as go
from transformers import Pipeline
from tqdm.auto import tqdm


@contextmanager
def measure_time():
    """Context manager for measuring execution time."""
    start_time = time.time()
    yield
    total_time = time.time() - start_time
    print(f"Total time spent in generation: {total_time:.2f} seconds")


def prepare_prompt(prompt, pipe):
    """Prepare a prompt for the pipeline."""
    return pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def test_inference_batch(
    prompts,
    pipe: Pipeline,
    max_new_tokens=1024,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    eos_token_id=None,
):
    """Generate text for a batch of prompts using the pipeline."""
    prepared_prompts = []
    for prompt in prompts:
        prepared_prompt = pipe.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prepared_prompts.append(prepared_prompt)

    outputs = pipe(
        prepared_prompts,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )
    generated_texts = [
        output[0]["generated_text"][len(prepared_prompts[i]) :].strip()
        for i, output in enumerate(outputs)
    ]
    return generated_texts


def process_sample(document_text, pii_spans, pipe: Pipeline):
    """Process a sample document and identify found and missed PII spans."""
    generated_text = test_inference_batch([document_text], pipe)[0]
    found_spans = []
    missed_spans = []

    for span in pii_spans:
        start, end, pii_type = span
        pii_value = document_text[start:end]
        expected_label_position = generated_text.find(pii_value)
        is_labeled_correctly = expected_label_position != -1 and generated_text[
            expected_label_position + len(pii_value) :
        ].startswith("}")

        if is_labeled_correctly:
            found_spans.append(span)
        else:
            missed_spans.append(span)

    return found_spans, missed_spans


def calculate_pii_accuracy(pii_type_counters, found_spans, missed_spans):
    """Calculate PII accuracy counters for found and missed spans."""
    for span in found_spans:
        pii_type = span[2]
        if pii_type not in pii_type_counters:
            pii_type_counters[pii_type] = {"found": 0, "missed": 0}
        pii_type_counters[pii_type]["found"] += 1

    for span in missed_spans:
        pii_type = span[2]
        if pii_type not in pii_type_counters:
            pii_type_counters[pii_type] = {"found": 0, "missed": 0}
        pii_type_counters[pii_type]["missed"] += 1

    return pii_type_counters


def plot_found_vs_missed(pii_accuracy_counters, model_name):
    """Plot a bar chart comparing found and missed counts for each category."""
    categories = list(pii_accuracy_counters.keys())
    found_counts = [pii_accuracy_counters[cat]["found"] for cat in categories]
    missed_counts = [pii_accuracy_counters[cat]["missed"] for cat in categories]

    total_found = sum(found_counts)
    total_missed = sum(missed_counts)
    total_elements = total_found + total_missed
    overall_accuracy = (total_found / total_elements) * 100 if total_elements > 0 else 0

    title_text = (
        f"Found vs. Missed for Each Category (Model: {model_name}) - "
        f"Overall Accuracy: {overall_accuracy:.2f}%, "
        f"Total PII Elements: {total_elements}"
    )

    fig = go.Figure(
        data=[
            go.Bar(name="Found", x=categories, y=found_counts, marker_color="green"),
            go.Bar(name="Missed", x=categories, y=missed_counts, marker_color="red"),
        ]
    )

    fig.update_layout(
        title=title_text,
        xaxis_title="Category",
        yaxis_title="Count",
        barmode="group",
        legend=dict(x=0.01, y=0.99),
    )

    fig.show()


def plot_detection_percentages_heatmap(detection_percentages, model_name):
    """Plot a heatmap of detection percentages by document type and PII type."""
    doc_types = list(detection_percentages.keys())
    pii_types = list(
        {
            pii_type
            for doc_type in detection_percentages.keys()
            for pii_type in detection_percentages[doc_type].keys()
        }
    )

    total_percentage = 0
    count = 0

    for doc_type in detection_percentages:
        for pii_type in detection_percentages[doc_type]:
            value = detection_percentages[doc_type][pii_type]
            if value is not None:
                total_percentage += min(value, 100.0)
                count += 1
            detection_percentages[doc_type][pii_type] = (
                np.nan if value is None else min(value, 100.0)
            )

    z_values = [
        [
            detection_percentages.get(doc_type, {}).get(pii_type, np.nan)
            for pii_type in pii_types
        ]
        for doc_type in doc_types
    ]

    overall_accuracy = total_percentage / count if count > 0 else 0

    custom_colorscale = [
        [0.00, "#440154"],
        [0.25, "#3b528b"],
        [0.5, "#21918c"],
        [0.75, "#5ec962"],
        [1.0, "#fde725"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=pii_types,
            y=doc_types,
            colorscale=custom_colorscale,
            colorbar=dict(title="Percentage"),
        )
    )

    title_text = f"Detection Percentages by Document Type and PII Type - Model: {model_name}, Overall Accuracy: {overall_accuracy:.2f}%"
    fig.update_layout(
        title=title_text, xaxis_title="PII Type", yaxis_title="Document Type"
    )

    fig.show()


def evaluate_pii_labeling_accuracy(dataset, pipe: Pipeline, num_samples=None):
    """
    Evaluates the accuracy of PII (Personally Identifiable Information) labeling performed by a given pipeline on a specified dataset.

    The function processes a subset of the dataset (specified by `num_samples`) to determine the accuracy of detected PII spans against known spans. It calculates overall accuracy, detection rates by PII type, and logs instances of missed detections.

    Args:
    - dataset (iterable): The dataset containing documents, where each document includes text and labeled PII spans.
    - pipe (Pipeline): The PII detection pipeline used to evaluate PII spans in the text.
    - num_samples (int, optional): The number of samples to evaluate from the dataset. If None, the entire dataset is used.

    Returns:
    - tuple: A tuple containing:
        - overall_accuracy (float): The overall accuracy of PII detection across all samples.
        - pii_type_counters (dict): A dictionary mapping each PII type to the number of correct detections and misses.
        - doc_type_pii_type_counters (dict): A dictionary categorizing accuracy data by document type and PII type.
        - detection_percentages (dict): Detection rates by document type and PII type.
        - missed_detections (list): A list of dictionaries, each describing a missed detection incident.
    """

    pii_type_counters = {}
    doc_type_pii_type_counters = {}
    detection_percentages = {}
    missed_detections = []

    num_samples = num_samples or len(dataset)
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    total_found = 0
    total_missed = 0

    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        document_text = sample["document_text"]
        doc_type = sample["document_type"]
        pii_spans = json.loads(sample["pii_spans"])

        found_spans, missed_spans = process_sample(document_text, pii_spans, pipe)
        total_found += len(found_spans)
        total_missed += len(missed_spans)

        pii_type_counters = calculate_pii_accuracy(
            pii_type_counters, found_spans, missed_spans
        )

        for span in missed_spans:
            start, end, pii_type = span
            pii_value = document_text[start:end]
            missed_detections.append(
                {
                    "sample_index": i,
                    "doc_type": doc_type,
                    "pii_type": pii_type,
                    "start": start,
                    "end": end,
                    "pii_value": pii_value,
                    "document_text": document_text,
                }
            )

        doc_type_pii_type_counters.setdefault(doc_type, {})
        for pii_type in set(span[2] for span in found_spans + missed_spans):
            doc_type_pii_type_counters[doc_type].setdefault(
                pii_type, {"found": 0, "missed": 0}
            )
            doc_type_pii_type_counters[doc_type][pii_type]["found"] += len(
                [s for s in found_spans if s[2] == pii_type]
            )
            doc_type_pii_type_counters[doc_type][pii_type]["missed"] += len(
                [s for s in missed_spans if s[2] == pii_type]
            )

    total_attempts = total_found + total_missed
    overall_accuracy = (total_found / total_attempts) * 100 if total_attempts > 0 else 0

    for doc_type, pii_types in doc_type_pii_type_counters.items():
        detection_percentages[doc_type] = {}
        for pii_type, counts in pii_types.items():
            total_attempts_per_pii = counts["found"] + counts["missed"]
            successful_detection_percentage = (
                (counts["found"] / total_attempts_per_pii) * 100
                if total_attempts_per_pii > 0
                else 0
            )
            detection_percentages[doc_type][pii_type] = successful_detection_percentage

    return (
        overall_accuracy,
        pii_type_counters,
        doc_type_pii_type_counters,
        detection_percentages,
        missed_detections,
    )
