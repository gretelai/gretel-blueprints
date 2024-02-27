import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gretel_client.tuner import BaseTunerMetric, MetricDirection


# Function to pad sequences with '[END]'
def pad_sequence(group, max_len, example_id_column, event_column, pad_value="[END]"):
    """
    Pads sequences within a DataFrame group to a specified maximum length, ensuring all other columns
    are padded with their latest value in the sequence.

    Parameters:
    - group: DataFrame group (subset of a DataFrame usually obtained through groupby operation).
    - max_len: The desired maximum length for the sequences.
    - example_id_column: Name of the column containing example IDs.
    - event_column: Name of the column containing events.
    - pad_value: The value used for padding. Defaults to "[END]".

    Returns:
    - A DataFrame with sequences padded to the specified maximum length.
    """
    pad_size = max_len - len(group)
    if pad_size > 0:
        # Initialize the padding dictionary with the ID and event columns
        padding_dict = {
            example_id_column: [group[example_id_column].iloc[0]] * pad_size,
            event_column: [pad_value] * pad_size,
        }
        
        # Add other columns to the padding dictionary, using their last value for padding
        for col in group.columns:
            if col not in [example_id_column, event_column]:
                padding_dict[col] = [group[col].iloc[-1]] * pad_size
        
        # Create a DataFrame from the padding dictionary
        padding = pd.DataFrame(padding_dict, index=[0] * pad_size)
        
        # Concatenate the original group with the padding DataFrame
        return pd.concat([group, padding], ignore_index=True)
    else:
        # If no padding is needed, return the original group
        return group


def undo_padding(group, event_column, pad_value="[END]"):
    """
    Removes padding from sequences within a DataFrame group.

    Parameters:
    - group: DataFrame group (a subset of a DataFrame, usually obtained through a groupby operation).
    - event_column: Name of the column containing events.
    - pad_value: The value used for padding. Defaults to "[END]".

    Returns:
    - A DataFrame with padding removed from the sequences.
    """
    # Identify rows that do not contain the padding value in the event_column
    is_not_padded = group[event_column] != pad_value
    
    # Filter out the padded rows
    unpadded_group = group[is_not_padded]
    
    return unpadded_group


def plot_event_type_distribution(
    df, event_column, df_ref=None, event_mapping=None
):
    """
    Plots the distribution of event types for reference and optionally generated data.

    Parameters:
    - df: DataFrame containing the reference data.
    - event_column: The name of the column containing event types.
    - df_ref: (Optional) DataFrame containing the generated data.
    - event_mapping: (Optional) Dictionary to map event column values to a new naming convention.
    """
    # Apply event_mapping if provided
    if event_mapping:
        df = df.copy()
        df.loc[:, event_column] = df[event_column].map(event_mapping)
        if df_ref is not None:
            df_ref = df_ref.copy()
            df_ref.loc[:, event_column] = df_ref[event_column].map(event_mapping)

    # Filter out rows with event_column as '[END]'
    df_filtered = df[df[event_column] != "[END]"]

    # Get the union of event types present in reference and optionally in generated data
    all_event_types = df_filtered[event_column].unique()
    if df_ref is not None:
        df_ref_filtered = df_ref[df_ref[event_column] != "[END]"]
        all_event_types = np.union1d(
            all_event_types, df_ref_filtered[event_column].unique()
        )

    # Calculate normalized value counts ensuring all event types are included
    ref_hist = (
        df_filtered[event_column]
        .value_counts(normalize=True)
        .reindex(all_event_types, fill_value=0)
        .sort_index()
    )
    bar_width = 0.35
    index = np.arange(len(all_event_types))

    plt.figure(figsize=(24 if df_ref is not None else 12, 6))
    plt.bar(
        index,
        ref_hist.values,
        bar_width,
        label="Reference",
        alpha=0.7,
        color="blue",
    )

    if df_ref is not None:
        gen_hist = (
            df_ref_filtered[event_column]
            .value_counts(normalize=True)
            .reindex(all_event_types, fill_value=0)
            .sort_index()
        )
        plt.bar(
            index + bar_width,
            gen_hist.values,
            bar_width,
            label="Generated",
            alpha=0.7,
            color="orange",
        )

    plt.xlabel(event_column)
    plt.ylabel("Normalized Frequency")
    plt.title(f"{event_column} Distribution: Reference vs Generated Data")
    plt.xticks(index + bar_width / 2, labels=all_event_types, rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_transition_matrices(df, event_column, example_id_column, df_ref=None, event_mapping=None):
    """
    Plots transition probability matrices for reference and optionally generated data, with generic column handling.

    Parameters:
    - df: DataFrame containing the reference data.
    - event_column: The name of the column containing event types.
    - example_id_column: The name of the column containing example IDs.
    - df_ref: (Optional) DataFrame containing the generated data.
    - event_mapping: (Optional) Dictionary to map event column values to a new naming convention.
    """
    # Create copies of the dataframes to avoid modifying the original data
    df_copy = df.copy()
    df_ref_copy = df_ref.copy() if df_ref is not None else None

    # Apply event_mapping if provided to the copies
    if event_mapping:
        df_copy[event_column] = df_copy[event_column].map(event_mapping)
        if df_ref_copy is not None:
            df_ref_copy[event_column] = df_ref_copy[event_column].map(event_mapping)
    
    # Filter out rows with event_column as '[END]'
    df_filtered = df_copy[df_copy[event_column] != "[END]"]
    if df_ref is not None:
        df_ref_filtered = df_ref_copy[df_ref_copy[event_column] != "[END]"]

    # Compute transition matrices using the filtered copies
    transition_matrix_ref = compute_transition_matrix(df_filtered, event_column, example_id_column)
    if df_ref is not None:
        transition_matrix_gen = compute_transition_matrix(df_ref_filtered, event_column, example_id_column)

    # Plot setup
    fig, axs = plt.subplots(nrows=1, ncols=2 if df_ref is not None else 1, figsize=(24 if df_ref is not None else 12, 10), sharey=True)
    if not isinstance(axs, np.ndarray):  # Adjust for a single subplot
        axs = [axs]

    # Plot the reference transition matrix
    sns.heatmap(transition_matrix_ref, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, linewidths=.5, ax=axs[0])
    axs[0].set_title('Reference Transition Probability Matrix')
    axs[0].set_xlabel('Next Event')
    axs[0].set_ylabel('Current Event')

    if df_ref is not None:
        # Plot the generated transition matrix
        sns.heatmap(transition_matrix_gen, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, linewidths=.5, ax=axs[1])
        axs[1].set_title('Generated Transition Probability Matrix')
        axs[1].set_xlabel('Next Event')

    plt.tight_layout()
    plt.show()


def plot_event_sequences(
    df,
    event_column,
    example_id_column,
    df_ref=None,
    num_sequences=5,
    event_mapping=None,
):
    """
    Plots event sequences for a specified number of randomly selected sequences from one or two DataFrames,
    allowing for generic column names.

    Parameters:
    - df: DataFrame containing the reference data.
    - event_column: The name of the column containing event types.
    - example_id_column: The name of the column containing example IDs.
    - df_ref: (Optional) DataFrame containing the generated data.
    - num_sequences: Number of sequences to randomly select for plotting.
    - event_mapping: (Optional) Dictionary to map event column values to a new naming convention.
    """

    def prepare_df(df):
        """Preprocess DataFrame and map event_column."""
        df_filtered = df[df[event_column] != "[END]"].copy()
        if event_mapping:
            df_filtered[event_column] = df_filtered[event_column].map(
                event_mapping
            )
        return df_filtered

    def plot_sequences(ax, df, random_ids, title):
        """Plot sequences on a given Axes object."""
        sorted_events = sorted(df[event_column].unique())
        event_to_int = {event: i for i, event in enumerate(sorted_events)}

        df = df.copy()
        df.loc[:, event_column + "_INT"] = df[event_column].map(event_to_int)
        for label in random_ids:
            dfi = df[df[example_id_column] == label].reset_index()
            ax.plot(
                dfi.index,
                dfi[event_column + "_INT"],
                marker="o",
                linestyle="-",
                label=f"{example_id_column}: {label}",
            )

        ax.set_xlabel("Event Sequence Steps")
        ax.set_ylabel("Events")
        ax.set_yticks(list(event_to_int.values()))
        ax.set_yticklabels(sorted_events)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True)
        ax.set_title(title)

    df_filtered = prepare_df(df)

    ncols = 2 if df_ref is not None else 1
    fig, axs = plt.subplots(1, ncols, figsize=(10 * ncols, 6))
    if ncols == 1:
        axs = [axs]  # Make it a list to use axs[0] without error

    random_ids = np.random.choice(
        df_filtered[example_id_column].unique(), num_sequences, replace=False
    )
    plot_sequences(
        axs[0],
        df_filtered[df_filtered[example_id_column].isin(random_ids)],
        random_ids,
        "Reference Data Sequences",
    )

    if df_ref is not None:
        df_ref_filtered = prepare_df(df_ref)
        random_ids_ref = np.random.choice(
            df_ref_filtered[example_id_column].unique(),
            num_sequences,
            replace=False,
        )
        plot_sequences(
            axs[1],
            df_ref_filtered[
                df_ref_filtered[example_id_column].isin(random_ids_ref)
            ],
            random_ids_ref,
            "Generated Data Sequences",
        )

    plt.tight_layout()
    plt.show()


def check_series_order(series, valid_sequence):
    """
    Checks if a series of events follows a specified valid order, including handling optional events.

    Parameters:
    - series: A pandas Series containing a sequence of events to check.
    - valid_sequence: A list specifying the valid order of events, where optional events are included as sublists.

    Returns:
    - True if the series follows the valid sequence order, including correctly placed optional events. False otherwise.
    """
    seq_index = 0
    non_optional_count = sum(
        1 for item in valid_sequence if not isinstance(item, list)
    )

    if len(series) < non_optional_count:
        return False  # More events than expected

    for event in series:
        if seq_index >= len(valid_sequence):
            return False  # More events than expected

        # Check if the event matches the current expected event or optional ones
        expected_event = valid_sequence[seq_index]
        if isinstance(expected_event, list):
            if event not in expected_event:
                seq_index += 1  # Move to the next expected event if current is not optional
        if (
            seq_index < len(valid_sequence)
            and event == valid_sequence[seq_index]
        ):
            seq_index += 1
        elif (
            seq_index < len(valid_sequence)
            and isinstance(valid_sequence[seq_index], list)
            and event in valid_sequence[seq_index]
        ):
            continue  # Valid optional event, stay at current sequence index
        else:
            return False  # Invalid event sequence
    return True

def remove_invalid_sequences(df, series_validity, example_id_column):
    """
    Filters out sequences based on their validity.

    Parameters:
    - df: DataFrame containing the data.
    - series_validity: A Series indicating the validity of each sequence (True for valid, False for invalid).
    - example_id_column: The name of the column containing example IDs.
    """
    # Filter out invalid sequences using the series_validity mask
    valid_series_ids = series_validity[series_validity].index.tolist()
    valid_df = df[df[example_id_column].isin(valid_series_ids)]
    return valid_df


def calculate_percentage_of_valid_sequences(series_validity):
    """
    Calculates the percentage of valid sequences in a given series.

    Parameters:
    - series_validity: A pandas Series where each value indicates whether a sequence is valid (True) or not (False).

    Returns:
    - The percentage of valid sequences.
    """
    percentage_valid = (series_validity.sum() / len(series_validity)) * 100
    return percentage_valid


def histogram_distance(hist1, hist2):
    """
    Calculates the distance between two histograms. Assumes histograms are pandas Series.

    Parameters:
    - hist1: Histogram 1 as a pandas Series.
    - hist2: Histogram 2 as a pandas Series.
    """
    # Get the union of all event types (index values) present in both histograms
    all_events = set(hist1.index).union(set(hist2.index))

    # Reindex both histograms to include all events, filling missing values with 0
    hist1_reindexed = hist1.reindex(all_events, fill_value=0)
    hist2_reindexed = hist2.reindex(all_events, fill_value=0)

    # Calculate the Euclidean distance between the two reindexed histograms
    distance = np.sqrt(((hist1_reindexed - hist2_reindexed) ** 2).sum())

    return distance


def transition_matrix_distance(tm1, tm2):
    """
    Calculates the Frobenius distance between two transition matrices.

    Parameters:
    - tm1: Transition matrix 1.
    - tm2: Transition matrix 2.
    """
    # Ensure both matrices cover the same events, filling missing values with zeros
    all_events = set(tm1.index).union(tm2.index)
    tm1_reindexed = tm1.reindex(
        index=all_events, columns=all_events, fill_value=0
    )
    tm2_reindexed = tm2.reindex(
        index=all_events, columns=all_events, fill_value=0
    )

    # Calculate the Frobenius distance
    distance = np.linalg.norm(tm1_reindexed - tm2_reindexed)
    return distance


def compute_transition_matrix(df, event_column, example_id_column):
    """
    Calculates the transition matrix for event sequences within each example ID.

    Parameters:
    - df: DataFrame containing the data.
    - event_column: The name of the column containing event types.
    - example_id_column: The name of the column containing example IDs.
    """
    df = df.copy()
    df.loc[:, "Next_" + event_column] = df.groupby(example_id_column)[
        event_column
    ].shift(-1)
    transition_counts = (
        df.groupby([event_column, "Next_" + event_column])
        .size()
        .unstack(fill_value=0)
    )
    transition_matrix = transition_counts.div(
        transition_counts.sum(axis=1), axis=0
    ).fillna(0)
    transition_matrix = transition_matrix.drop(
        index="[END]", columns="[END]", errors="ignore"
    )
    return transition_matrix


class EventTypeHistogramAndTransitionDistance(BaseTunerMetric):
    def __init__(
        self,
        reference_df,
        event_column,
        example_id_column,
        num_samples=100,
        hist_weight=0.5,
        trans_weight=0.5,
    ):
        """
        Initializes the metric calculation class with reference data and parameters.

        Parameters:
        - reference_df: DataFrame containing the reference data.
        - event_column: The name of the column containing event types.
        - example_id_column: The name of the column containing example IDs.
        - num_samples: Number of samples to generate for comparison.
        - hist_weight: Weight for the histogram distance in the final score.
        - trans_weight: Weight for the transition matrix distance in the final score.
        """
        self.reference_df = reference_df
        self.event_column = event_column
        self.example_id_column = example_id_column
        self.num_samples = num_samples
        self.hist_weight = hist_weight
        self.trans_weight = trans_weight
        self.direction = MetricDirection.MINIMIZE

    def __call__(self, model):
        """
        Generates synthetic data using the provided model, compares it to the reference data,
        and calculates a weighted score based on histogram and transition matrix distances.
        """
        generated_data = self.submit_generate_for_trial(
            model, num_records=self.num_samples
        )

        # Compute and normalize histograms for the event column
        ref_hist = self.reference_df[self.event_column].value_counts(
            normalize=True
        )
        gen_hist = generated_data[self.event_column].value_counts(
            normalize=True
        )
        hist_distance = histogram_distance(ref_hist, gen_hist)

        # Compute normalized transition matrices
        ref_tm = compute_transition_matrix(
            self.reference_df, self.event_column, self.example_id_column
        )
        gen_tm = compute_transition_matrix(
            generated_data, self.event_column, self.example_id_column
        )
        tm_distance = transition_matrix_distance(ref_tm, gen_tm)

        # Calculate final weighted score
        final_score = (self.hist_weight * hist_distance) + (
            self.trans_weight * tm_distance
        )
        return final_score
