import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gretel_client.tuner import BaseTunerMetric, MetricDirection


def calculate_sample_length(desired_max_len=0):
  """Calculates the sample length and adjusted maximum sequence length
  based on the desired ratio.

  Args:
    target_ratio: An optional integer representing the desired ratio of 
                  max_sequence_len to sample_len. Defaults to 15, but can be
                  overridden.
    desired_max_len: An optional integer representing the desired maximum sequence length. 
                      If 0, only the target_ratio is used to calculate sample_length. 
                      If provided, it is adjusted to be divisible by target_ratio 
                      and then used to calculate sample_length.

  Returns:
    A tuple containing two integers:
      - The calculated sample length.
      - The adjusted maximum sequence length divisible by the target ratio 
        and closest to the desired_max_len, or 0 if desired_max_len is not provided.
  """

  if desired_max_len > 0 and (not isinstance(desired_max_len, int) or desired_max_len <= 0):
    raise ValueError("desired_max_len must be a positive integer if provided")

  # Adjust desired_max_len based on adjusted_target_ratio
  if desired_max_len < 25:
    return 1, desired_max_len, desired_max_len
  
  # Consider potential target_ratios within the desired range (10-20)
  potential_ratios = range(10, 21)

  # Calculate adjusted_max_len and distance for each potential_ratio
  adjusted_max_lens = [math.ceil(desired_max_len / ratio) * ratio for ratio in potential_ratios]
  distances = [abs(adjusted_max_len - desired_max_len) for adjusted_max_len in adjusted_max_lens]

  # Find the index of the target_ratio with the smallest distance
  closest_ratio_index = distances.index(min(distances))

  # Extract the closest target_ratio and adjusted_max_len
  closest_target_ratio = potential_ratios[closest_ratio_index]
  adjusted_max_len = adjusted_max_lens[closest_ratio_index]

  # Calculate sample_length based on the closest target_ratio
  sample_length = adjusted_max_len // closest_target_ratio

  # Handle division by zero
  if sample_length == 0:
    sample_length = 1

  return sample_length, adjusted_max_len, closest_target_ratio


def pad_sequence(group, max_len, example_id_column, event_column, pad_value="[END]", attribute_columns=None):
    """
    Pads sequences within a DataFrame group to a specified maximum length, using the mean value
    for padding numeric columns, a specific pad value for categorical columns, and the existing value
    for specified attribute columns.
    
    Parameters:
    - group: DataFrame group (subset of a DataFrame usually obtained through groupby operation).
    - max_len: The desired maximum length for the sequences.
    - example_id_column: Name of the column containing example IDs.
    - event_column: Name of the column containing events.
    - pad_value: The value used for padding categorical columns. Defaults to "[END]".
    - attribute_columns: List of columns that should be padded with their existing value in the sequence.
    
    Returns:
    - A DataFrame with sequences padded to the specified maximum length.
    """
    pad_size = max_len - len(group)
    
    # Initialize padding dictionary with example_id_column and event_column
    padding_dict = {
        example_id_column: [group[example_id_column].iloc[0]] * pad_size,
        event_column: [pad_value] * pad_size,
    }
    
    # Automatically determine other columns to pad if attribute_columns is not specified
    if attribute_columns is None:
        attribute_columns = []
    
    other_columns = group.columns.difference([example_id_column, event_column] + attribute_columns)
    
    # Separate numeric and categorical columns
    numeric_cols = group[other_columns].select_dtypes(include=['number']).columns
    categorical_cols = group[other_columns].select_dtypes(exclude=['number']).columns
    
    # Pad numeric columns with their mean value
    for col in numeric_cols:
        mean_value = group[col].mean()
        padding_dict[col] = [mean_value] * pad_size
    
    # Pad categorical columns with the pad_value
    for col in categorical_cols:
        padding_dict[col] = [pad_value] * pad_size
    
    # Pad attribute columns with their existing value in the sequence
    for col in attribute_columns:
        attribute_value = group[col].iloc[0]  # Assuming the column has a fixed value for each sequence
        padding_dict[col] = [attribute_value] * pad_size
    
    # Create padding DataFrame
    padding = pd.DataFrame(padding_dict, index=[0] * pad_size)
    
    # Concatenate group with padding
    padded_group = pd.concat([group, padding], ignore_index=True)
    
    return padded_group


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
    df, event_column, df_ref=None, event_mapping=None, pad_value="[END]", output_file=None
):
    """
    Plots the distribution of event types for reference and optionally generated data.

    Parameters:
    - df: DataFrame containing the reference data.
    - event_column: The name of the column containing event types.
    - df_ref: (Optional) DataFrame containing the generated data.
    - event_mapping: (Optional) Dictionary to map event column values to a new naming convention.
    - output_file: (Optional) Path to save the plot instead of displaying it.
    """
    # Apply event_mapping if provided
    if event_mapping:
        df = df.copy()
        df.loc[:, event_column] = df[event_column].map(event_mapping)
        if df_ref is not None:
            df_ref = df_ref.copy()
            df_ref.loc[:, event_column] = df_ref[event_column].map(event_mapping)

    # Filter out rows with event_column as '[END]'
    df_filtered = df[df[event_column] != pad_value]

    # Get the union of event types present in reference and optionally in generated data
    all_event_types = df_filtered[event_column].unique()
    if df_ref is not None:
        df_ref_filtered = df_ref[df_ref[event_column] != pad_value]
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
        label="Synthetic" if df_ref is not None else "Data",
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
            label="Reference",
            alpha=0.7,
            color="orange",
        )

    plt.xlabel(event_column)
    plt.ylabel("Normalized Frequency")
    if df_ref is not None:
        plt.title(f"{event_column} Distribution: Reference vs Synthetic Data")
    else:
        plt.title(f"{event_column} Distribution")
    plt.xticks(index + bar_width / 2, labels=all_event_types, rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save to file if output_file is provided, else display
    if output_file:
        plt.savefig(output_file)
        plt.close()  # Close the figure to prevent display in notebook environments
    else:
        plt.show()


def plot_transition_matrices(df, event_column, example_id_column, df_ref=None, event_mapping=None, pad_value="[END]", output_file=None):
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
    df_filtered = df_copy[df_copy[event_column] != pad_value]
    if df_ref is not None:
        df_ref_filtered = df_ref_copy[df_ref_copy[event_column] != pad_value]

    # Compute transition matrices using the filtered copies
    transition_matrix = compute_transition_matrix(df_filtered, event_column, example_id_column)
    if df_ref is not None:
        transition_matrix_ref = compute_transition_matrix(df_ref_filtered, event_column, example_id_column)

    # Plot setup
    fig, axs = plt.subplots(nrows=1, ncols=2 if df_ref is not None else 1, figsize=(24 if df_ref is not None else 12, 10), sharey=True)
    if not isinstance(axs, np.ndarray):  # Adjust for a single subplot
        axs = [axs]

    # Plot the transition matrix
    sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, linewidths=.5, ax=axs[0])
    if df_ref is not None:
        axs[0].set_title('Synthetic Transition Probability Matrix')
    else:
        axs[0].set_title('Transition Probability Matrix')
    axs[0].set_xlabel('Next Event')
    axs[0].set_ylabel('Current Event')

    if df_ref is not None:
        # Plot the reference transition matrix
        sns.heatmap(transition_matrix_ref, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, linewidths=.5, ax=axs[1])
        axs[1].set_title('Reference Transition Probability Matrix')
        axs[1].set_xlabel('Next Event')

    plt.tight_layout()

    # Save to file if output_file is provided, else display
    if output_file:
        plt.savefig(output_file)
        plt.close()  # Close the figure to prevent display in notebook environments
    else:
        plt.show()


def plot_event_sequences(
    df,
    event_column,
    example_id_column,
    df_ref=None,
    num_sequences=5,
    event_mapping=None,
    pad_value="[END]",
    output_file=None
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
        df_filtered = df[df[event_column] != pad_value].copy()
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
                label=f"{example_id_column}: {str(label)[:10] + '...' if len(str(label)) > 10 else str(label)}",
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
        axs = [axs]

    random_ids = np.random.choice(
        df_filtered[example_id_column].unique(), num_sequences, replace=False
    )
    plot_sequences(
        axs[0],
        df_filtered[df_filtered[example_id_column].isin(random_ids)],
        random_ids,
        "Synthetic Sequences" if df_ref is not None else "Data Sequences",
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
            "Reference Data Sequences",
        )

    plt.tight_layout()

    # Save to file if output_file is provided, else display
    if output_file:
        plt.savefig(output_file)
        plt.close()  # Close the figure to prevent display in notebook environments
    else:
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


def is_strictly_subsequent(sequence):
    """
    Checks if a given sequence is strictly subsequent, starting with 1, allowing repeats but no gaps.

    This function assumes input as a pandas Series to leverage vectorized operations for efficiency.
    It checks if the sequence starts with 1 and if the difference between consecutive numbers is either 0 (repeat) or 1 (subsequent),
    which meets the criteria for being strictly subsequent.

    Parameters:
    - sequence (pandas.Series): A pandas Series of numeric values representing the sequence.

    Returns:
    - bool: True if the sequence starts with 1 and is strictly subsequent; False otherwise.
    """
    # Check if the sequence starts with 1
    if sequence.iloc[0] != 1:
        return False

    diffs = sequence.diff().fillna(1)  # Handle the first element as valid
    return ((diffs == 1) | (diffs == 0)).all()


def get_strictly_subsequent_sequences(df, example_id_column, event_column, pad_value="[END]"):
    """
    Filters sequences in a DataFrame for strict subsequency in events, calculating their prevalence.

    This function first removes padding from the DataFrame. It then evaluates each sequence identified
    by a unique identifier in the example_id_column for strict subsequency in the event_column. Sequences
    that are strictly subsequent (allow for consecutive or repeated events without gaps) are retained.
    The function returns these sequences and the percentage of total sequences that are strictly subsequent.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame with sequences to filter.
    - example_id_column (str): The column name identifying unique sequences.
    - event_column (str): The column name with numeric events to check for strict subsequency.
    - pad_value (str, optional): The padding value to be ignored. Defaults to "[END]".

    Returns:
    - (pandas.DataFrame, float): A tuple containing a filtered DataFrame of valid sequences and the percentage
      of sequences that are strictly subsequent.
    """
    df = undo_padding(df, event_column, pad_value=pad_value)
    
    valid_ids = []
    for identifier, group in df.groupby(example_id_column):
        if is_strictly_subsequent(group[event_column].reset_index(drop=True)):
            valid_ids.append(identifier)

    percentage_valid = (len(valid_ids) / df[example_id_column].nunique()) * 100
    df_valid = df[df[example_id_column].isin(valid_ids)]
    
    return df_valid, percentage_valid


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


def compute_transition_matrix(df, event_column, example_id_column, pad_value="[END]"):
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
        index=pad_value, columns=pad_value, errors="ignore"
    )
    return transition_matrix


def calculate_histograms(df, sequence_id, feature_columns, bin_configs=[25, 50, 100, 150]):
    """
    Calculate histograms for mean, min-max difference, median, and standard deviation for each numerical feature in the dataframe,
    across multiple bin configurations.
    """
    histograms = {}
    for feature in feature_columns:
        histograms[feature] = {config: {} for config in bin_configs}
        feature_data = df.groupby(sequence_id)[feature].agg(['mean', lambda x: x.max() - x.min(), 'median', 'std'])
        feature_data.rename(columns={'<lambda_0>': 'min_max_diff', 'median': 'medians', 'std': 'stdevs'}, inplace=True)

        for bins in bin_configs:
            histograms[feature][bins] = {
                'means': np.histogram(feature_data['mean'], bins=bins, density=True),
                'min_max_diff': np.histogram(feature_data['min_max_diff'], bins=bins, density=True),
                'medians': np.histogram(feature_data['medians'], bins=bins, density=True),
                'stdevs': np.histogram(feature_data['stdevs'], bins=bins, density=True)
            }
    return histograms

def calculate_weighted_histogram_distance(histogramsA, histogramsB, bin_configs=[25, 50, 100, 150], weights={'means': 0.25, 'min_max_diff': 0.25, 'medians': 0.25, 'stdevs': 0.25}):
    """
    Calculate the average weighted sum of Wasserstein distances between corresponding histograms for mean, 
    min-max difference, medians, and standard deviations across multiple bin configurations.
    """
    total_distances = []
    for bins in bin_configs:
        bin_distance = 0
        for feature in histogramsA:
            feature_distance = sum(
                weights[hist_type] * wasserstein_distance(histogramsA[feature][bins][hist_type][0], histogramsB[feature][bins][hist_type][0])
                for hist_type in ['means', 'min_max_diff', 'medians', 'stdevs']
                if hist_type in weights
            )
            bin_distance += feature_distance
        total_distances.append(bin_distance)
    
    # Average the distances across bin configurations
    avg_distance = np.mean(total_distances) if total_distances else 0
    return avg_distance


class EventTypeHistogramAndTransitionDistance(BaseTunerMetric):
    def __init__(
        self,
        reference_df,
        event_column,
        example_id_column,
        num_samples=100,
        hist_weight=0.5,
        trans_weight=0.5,
        pad_value="[END]"
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
        self.pad_value = pad_value
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
            self.reference_df, self.event_column, self.example_id_column, pad_value=self.pad_value
        )
        gen_tm = compute_transition_matrix(
            generated_data, self.event_column, self.example_id_column, pad_value=self.pad_value
        )
        tm_distance = transition_matrix_distance(ref_tm, gen_tm)

        # Calculate final weighted score
        final_score = (self.hist_weight * hist_distance) + (
            self.trans_weight * tm_distance
        )
        return final_score


class TimeSeriesDistance(BaseTunerMetric):
    def __init__(self, reference_df, example_id_column, feature_columns, num_samples=100):
        """
        Initialize the metric calculation class with reference data and parameters.
        
        Parameters:
        - reference_df (DataFrame): DataFrame containing the reference data.
        - example_id_column (str): The name of the column containing example IDs.
        - feature_columns (list): List of feature column names to be considered for distance calculation.
        - num_samples (int): Number of samples to generate for comparison.
        """
        self.reference_df = reference_df
        self.example_id_column = example_id_column
        self.feature_columns = feature_columns
        self.num_samples = num_samples
        self.direction = MetricDirection.MINIMIZE  # Specify the direction for metric optimization.

    def calculate_final_score(self, generated_df):
        """
        Calculates a weighted score based on histogram and autocorrelation distances between
        the reference and generated data.
        """
        # Compute histogram distances
        ref_hist = calculate_histograms(self.reference_df, self.example_id_column, self.feature_columns)
        gen_hist = calculate_histograms(generated_df, self.example_id_column, self.feature_columns)
        hist_distance = calculate_weighted_histogram_distance(ref_hist, gen_hist)
        
        return hist_distance

    def __call__(self, model):
        """
        Generates synthetic data using the provided model, compares it to the reference data,
        and calculates a weighted score based on histogram and autocorrelation distances.
        """
        generated_data = self.submit_generate_for_trial(
            model, num_records=self.num_samples
        )

        final_score = self.calculate_final_score(generated_data)
        return final_score