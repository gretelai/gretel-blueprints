# utils.py
import os
import streamlit as st
from io import StringIO
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gretel_client import Gretel
from gretel_client.config import GretelClientConfigurationError
import subprocess, random
from st_aggrid import AgGrid
import matplotlib.pyplot as plt

def local_css(file_name: str) -> None:
    """Loads CSS styles from a local file into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def load_svg(svg_file_path: str) -> str:
    """Loads an SVG file and returns it as a string."""
    with open(svg_file_path, "r") as f:
        return f.read()

def display_header(logo_path: str) -> None:
    """Displays the app header with a logo."""
    svg_logo = load_svg(logo_path)
    scaled_svg = f"""<div class="logo-div">{svg_logo}</div>"""
    st.sidebar.markdown(scaled_svg, unsafe_allow_html=True)

def initialize_app():
    local_css("style.css")
    display_header(logo_path="gretel_logo.svg")
    init_session_state()

def init_session_state():
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Upload and Chunk"
    if "chunked_texts" not in st.session_state:
        st.session_state["chunked_texts"] = {}
    if "synthetic_data_ready" not in st.session_state:
        st.session_state["synthetic_data_ready"] = False

def display_navigation():
    st.sidebar.header("Navigation")
    # Button to navigate back to Upload and Chunk
    if st.sidebar.button("Upload and Chunk"):
        st.session_state["current_page"] = "Upload and Chunk"
        st.session_state["synthetic_data_ready"] = False  # Reset to allow re-chunking
    
    # Button to navigate to Generate Synthetic Data (enabled after chunking)
    if st.session_state["chunked_texts"]:
        if st.sidebar.button("Generate Synthetic Data"):
            st.session_state["current_page"] = "Generate Synthetic Data"

def render_current_page():
    if st.session_state["current_page"] == "Upload and Chunk":
        handle_upload_and_chunk()
    elif st.session_state["current_page"] == "Generate Synthetic Data" and st.session_state["chunked_texts"]:
        handle_generate_synthetic_data()

def handle_upload_and_chunk():
    uploaded_files = st.file_uploader("Upload files for text chunking", accept_multiple_files=True, type=["txt", "md"])
    if uploaded_files:
        process_and_display_chunks(uploaded_files)

def process_and_display_chunks(uploaded_files):
    chunk_size, chunk_overlap, min_chunk_chars = display_chunking_ui()
    if st.button("Chunk Texts"):
        st.session_state["chunked_texts"] = process_uploaded_files(
            uploaded_files, chunk_size, chunk_overlap, min_chunk_chars
        )
        display_chunk_info(st.session_state["chunked_texts"])
        st.session_state["synthetic_data_ready"] = True
    if "chunked_texts" in st.session_state and st.session_state["chunked_texts"]:
        show_random_chunk(st.session_state["chunked_texts"])

def handle_generate_synthetic_data():
    gretel_client = authenticate_gretel()
    if gretel_client:
        selected_topics, selected_user_profiles, selected_language = display_ui_components()
        if st.button("Generate Synthetic Records with Gretel"):
            synthetic_data = generate_synthetic_data(
                gretel_client, selected_topics, selected_user_profiles, selected_language, st.session_state.get("chunked_texts", {})
            )
            display_synthetic_data(synthetic_data)
            if not synthetic_data.empty:
                st.session_state["synthetic_data"] = synthetic_data
                download_synthetic_data(synthetic_data)

def download_synthetic_data(synthetic_data):
    csv = synthetic_data.to_csv(index=False)
    st.download_button(
        label="Download Synthetic Data as CSV",
        data=csv,
        file_name='synthetic_data.csv',
        mime='text/csv',
    )


def authenticate_gretel():
    """Authenticates with the Gretel API and returns a client instance."""
    st.text("To get started, please provide your Gretel API Key.")
    api_key = st.text_input("Gretel API Key", type="password")

    if api_key:
        try:
            gretel = Gretel(
                api_key=api_key,
                validate=True,
                clear=True,
            )
            return gretel

        except GretelClientConfigurationError:
            st.error(
                "Could not authenticate to Gretel. Please verify your API key is correct and belongs to an account on Gretel Teams or Enterprise."
            )
            st.stop()

def process_uploaded_files(uploaded_files, chunk_size: int, chunk_overlap: int, min_chunk_chars: int):
    """Processes uploaded files and returns a dictionary of chunked texts."""
    chunked_texts = {}
    for uploaded_file in uploaded_files:
        text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        valid_chunks = [chunk for chunk in chunks if len(chunk) > min_chunk_chars]
        if valid_chunks:
            chunked_texts[uploaded_file.name] = valid_chunks
        st.session_state['chunked_texts'] = chunked_texts
    return chunked_texts

def display_chunking_ui():
    """Displays UI components for text chunking parameters and returns the values."""
    st.subheader("Text Chunking Parameters")
    
    # User inputs for parameters with default values
    CHUNK_SIZE = st.number_input('Chunk Size (Tokens)', value=1500, min_value=1, help="Maximum number of tokens per chunk.")
    CHUNK_OVERLAP = st.number_input('Chunk Overlap (Tokens)', value=0, min_value=0, help="Number of tokens to overlap between consecutive chunks.")
    MIN_CHUNK_CHARS = st.number_input('Minimum Chunk Characters', value=1000, min_value=1, help="Minimum number of characters required for a chunk to be valid.")
    
    return CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS

def display_chunk_info(chunked_texts: dict) -> None:
    """Displays information about the processed chunks."""
    total_chunks = sum(len(chunks) for chunks in chunked_texts.values())
    st.write(f"Processed {len(chunked_texts)} files and found {total_chunks} chunks.")

def show_random_chunk(chunked_texts: dict) -> None:
    """Displays a button to show a random chunk from the processed texts, indicating its position."""
    if st.button('Show Random Chunk'):
        all_chunks = [chunk for chunks in chunked_texts.values() for chunk in chunks]
        if not all_chunks:
            st.warning("No chunks available to show.")
            return

        random_index = random.randint(0, len(all_chunks) - 1)
        random_chunk = all_chunks[random_index]
        chunk_position_message = f"Showing Chunk {random_index + 1} of {len(all_chunks)}"

        st.text_area(f"**{chunk_position_message}**", random_chunk, height=600)

def display_ui_components():
    """Displays UI components for selecting topics, user profiles, and language. Returns selections."""
    # Improved layout for selections using columns
    col1, col2, col3 = st.columns(3)

    # Topics selection
    with col1:
        st.markdown("""
            <h2 style="font-size: 24px; color: #000; font-weight: 600;">
                Select Topics
            </h2>
            """,unsafe_allow_html=True)
        topic_options = [
            'Basic Information', 'Pricing and Warranty', 'Usage',
            'Technical Details', 'Sustainability', 'Security', 'Future Updates'
        ]
        selected_topics = []
        for option in topic_options:
            if st.checkbox(option, key=f'topic_{option}'):
                selected_topics.append(option)

    # User profile selection
    with col2:
        st.markdown("""
            <h2 style="font-size: 24px; color: #000; font-weight: 600;">
                Select User Profile
            </h2>
            """,unsafe_allow_html=True)
        user_profile_options = ['beginner', 'intermediate', 'expert']
        selected_user_profiles = []
        for option in user_profile_options:
            if st.checkbox(option, key=f'profile_{option}'):
                selected_user_profiles.append(option)

    # Language selection
    with col3:
        st.markdown("""
            <h2 style="font-size: 24px; color: #000; font-weight: 600;">
                Select Language
            </h2>
            """,unsafe_allow_html=True)
        selected_language = st.radio(
            'Choose language from:',
            ('English', 'Dutch', 'French', 'Spanish'),
            key='language'
        )

    return selected_topics, selected_user_profiles, selected_language

# Function to construct the prompt based on user selections
def construct_prompt(topics, user_profiles, language):
    INTRO_PROMPT = "From the source text below, create a dataset with the following columns:\n"
    COLUMN_DETAILS = (
        "* `question`: Ask a set of unique questions related to the topic that a customer might ask. "
        "Questions should be relatively complex and specific enough to be addressed in a short answer.\n"
        "* `context`: Copy the exact sentence(s) from the source text and surrounding details from where the answer can be derived.\n"
        "* `truth`: Respond to the question with a clear, textbook quality answer that provides relevant details to fully address the question.\n"
    )

    SEED_PROMPT_1 = "* `topic`: select a topic from" + ', '.join(topics) + ".\n" if topics else ""
    SEED_PROMPT_2 = "* `user_profile`: The complexity level of the question and truth: chose from" + ', '.join(user_profiles) + ".\n" if user_profiles else ""
    SEED_PROMPT_3 = "* `language`: set the language as" + language + ".\n" if language else ""

    PROMPT = INTRO_PROMPT + SEED_PROMPT_1 + SEED_PROMPT_2 + SEED_PROMPT_3 + COLUMN_DETAILS
    return PROMPT

# Define a function to initialize the Gretel client
def initialize_tabllm(gretel):
    try:
        tabllm = gretel.factories.initialize_inference_api("tabllm", backend_model="gretelai/tabular-v0c")
        return tabllm
    except Exception as e:
        st.error(f"Error initializing Gretel client: {e}")
        return None

def generate_synthetic_data(gretel, selected_topics, selected_user_profiles, selected_language, chunked_texts):

    tabllm = initialize_tabllm(gretel)
    print(tabllm)

    # Define the number of samples per document and generation parameters
    SAMPLES_PER_DOC = 4
    GENERATE_PARAMS = {
        "num_records": SAMPLES_PER_DOC,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40
    }

    # Assuming the existence of a function to generate data, replaced with simplified logic for demonstration
    all_chunks = [chunk for chunks in st.session_state['chunked_texts'].values() for chunk in chunks]
    df = pd.DataFrame()  # Initialize dataframe to store results
    with st.spinner('Generating Data...'):

        for chunk in all_chunks:
            PROMPT = construct_prompt(selected_topics, selected_user_profiles, selected_language)
            df_doc = tabllm.generate(f"{PROMPT}\n\n{chunk}", **GENERATE_PARAMS)
            df = pd.concat([df, df_doc], ignore_index=True)
            
        # st.session_state['generated_data'] = df  # Save generated data in session state
        st.success('Data generation complete!')

        return df

def display_synthetic_data(synthetic_data: pd.DataFrame) -> None:
    """Displays the generated synthetic data in the Streamlit app.

    Args:
    - synthetic_data: A pandas DataFrame containing the synthetic data to display.
    """
    if synthetic_data.empty:
        st.write("No synthetic data has been generated.")
        return

    # Display DataFrame in the app
    st.write("Generated Synthetic Data:")
    AgGrid(synthetic_data[['topic','user_profile','language','question','context','truth']])

    # Optionally, provide options to download the synthetic data
    # download_button(synthetic_data)

def evaluate_rag_button(jsonl_file_path: str, df: pd.DataFrame = None) -> None:
    """Button to evaluate RAG and ensure the JSONL file exists or is created."""
    if st.button("Evaluate RAG"):
        if not os.path.exists(jsonl_file_path) and df is not None:
            df.to_json(jsonl_file_path, orient='records', lines=True, force_ascii=False)
            st.success(f"File {jsonl_file_path} created.")

        # Placeholder for the command execution logic
        try:
            result = subprocess.run(['ai', 'chat', 'evaluate', '--input-data', jsonl_file_path], check=True, capture_output=True, text=True)
            st.text_area("Command Output:", value=result.stdout, height=300)
        except subprocess.CalledProcessError as e:
            st.error(f"Command execution error: {e}")

def evaluate_rag(uploaded_file=None, data=None):
    """Evaluates RAG by plotting histograms of gpt_relevance, gpt_groundedness, and gpt_coherence."""
    df = pd.read_json('gretel_samples.jsonl', lines=True)

    # Display DataFrame in the app
    st.write("Generated Synthetic Data:")
    # st.dataframe(synthetic_data)
    # AgGrid(data[['topic','user_profile','language','question','context','truth']])

    filtered_scores = df[['gpt_relevance', 'gpt_groundedness', 'gpt_coherence']]

    # Changing the subplot layout to vertical (3 rows, 1 column)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))  # Adjust figsize for readability
    score_columns = ['gpt_relevance', 'gpt_groundedness', 'gpt_coherence']
    blurple_color = "#5534A5"  # Define the blurple color for the bars
    font_size = 14  # Define a smaller font size

    for i, col in enumerate(score_columns):
        axes[i].hist(filtered_scores[col].dropna(), bins=range(1, 7), align='left', rwidth=0.8, color=blurple_color, edgecolor=blurple_color)
        axes[i].set_title(f'{col} (Avg: {filtered_scores[col].mean():.2f})', fontsize=font_size + 2)  # Slightly larger font for titles
        axes[i].set_xticks(range(1, 6))
        axes[i].set_xlabel('Score', fontsize=font_size)
        axes[i].set_ylabel('Frequency', fontsize=font_size)
        axes[i].tick_params(axis='both', which='major', labelsize=font_size)  # Adjust tick label font size
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines

    plt.tight_layout()
    st.pyplot(fig)