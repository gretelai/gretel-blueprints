import streamlit as st
import pandas as pd
import random
import itertools
from faker import Faker
from gretel_client import Gretel


def local_css(file_name: str) -> None:
    """Loads CSS styles from a local file into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def load_svg(svg_file_path: str) -> str:
    """Loads an SVG file and returns it as a string."""
    with open(svg_file_path, "r") as f:
        return f.read()


def display_header(logo_path: str) -> None:
    """Displays the app header with a logo and a gray line underneath as a banner."""
    svg_logo = load_svg(logo_path)
    banner_line_html = (
        '<div class="banner-line"></div>'  # Using the CSS class for styling
    )
    scaled_svg = (
        f"""<div class="logo-div">{svg_logo}</div>{banner_line_html}"""
    )
    st.markdown(scaled_svg, unsafe_allow_html=True)


def initialize_app():
    # st.set_page_config(layout="wide")
    local_css("style.css")
    display_header(logo_path="gretel_logo.svg")


# ----- Utility Functions -----
def initialize_gretel_client():
    """
    Initialize the Gretel client with a user-provided API key.

    Returns:
        tuple: (gretel, tabllm) if successful, otherwise (None, None).
    """
    st.subheader("1. Initialization")
    st.write(
        """
    Before we start generating synthetic documents, we need to initialize the connection to Gretel's services.
    Please enter your Gretel API key below to authenticate and initialize the necessary models for data generation.
    """
    )
    # Input for the API key
    api_key = st.text_input("Enter your Gretel API key", type="password")
    # api_key = st.secrets["gretel"]["api_key"]

    if api_key:
        try:
            # Assuming Gretel and its dependencies are imported and available
            gretel = Gretel(api_key=api_key)
            tabllm = gretel.factories.initialize_inference_api(
                backend_model="gretelai/tabular-v0"
            )
            st.success("Gretel client initialized successfully.")

            # Display the essential calls in an expandable code block
            code_snippet = """
# Code to initialize Gretel client
# $ pip install gretel_client
# Retrieve the API key from https://console.gretel.ai/users/me/key

from gretel_client import Gretel
gretel = Gretel(api_key="prompt")
navigator = gretel.factories.initialize_inference_api(backend_model="gretelai/tabular-v0")
            """
            with st.expander("Show Initialization Code"):
                st.code(code_snippet, language="python")

            return gretel, tabllm
        except Exception as e:
            st.error(f"Failed to initialize Gretel client: {e}")
    else:
        st.warning("Please enter your Gretel API key.")

    return None, None


def get_document_types():
    """
    Returns a list of predefined document types.
    """
    document_types = [
        "Email",
        "Customer support conversation",
        "Financial Statement",
        "Insurance Policy",
        "Loan Application",
        "Bill of Lading",
        "Safety Data Sheet",
        "Policyholder's Report",
        "XBRL",
        "EDI",
        "SWIFT Messages",
        "FIX Protocol",
        "FpML",
        "ISDA Definitions",
        "BAI Format",
        "MT940",
    ]
    return document_types


def create_doc_prompt(selected_document_types):
    """
    Dynamically creates a document prompt based on selected document types.

    Args:
        selected_document_types (list): List of user-selected document types.

    Returns:
        str: A dynamically constructed document prompt.
    """
    document_types_examples = ", ".join(selected_document_types)

    DOCUMENT_TYPE_PROMPT = f"""
    You are a data expert across the financial services, 
    insurance, and banking verticals. Generate a diverse dataset of domains and
    detailed descriptions for various document types, including specific formats and 
    schemas, as they relate to the customer journey within a Finance, 
    Insurance, Fintech, or Banking company.

    Columns:
    * document_type: choose from the following list ({document_types_examples}).
    * document_description: A one-sentence detailed description of the kind of 
      documents found in this domain, including specifics about format, 
      common fields, and content type where applicable. Describe the schema, 
      structure, and length of the data format that could be used as instructions
      to create a document from scratch.

    Remember to customize fields and formats based on the specific requirements
    of each domain to accurately reflect the variety and complexity of documents 
    in a SaaS company environment.
    """

    return DOCUMENT_TYPE_PROMPT.strip()


def generate_document_descriptions(tabllm, prompt, num_document_types):
    """
    Generate descriptions for document types using Gretel's tabllm.generate based on a provided prompt.

    Args:
        tabllm: Initialized Gretel tabllm client.
        prompt (str): The constructed prompt to guide the LLM generation process.
        num_document_types (int): The number of document descriptions to generate.

    Returns:
        pd.DataFrame: DataFrame containing the generated document types and their descriptions.
    """
    # Generate document descriptions based on the provided prompt
    df = tabllm.generate(prompt=prompt, num_records=num_document_types)

    return df


def show_generation_code(doc_prompt, num_document_types):
    code_snippet = f"""
# Dynamically constructed prompt for document type descriptions
DOCUMENT_TYPE_PROMPT = f\"\"\"{doc_prompt}\"\"\"

# Gretel navigator.generate call using the constructed prompt
descriptions_df = navigator.generate(prompt=DOCUMENT_TYPE_PROMPT, num_records={num_document_types})
    """
    st.code(code_snippet, language="python")


def create_document_types_section(gretel, tabllm):
    st.subheader("2. Create Document Types")
    st.write(
        """
    Our goal is to create synthetic examples across a wide variety of document types.
    This is a case where we can leverage an LLM's inherent knowledge of different industry verticals
    and data types to generate these document types and schemas, without having to go to the trouble of
    thinking of all possibilities, and then crawling the web to find examples.
    """
    )

    # Document Type Selection and Customization
    document_types = (
        get_document_types()
    )  # Assuming this function is defined elsewhere
    selected_document_types = st.multiselect(
        "Select document types", document_types
    )

    # Optionally, allow adding a custom tag
    custom_tag = st.text_input("Add a custom document type (optional)")
    if custom_tag:
        selected_document_types.append(custom_tag)

    if selected_document_types:
        # User-configurable parameter for the number of document types
        num_document_types = st.number_input(
            "Number of document types to generate descriptions for",
            min_value=1,
            max_value=100,
            value=len(selected_document_types),
            step=1,
        )

        # Display the generation code as soon as document types are selected
        doc_prompt = create_doc_prompt(selected_document_types)
        with st.expander("Show Document Type Generation Code"):
            show_generation_code(
                doc_prompt, num_document_types
            )  # This displays the code snippet

        if st.button("Generate Document Type Descriptions"):
            with st.spinner("Generating document type descriptions..."):
                doc_df = generate_document_descriptions(
                    tabllm, doc_prompt, num_document_types
                )

                # Display the generated descriptions
                st.dataframe(doc_df)

                # Update the session state with generated document types and descriptions
                st.session_state.document_types = dict(
                    zip(
                        doc_df["document_type"], doc_df["document_description"]
                    )
                )
        else:
            st.warning("Please select at least one document type.")


def show_contextual_tags_code_snippet():
    code_snippet = """
# Sample code for generating contextual tags
sampled_contextual_tag_data = [
(
    document_type,
    document_type_dict[document_type],
    pii_type,
    locale,
    pii_generator.sample(pii_type, sample_size=pii_values_count)
)
for _ in range(n_rows)
for document_type in [random.choice(list(document_type_dict.keys()))]
for pii_type in [random.choice(selected_pii_types)]
for locale in [random.choice(list(language_dict.keys()))]
]
contextual_tags_df = pd.DataFrame(sampled_contextual_tag_data)
    """
    st.code(code_snippet, language="python")


def configure_pii_generator():
    # Instantiate and configure the PII generator
    locale_list = ["en_US", "nl_NL"]
    pii_generator = PIIGenerator(locales=locale_list)

    # Add Faker generators for various PII types
    pii_types_to_generate = {
        # 'Name': 'name',
        "First name": "first_name",
        "Last name": "last_name",
        "Email": "email",
        "Phone number": "phone_number",
        "Full address": "address",
        "Street address": "street_address",
        "Credit card": "credit_card_number",
        "Org or Company Name": "company",
        "Date of birth": "date_of_birth",
        "Zip code": "zipcode",
        "IBAN number": "iban",
        "IPv4 address": "ipv4",
        "IPv6 address": "ipv6",
        "US bank number": "bban",
        # 'Job Title': 'job',  # Adds job titles
        # 'License Plate': 'license_plate',  # Adds vehicle license plate numbers
        "SSN": "ssn",  # Adds U.S. Social Security numbers
        # 'User Name': 'user_name',  # Adds a username
        "Password": "password",  # Generates a random password
        # 'City': 'city',  # Generates a city name
        # 'Country': 'country',  # Generates a country name
        # 'Currency Code': 'currency_code',  # Generates currency codes
        # 'File Path': 'file_path',  # Generates a random file path
        # 'Language Code': 'language_code',  # Generates language codes
        # 'Phone number (International)': 'phone_number',  # International format phone numbers
        # 'Time Zone': 'timezone',  # Generates time zones
        # 'URL': 'url',  # Generates a random URL
        # 'User Agent': 'user_agent',  # Generates a user agent string for browsers
        # 'Color': 'color',  # Generates a random color
        "Company Email": "company_email",  # Generates a company email address
        # 'Domain Name': 'domain_name',  # Generates domain names
    }

    for name, method in pii_types_to_generate.items():
        pii_generator.add_faker_generator(name, method)

    # Add custom lists for specific PII types
    pii_generator.add_custom_list(
        "GPS latitude and longitude coordinates",
        [
            "40.56754, -89.64066",
            "25.13915, 73.06784",
            "-7.60361, 37.00438",
            "33.35283, -111.78903",
            "17.54907, 82.85749",
        ],
    )
    pii_generator.add_custom_list(
        "Customer ID", ["ID-001", "ID-002", "ID-003", "ID-004", "ID-005"]
    )

    # Build and return a dictionary of all PII generators
    return pii_generator


def generate_contextual_tags(
    selected_pii_types, document_type_dict, pii_generator
):
    # Header for the section
    st.subheader("3. Generate Contextual Tags")

    # Descriptive text explaining the process and objective
    st.write(
        """
    This is the final stage of data preparation before using Gretel to generate data at scale. In this step, we compile all of the following tags to create a "recipe" that can guide Gretel Navigator to generate highly diverse synthetic data at scale.

    For this dataset, we will guide each LLM generation with the following properties:
    - **Document type**
    - **Document description**
    - **Language**
    - **PII type**
    - **PII values**
    """
    )

    # Sliders for numeric inputs
    n_rows = st.slider(
        "Total number of contextual tags to generate",
        min_value=5,
        max_value=5000,
        value=1000,
        step=100,
    )
    max_pii_types = st.slider(
        "Maximum number of PII types in each document",
        min_value=1,
        max_value=3,
        value=2,
    )
    pii_values_count = st.slider(
        "Number of PII values to generate for each PII type",
        min_value=1,
        max_value=10,
        value=3,
    )

    # Multiselect for languages
    language_options = {
        "english_us": "Content in English as spoken and written in the United States",
        "spanish_spain": "Content in Spanish as spoken and written in Spain",
        "french_france": "Content in French as spoken and written in France",
        "german_germany": "Content in German as spoken and written in Germany",
        "italian_italy": "Content in Italian as spoken and written in Italy",
        "japanese_japan": "Content in Japanese as spoken and written in Japan",
        "dutch_netherlands": "Content in Dutch as spoken and written in the Netherlands",
        "swedish_sweden": "Content in Swedish as spoken and written in Sweden",
        "english_uk": "Content in English as spoken and written in the United Kingdom",
        "spanish_mexico": "Content in Spanish as spoken and written in Mexico",
        "portuguese_brazil": "Content in Portuguese as spoken and written in Brazil",
    }
    selected_languages = st.multiselect(
        "Select languages",
        list(language_options.keys()),
        default=["english_us"],
    )
    language_dict = {
        lang: language_options[lang] for lang in selected_languages
    }

    # Logic to generate sampled contextual tag data
    if st.button("Generate Contextual Tags"):
        sampled_contextual_tag_data = []
        for _ in range(n_rows):
            document_type = random.choice(list(document_type_dict.keys()))
            locale = random.choice(list(language_dict.keys()))
            # Select a random number of PII types between 1 and 3
            num_pii_types = random.randint(1, max_pii_types)
            print(pii_values_count)
            pii_types = random.sample(
                list(selected_pii_types.keys()), num_pii_types
            )

            # Initialize lists to hold the selected PII types and their corresponding values
            selected_pii_types_list = []
            pii_values_list = []

            for pii_type in pii_types:
                # Sample the PII values for each selected PII type
                pii_values = pii_generator.sample(
                    pii_type, sample_size=pii_values_count
                )
                selected_pii_types_list.append(pii_type)
                pii_values_list.append(pii_values)

            # Create a single data entry with lists of PII types and their values
            data_entry = (
                document_type,
                document_type_dict[document_type],
                selected_pii_types_list,  # This now contains a list of selected PII types
                locale,
                pii_values_list,  # This now contains a list of lists of PII values
            )
            sampled_contextual_tag_data.append(data_entry)

        # Convert sampled data to a DataFrame
        contextual_tags_df = pd.DataFrame(
            sampled_contextual_tag_data,
            columns=[
                "document_type",
                "document_description",
                "pii_type",
                "language",
                "pii_values",
            ],
        )
        st.session_state.contextual_tags = contextual_tags_df

        # Display the number of contextual tag permutations created and a preview of the DataFrame
        st.write(
            f"Created {len(contextual_tags_df)} contextual tag permutations"
        )
        st.dataframe(contextual_tags_df.head(10))


def add_markup_to_text(text, pii_types_dict):
    for pii_type, pii_value_list in pii_types_dict.items():
        for pii_value in pii_value_list:
            marked_up_pii = f"{{[{pii_type}]{pii_value}}}"
            text = text.replace(pii_value, marked_up_pii)
    return text


def generate_text2pii_data(tabllm, row, num_docs_per_context, min_text_length):
    document_type = row["document_type"]
    document_description = row["document_description"]
    pii_types_dict = {}
    pii_type = row["pii_type"]
    for k in range(len(pii_type)):
        pii_types_dict[pii_type[k]] = row["pii_values"][k]
    pii_values_markdown = ", or ".join(
        [
            f"'{item}'"
            for key, values_list in pii_types_dict.items()
            for item in values_list
        ]
    )
    language = row["language"]

    generated_records = []
    failed_count = 0

    create_prompt = f"""
Create a unique, comprehensive dataset entry as described below. 
Each entry should differ substantially in content, style, and perspective.

Dataset format: Two columns - 'document_type' and 'document_text'

Entry specifications:

'document_type': "{document_type}"
'document_text': A complete, coherent, and distinct synthetic {document_description} in {language}, formatted as a detailed {document_type}
  * Incorporate varied themes, styles, viewpoints, and structures
  * Use vivid descriptions, examples, and elaborations
  * Avoid repetition; ensure each entry stands out
  * Maintain coherence and logical flow
  * Seamlessly integrate the following {pii_type} values exactly as provided into the text: {pii_values_markdown}
  * Identify appropriate locations within the document to naturally incorporate these values
  * Provide context for each {pii_type}, explaining its relevance to the {document_type}
  * Ensure the {pii_type} values fit grammatically and contextually within the surrounding text
  * Maintain the overall structure and coherence of the {document_type}
Aim to create a rich, detailed, and engaging {document_type} that showcases creativity and diversity while seamlessly incorporating the provided {pii_type} values.
"""

    while len(generated_records) < num_docs_per_context:
        # Generate initial documents
        results = tabllm.generate(
            prompt=create_prompt, num_records=num_docs_per_context
        )

        # Add 'markup' column by applying the markup helper function
        results["text_markup"] = results["document_text"].apply(
            lambda text: add_markup_to_text(text, pii_types_dict)
        )

        # Filter out rows where the marked-up text is not different from the provided text
        failed_results = results[
            (results["text_markup"] == results["document_text"])
            | (results["document_text"].str.len() < min_text_length)
        ]

        # Store the successfully generated records
        generated_records.extend(
            results[~results.index.isin(failed_results.index)][
                ["document_type", "document_text", "text_markup"]
            ].values.tolist()
        )
        failed_count += len(failed_results)

    return pd.DataFrame(
        generated_records,
        columns=["document_type", "document_text", "text_markup"],
    )


def display_generated_data(dataframe):
    # Display the dataframe in Streamlit with option for verbose mode
    st.write(dataframe)


def create_synthetic_dataset(tabllm, contextual_tags_df):
    st.subheader("4. Creating Synthetic Text-to-PII Dataset")
    st.write(
        """
    We have completed the contextual tags to guide our LLM with synthetic data generation, and now we are ready to generate synthetic data at scale. To do this, we'll prompt Gretel to create a new dataset of synthetic records matching the desired `document_type`, `language`, and sampling `PII` attributes from our generator.
    """
    )

    num_docs_per_context = st.number_input(
        "Number of Documents per Context", min_value=1, value=3, step=1
    )
    min_text_length = st.number_input(
        "Minimum Text Length", min_value=50, value=200, step=50
    )

    # Use a button to trigger the display of the code for review
    with st.expander("Show Synthetic Data Generation Code"):
        show_synthetic_data_generation_code(
            contextual_tags_df.iloc[0], num_docs_per_context, min_text_length
        )

    if st.button("Generate Synthetic Dataset"):
        results = []

        # Assume tqdm works well in your environment; replace with a suitable progress bar if needed
        with st.spinner("Generating text-to-pii data..."):
            for index, row in contextual_tags_df.iterrows():
                result_df = generate_text2pii_data(
                    tabllm, row, num_docs_per_context, min_text_length
                )
                results.append(result_df)

        # Concatenate all the DataFrames in the list into a single DataFrame
        final_results = pd.concat(results, ignore_index=True)

        # Apply text wrapping to the DataFrame before displaying
        final_results_styled = final_results.style.set_properties(
            subset=["document_text", "text_markup"],
            **{"width": "300px", "white-space": "normal"},
        )
        st.write(final_results_styled.to_html(), unsafe_allow_html=True)


def show_synthetic_data_generation_code(
    row, num_docs_per_context, min_text_length
):
    """
    Display the code used for generating document text based on PII data in Streamlit.

    Args:
        row (dict): A dictionary representing a single row from the DataFrame used in generation.
        num_docs_per_context (int): The number of documents to generate per context.
        min_text_length (int): The minimum text length for the generated documents.
    """
    # Reconstructing the pii_types_dict and pii_values_markdown from the row
    pii_types_dict = {
        pii_type: row["pii_values"][k]
        for k, pii_type in enumerate(row["pii_type"])
    }
    pii_values_markdown = ", or ".join(
        [
            f"'{item}'"
            for key, values_list in pii_types_dict.items()
            for item in values_list
        ]
    )

    # Building the create_prompt string
    create_prompt = f"""
Create a unique, comprehensive dataset entry as described below. Each entry should differ substantially in content, style, and perspective.

Dataset format: Two columns - 'document_type' and 'document_text'

Entry specifications:

'document_type': "{row['document_type']}"
'document_text': A complete, coherent, and distinct synthetic {row['document_description']} in {row['language']}, formatted as a detailed {row['document_type']}
  * Incorporate varied themes, styles, viewpoints, and structures
  * Use vivid descriptions, examples, and elaborations
  * Avoid repetition; ensure each entry stands out
  * Maintain coherence and logical flow
  * Seamlessly integrate the following {row['pii_type']} values exactly as provided into the text: {pii_values_markdown}
  * Identify appropriate locations within the document to naturally incorporate these values
  * Provide context for each {row['pii_type']}, explaining its relevance to the {row['document_type']}
  * Ensure the {row['pii_type']} values fit grammatically and contextually within the surrounding text
  * Maintain the overall structure and coherence of the {row['document_type']}
Aim to create a rich, detailed, and engaging {row['document_type']} that showcases creativity and diversity while seamlessly incorporating the provided {row['pii_type']} values.
"""

    code_snippet = f'''
# Generated Prompt:
create_prompt = """{create_prompt.strip()}"""

# Generate initial documents
results = navigator.generate(prompt=create_prompt, num_records={num_docs_per_context})

results['text_markup'] = results['document_text'].apply(lambda text: add_markup_to_text(text, pii_types_dict))
    '''

    # Display the code snippet in Streamlit
    st.code(code_snippet, language="python")


class PIIGenerator:
    def __init__(self, locales=["en_US"]):
        self.faker = Faker(locales)
        self.locales = locales
        self.pii_types = {}

    def add_faker_generator(self, name, method, *args, **kwargs):
        """
        Adds a Faker-based generator for a specific PII type.
        """
        self.pii_types[name] = (
            self._generate_faker_data,
            (method, args, kwargs),
            "generator",
        )

    def add_custom_list(self, name, custom_list):
        """
        Adds a custom list of values for a specific PII type.
        """
        self.pii_types[name] = (itertools.cycle, (custom_list,), "list")

    def _generate_faker_data(self, method, args, kwargs):
        """
        Internal method to generate data using Faker.
        """
        result = getattr(self.faker, method)(*args, **kwargs)
        if isinstance(result, tuple):
            # Concatenate tuple elements into a single string
            return " ".join(map(str, result))
        else:
            return str(result)

    def get_pii_generator(self, name, count=1):
        """
        Retrieves a generator for the specified PII type.
        """
        if name in self.pii_types:
            func, args, _ = self.pii_types[name]
            for _ in range(count):
                yield func(*args)
        else:
            raise ValueError(f"PII type '{name}' not defined.")

    def sample(self, name, sample_size=1):
        """
        Samples data for the specified PII type without exhausting the generator.
        """
        if name not in self.pii_types:
            raise ValueError(f"PII type '{name}' not defined.")

        _, args, type = self.pii_types[name]

        if type == "generator":
            # For generators, generate a larger pool then sample, as direct sampling is not possible
            pool_size = max(
                10, sample_size
            )  # Ensure at least 10 or the requested sample size
            pool = [
                next(self.get_pii_generator(name, 1)).replace("\n", " ")
                for _ in range(pool_size)
            ]
            return random.sample(pool, k=sample_size)
        elif type == "list":
            # Directly sample from the list
            return random.sample(args[0], k=sample_size)

    def get_all_pii_generators(self):
        """
        Returns a dictionary of all PII types with their corresponding generators.
        """
        return {name: self.get_pii_generator(name) for name in self.pii_types}

    def print_examples(self):
        """
        Prints two examples of each PII type.
        """
        print("Current Locales:", self.locales)

        for name, _ in self.pii_types.items():
            examples = list(self.sample(name, sample_size=2))
            print(f"Examples of {name}: {examples}")
