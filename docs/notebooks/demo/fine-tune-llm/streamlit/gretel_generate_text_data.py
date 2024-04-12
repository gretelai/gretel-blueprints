import streamlit as st
from utils import (
    initialize_app,
    initialize_gretel_client,
    create_document_types_section,
    configure_pii_generator,
    generate_contextual_tags,
    create_synthetic_dataset,
)


# ----- Main Streamlit App -----
def main():
    # Apply custom CSS and display header
    initialize_app()

    st.markdown(
        '<h1 class="app-title">Synthetic Data Generator for Financial Documents</h1>',
        unsafe_allow_html=True,
    )

    # Initialize Gretel client
    gretel, tabllm = initialize_gretel_client()

    if gretel is not None and tabllm is not None:
        create_document_types_section(gretel, tabllm)

        if "document_types" in st.session_state:
            st.success("Document types section completed successfully.")

            pii_generator = configure_pii_generator()
            pii_type_dict = pii_generator.get_all_pii_generators()

            generate_contextual_tags(
                pii_type_dict, st.session_state.document_types, pii_generator
            )

            if "contextual_tags" in st.session_state:
                create_synthetic_dataset(
                    tabllm, st.session_state.contextual_tags
                )


if __name__ == "__main__":
    main()
