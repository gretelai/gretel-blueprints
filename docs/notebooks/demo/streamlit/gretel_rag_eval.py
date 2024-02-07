# streamlit_app.py
import streamlit as st
import pandas as pd
from utils import (
    initialize_app,
    display_navigation,
    render_current_page
)

def main():
    # Apply custom CSS and display header
    initialize_app()
    
    # Display Navigation in the sidebar
    display_navigation()
    
    # Render the current page content based on navigation
    render_current_page()

if __name__ == "__main__":
    main()
