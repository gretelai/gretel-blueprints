"""
Example usage for Python + gretel-client + Tabular LLM Endpoints

Usage:
    pip install -U gretel-client
    gretel configure
    python main.py
"""
from gretel import TabLLMRequest, TabLLMStream, TabLLMStreamError, validate_gretel


PROMPT = """
Generate a mock dataset for users from the Foo company based in France.
  Each user should have the following columns: 
  * first_name: traditional French first names. 
  * last_name: traditional French surnames. 
  * email: formatted as the first letter of their first name followed by their last name @foo.io (e.g., jdupont@foo.io).
  * gender: Male/Female. 
  * city: a city in France. 
  * country: always 'France'.
"""

NUM_RECORDS = 55

MODEL_ID = "gretelai/tabular-v0"


def main():
    validate_gretel()
    request = TabLLMRequest(
        model_id=MODEL_ID, prompt=PROMPT, temperature=0.7, top_k=40, top_p=0.9
    )
    stream = TabLLMStream(request=request, record_count=NUM_RECORDS)

    try:
        for record_dict in stream.iter_stream():
            print(record_dict)
    except TabLLMStreamError as err:
        print(f"Received Tabular LLM Error: {str(err)}")


if __name__ == "__main__":
    main()
