# Gretel Navigator Real Time Inference API

**NOTE**: Gretel Navigator API support is now available in the [Gretel Python SDK](https://github.com/gretelai/gretel-python-client) starting with version `0.18.0`. You can find example usage in the `docs/notebooks/navigator-*` Jupyter Notebooks.

This README details the API that powers [Gretel Navigator](https://console.gretel.ai/playground).

In addition to the Gretel Python SDK, this directory contains usage examples with Javascript.

# General Operation

Gretel Navigator real-time inferrence utilizes lightweight streams to generate and retrieve generated tabular data. Compared to other LLM streaming mechanisms, such as Server-Sent Events (SSE), Gretel does not stream back individual tokens. The Navigator infrastructure automatically utilizes multiple LLMs and Agents to generate tokens and re-assemble them into tabular data. Client-side applications can then poll specific REST API endpoints to retrieve table rows as they are generated and assembled. By using a lightweight polling mechanism, Navigaor can be more flexibly integrated into apps that may not have streaming capabilities.

There are two main steps to utilize the Navigator APIs:

1. First, a request is made that contains the prompt and LLM parameters. This request will return immediately with a _inference stream ID_. This ID represents a lightweight stream that can be itereated using a second endpoint. Behind the scenes, this request will be processed and tabular records will be generated and written to the stream.

2. Second, API calls are made to iterate the _inference stream_ to read tabular records (in JSON format) until generation is complete.

All API calls are made using `HTTP POST` requests.

## Authentication

In order to use this API you will need a [Gretel API key](https://console.gretel.ai/users/me/key). This API key will need to be included in all API call [HTTP headers](https://api.docs.gretel.ai/). If you are using the Javascript examples, you can set the API key as an enviroment variable. If you are using the Python SDK, we highly suggest you install the [Gretel Python SDK](https://docs.gretel.ai/guides/environment-setup/cli-and-sdk) and run `gretel configure` to configure API authentication and use the SDK directly for interacting with the API.

## Rate Limiting

Stream creation (data generation requests) are rate limited to approximately 3 requests every 15 seconds. On average, you may create a stream every 5 seconds. If this limit is exceeded, the stream creation endpoint will return HTTP 403 errors.

# API Usage

These sections detail the specifics of making API calls to Navigator inference endpoints. All requests are made with `HTTP POST` requests.

## Make a Data Generation Request

To submit a prompt to Navigator, make the following call. This example uses one of the sample prompts from the Model Playground. Please update the prompt to suit your needs.

```
POST https://api.gretel.cloud/v1/inference/tabular/stream

{
  "model_id": "gretelai/auto",
  "prompt": "Generate a mock dataset for users from the Foo company based in France.\n  Each user should have the following columns: \n  * first_name: traditional French first names. \n  * last_name: traditional French surnames. \n  * email: formatted as the first letter of their first name followed by their last name @foo.io (e.g., jdupont@foo.io).\n  * gender: Male/Female. \n  * city: a city in France. \n  * country: always 'France'.",
  "params": {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9
  },
  "num_rows": 10,
}
```

If the call is successful, you will get a response like this:

```
{"stream_id": "inference_stream_2YoHrVmtacdqvtjsMGqYCCrXNPa"}
```

Save this `stream_id` as you will need it to access the tabular records.

If you are rate-limited, the API will return a `HTTP 403`.

## Iterate Generated Records

With a `stream_id`, you may now start to iterate the stream to retrieve generated records.

When reading from the stream, there are some parameters to pay attention to:

- `stream_id`: This will be the saved value from the previous response from stream creation.
- `num_records`: The number of data records to retrieve from the stream. Data records here may be actual synthetic data or logging messages from the backend generation process. A good starting value for this is 5 or 10.
- `iterator`: If provided, this represents where in the stream to continue reading from. Navigator streams can only be read forward (they are one way). If this value is an empty string, records will be read from the start (head) of the stream.

API calls to read from a stream look like this:

```
POST https://api.gretel.cloud/v1/inference/tabular/stream/iterate

{
  "stream_id": "inference_stream_2YoHrVmtacdqvtjsMGqYCCrXNPa",
  "iterator": "",
  "count": 10
}
```

Responses from this API call have the following structure (in JSON):

- `data`: This is an array of JSON objects that contain actual data from the stream. It may contain logging messages or actual synthetic data. Each JSON object within the array has the following keys:
  - `record_id`: A unique record ID for the data
  - `data`: The actual contents of the record from the stream.
  - `data_type`: The type of data returned. **NOTE:** You will want to switch on this value when processing responses.
  - `error_code`: If the `data_type` is `logger.error`, this value will be set to a HTTP error code. This helps provide context on what kind of error was encountered. We use HTTP error codes (4xx, 5xx) as they are an established pattern already.
- `next_iterator`: This value should be used for the `iterator` value in the next API call in order to continue reading from the stream.
- `stream_state`: This object contains some metadata about the stream, it has the following keys:

  - `created_at`: When the stream was created.
  - `expires_at`: When this stream will expire. By default, it's approximately 10 minutes after creation. The stream will auto-delete around this time. It is the client's responsibility to read all records prior to deletion. If a request is made to an expired stream, the API will return a `HTTP 404`.
  - `status`: One of: `open` or `closed`. If `open`, then data is still being generated and written into the stream. If `closed`, then no new data will be written into the stream.
  - `record_count`: The total number of records in the stream. This includes log messages and generated synthetic data records.

  **Important Notes**

  - We suggest sleeping a second or so between calls to the `/iterate` endpoint.
  - If the stream state is `open` AND the `data` array is _empty_, this means data is till being generated. Continue iterating until records are received.
  - Pay attention to the `data[].data_type` value. It should be one of `logger.*` or `TabularResponse`. For `logger.*` values, these are informative messages that you can handle accordingly. The `TabularResponse` messages contain actual generated synthetic data.
  - If you encounter a `logger.error` message in the stream, **this will be the last item in the stream**, this record will contain why generation was stopped.
  - If the stream state is `closed` and the `data` array is _empty_, you are at the end of the stream.

### Step-by-Step Examples

Your first API call will be to read from the "head" of the stream:

```
POST https://api.gretel.cloud/v1/inference/tabular/stream/iterate

{
  "stream_id": "inference_stream_2YoHrVmtacdqvtjsMGqYCCrXNPa",
  "iterator": "",
  "count": 10
}
```

Let's look at the response:

```json
{
  "data": [
    {
      "record_id": "jY6VVoyRBfM",
      "data": "Creating execution plan from prompt",
      "data_type": "logger.info",
      "error_code": null
    }
  ],
  "next_iterator": "jY6VVoyRBfM",
  "stream_state": {
    "created_at": "2023-11-28T15:26:50.423285Z",
    "expires_at": "2023-11-28T15:36:50.423285Z",
    "status": "open",
    "record_count": 1
  }
}
```

The only record returned is a logging message that the execution plan is being created. Note that we have a new `next_iterator` value that should be used on the next API call:

```
POST https://api.gretel.cloud/v1/inference/tabular/stream/iterate

{
  "stream_id": "inference_stream_2YoHrVmtacdqvtjsMGqYCCrXNPa",
  "iterator": "jY6VVoyRBfM",
  "count": 10
}
```

Response:

```json
{
  "data": [],
  "next_iterator": "jY6VVoyRBfM",
  "stream_state": {
    "created_at": "2023-11-28T15:26:50.423285Z",
    "expires_at": "2023-11-28T15:36:50.423285Z",
    "status": "open",
    "record_count": 1
  }
}
```

No data was received. That's OK! Navigator is busy generating your data. Our `next_iterator` should not have changed in this case, so continue to poll the API. You may receive additional `logger.info` messages, handle them appropiately. Continue to poll until you see `TabularResponse` data:

```json
{
  "data": [
    {
      "record_id": "79V77QvnBUx",
      "data": {
        "table_headers": [
          "first_name",
          "last_name",
          "email",
          "gender",
          "city",
          "country"
        ],
        "table_data": [
          {
            "first_name": "Louis",
            "last_name": "Lefebvre",
            "email": "llefebvre@foo.io",
            "gender": "Male",
            "city": "Paris",
            "country": "France"
          }
        ]
      },
      "data_type": "TabularResponse",
      "error_code": null
    }
  ],
  "next_iterator": "79V77QvnBUx",
  "stream_state": {
    "created_at": "2023-11-28T15:26:50.423285Z",
    "expires_at": "2023-11-28T15:36:50.423285Z",
    "status": "open",
    "record_count": 3
  }
}
```

Now we start to see `TabularResponse` data. These will contain the table headers and concrete rows/records that were generated. Parse these from the response and use them as needed in your app!

As we continue to iterate we will see the `stream_state` change to `closed`:

```json
{
  "data": [
    {
      "record_id": "Og6ooNPopT9",
      "data": {
        "table_headers": [
          "first_name",
          "last_name",
          "email",
          "gender",
          "city",
          "country"
        ],
        "table_data": [
          {
            "first_name": "Clara",
            "last_name": "Meunier",
            "email": "cmeunier@foo.io",
            "gender": "Female",
            "city": "Quimper",
            "country": "France"
          }
        ]
      },
      "data_type": "TabularResponse",
      "error_code": null
    },
    {
      "record_id": "83K77jM7mIZ",
      "data": {
        "table_headers": [
          "first_name",
          "last_name",
          "email",
          "gender",
          "city",
          "country"
        ],
        "table_data": [
          {
            "first_name": "Gabriel",
            "last_name": "Lemoine",
            "email": "glemoine@foo.io",
            "gender": "Male",
            "city": "Troyes",
            "country": "France"
          }
        ]
      },
      "data_type": "TabularResponse",
      "error_code": null
    },
    {
      "record_id": "99Y77kDEPcg",
      "data": "Synthetic data generation completed.",
      "data_type": "logger.info",
      "error_code": null
    }
  ],
  "next_iterator": "99Y77kDEPcg",
  "stream_state": {
    "created_at": "2023-11-28T15:26:50.423285Z",
    "expires_at": "2023-11-28T15:36:50.423285Z",
    "status": "closed",
    "record_count": 53
  }
}
```

As you can see, there are 3 messages in this iteration response. The last one is a `logger.info` indicating generation is complete. If we poll the endpoint again we will eventually see:

```json
{
  "data": [],
  "next_iterator": "99Y77kDEPcg",
  "stream_state": {
    "created_at": "2023-11-28T15:26:50.423285Z",
    "expires_at": "2023-11-28T15:36:50.423285Z",
    "status": "closed",
    "record_count": 53
  }
}
```

At this point the `status` is `closed` and there is no more data. You are at the end of the stream!
