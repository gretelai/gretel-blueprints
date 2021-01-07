# Gretel NER Container Quick Start

This blueprint demonstrates how to launch the Gretel NER container with custom predictors. This assumes you have already
pulled the container image from DockerHub.

**NOTE**: You must set your `GRETEL_API_KEY` in `docker-compose.yml`

The following files are volume mounted into the container(s):

- `config.yml`: Defines all custom predictors that should be loaded for use by the API.

- `codewords.txt`: Is an example phrase list that can be used by the phrase predictor. Each phrase much be on a its own line.

To launch the container and test:

```
docker-compose up --scale ner=1
```

```
curl --request POST \
  --url http://localhost:8000/records/detect_entities \
  --header 'Content-Type: application/json' \
  --data '{"message": "Update: user_id12345678_BAL has completed working on operation ROCKYBALBOA as of 12/28/2020"}'
```

Resulting predictions:

```
[
  {
    "data": {
      "message": "Update: user_id12345678_BAL as completed working on operation ROCKYBALBOA as of 12/28/2020"
    },
    "metadata": {
      "gretel_id": "4a5223dfe38348b6bbf99aae1441c773",
      "fields": {
        "message": {
          "ner": {
            "labels": [
              {
                "start": 8,
                "end": 27,
                "label": "acme/user_id",
                "score": 0.8,
                "source": "acme/user_id",
                "text": "user_id12345678_BAL"
              },
              {
                "start": 62,
                "end": 73,
                "label": "acme/codewords",
                "score": 0.9,
                "source": "acme/codewords",
                "text": "ROCKYBALBOA"
              }
            ]
          }
        }
      },
      "entities": {
        "score_high": [
          "acme/user_id",
          "acme/codewords"
        ],
        "score_med": [],
        "score_low": [],
        "fields_by_entity": {
          "acme/user_id": [
            "message"
          ],
          "acme/codewords": [
            "message"
          ]
        }
      },
      "received_at": "2021-01-05T21:08:47.935139Z"
    }
  }
]
```