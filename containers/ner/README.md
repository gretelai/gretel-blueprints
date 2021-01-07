# Gretel NER Container Quick Start

This blueprint demonstrates how to launch the Gretel NER container with custom predictors. This assumes you have already
pulled the container image from DockerHub.

The following files are volume mounted into the container(s):

- `config.yml`: Defines all custom predictors that should be loaded for use by the API.

- `codewords.txt`: Is a example phrase list that can be used by the phrase predictor. Each phrase much be on a its own line.

To launch the container and test:

```
docker-compose up --scale ner=1
```

```
curl --request POST \
  --url http://localhost:8000/records/detect_entities \
  --header 'Content-Type: application/json' \
  --data '{"message": "Update: user_id12345678_BAL as completed working on operation ROCKYBALBOA as of 12/28/2020"}'
```