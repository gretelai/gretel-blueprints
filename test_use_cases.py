import os
from pathlib import Path

import pytest
import json
import jsonschema
from jsonschema import validate

# https://python-jsonschema.readthedocs.io/en/stable/
cardSchema = {
  "type": "object",
  "properties": {
    "title": {
      "type": "string",
    },
    "description": {
      "type": "string", 
    },
    "imageName": {
      "type": "string",
    },
    "model": {
      "type": "string", 
      "enum": [ "synthetics", "transform", "classify", "ctgan", "amplify", "gpt_x", "evaluate" ]
    },
    "defaultConfig": {
      "type": "string",
    },
    "sampleDataset": {
      "type": "string",
    },
    "docsUrl": {
      "type": "string",
    },
    "tag": {
      "type": "string",
      "enum": [ "New", "Beta", "Preview", "Popular", "Deprecated" ],
    },
    "gtmId": {
      "type": "string",
    }
  },
  "required": ["title", "description", "gtmId", "imageName"],
}

# run with: `pytest test_use_cases.py -s`
def test_use_cases():
  # load file
  usecases_JSON= (Path(__file__).parent / "use_cases/gretel.json")
  # check that the file is valid JSON:
  isValid = validate_JSON(usecases_JSON.open());
  assert isValid

  titles = []
  gtmIds = []
  # validate cards' format
  cardsData = json.load(usecases_JSON.open())  
  for card in cardsData['cards']:
    cardIsValid = validateUseCaseJSON(card)
    assert cardIsValid
    gtmIds.append(card['gtmId'])
    titles.append(card['title'])
    # check that expected images exist
    validateImages(card['title'], card['imageName'])

  # check title and gtmIds are unique across all cards
  validateUnique(gtmIds)
  validateUnique(titles)
  

def validate_JSON(json_data):
  try:
    json.load(json_data)
  except ValueError as err:
      return False
  return True


def validateUseCaseJSON(card):
  try:
    validate(instance=card, schema=cardSchema)
  except jsonschema.exceptions.ValidationError as err:
    print('error format:', err)
    return False
  return True


def validateUnique(fieldValues):
  assert len(fieldValues) == len(set(fieldValues))


def validateImages(useCase, imageName):
  dirPath = "use_cases/images"
  splitName = imageName.split(".")
  twoX = splitName[0] + "@2x."+  splitName[1]
  threeX = splitName[0] + "@3x."+  splitName[1]
  
  baseImageFilePath = (Path(__file__).parent / dirPath / imageName)
  assert os.path.exists(baseImageFilePath)
  assert os.path.exists((Path(__file__).parent / dirPath / twoX))
  assert os.path.exists((Path(__file__).parent / dirPath / threeX))