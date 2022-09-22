import json
import os
from pathlib import Path

import jsonschema
import pytest
from jsonschema import validate

card_schema = {
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


def test_use_cases():
    usecases_JSON= (Path(__file__).parent / "use_cases/gretel.json")
    is_valid = validate_JSON(usecases_JSON.open());
    assert is_valid

    titles = []
    gtm_ids = []
    
    cards_data = json.load(usecases_JSON.open())  
    for card in cards_data['cards']:
        card_is_valid = validate_use_case_JSON(card)
        assert card_is_valid
        gtm_ids.append(card['gtmId'])
        titles.append(card['title'])
        validate_images_exist(card['imageName'])

    validate_unique(gtm_ids)
    validate_unique(titles)
  

def validate_JSON(json_data):
    try:
        json.load(json_data)
    except ValueError as err:
        return False
    return True


def validate_use_case_JSON(card):
    try:
        validate(instance=card, schema=card_schema)
    except jsonschema.exceptions.ValidationError as err:
        print('error format:', err)
        return False
    return True


def validate_unique(field_values):
    assert len(field_values) == len(set(field_values))


def validate_images_exist(image_name):
    dir_path = "use_cases/images"
    split_name = image_name.split(".")
    two_x = split_name[0] + "@2x."+  split_name[1]
    three_x = split_name[0] + "@3x."+  split_name[1]
  
    assert os.path.exists((Path(__file__).parent / dir_path / image_name))
    assert os.path.exists((Path(__file__).parent / dir_path / two_x))
    assert os.path.exists((Path(__file__).parent / dir_path / three_x))