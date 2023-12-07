import json
from pathlib import Path

import jsonschema
from jsonschema import validate

card_schema = {
    "type": "object",
    "properties": {
        "gtmId": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "cardType": {"type": "string", "enum": ["Console", "Notebook"]},
        "imageName": {"type": "string"},
        "tag": {
            "type": "string",
            "enum": ["New", "Beta", "Preview", "Popular", "Deprecated","Labs"],
        },
        "modelType": {
            "type": "string",
            "enum": [
                "actgan",
                "amplify",
                "classify",
                "ctgan",
                "evaluate",
                "gpt_x",
                "lstm",
                "synthetics",
                "timeseries_dgan",
                "transform",
                "transform_v2",
                "tabular_dp",
            ],
        },
        "modelCategory": {
            "type": "string",
            "enum": [
                "synthetics",
                "transform",
                "classify",
                "evaluate",
            ],
        },
        "defaultConfig": {"type": "string"},
        "sampleDataset": {
            "type": "object",
            "properties": {
                "fileName": {"type": "string"},
                "description": {"type": "string"},
                "records": {"type": "number"},
                "fields": {"type": "number"},
                "trainingTime": {"type": "string"},
                "bytes": {"type": "number"},
            },
            "required": [
                "fileName",
                "description",
                "records",
                "fields",
                "trainingTime",
                "bytes",
            ],
        },
        "detailsFileName": {"type": "string"},
        "button1": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "link": {"type": "string"}},
        },
        "button2": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "link": {"type": "string"}},
        },
    },
    "required": ["gtmId", "title", "description", "cardType", "imageName"],
}


def test_use_cases():
    usecases_JSON = Path(__file__).parent / "use_cases/gretel.json"
    is_valid = validate_JSON(usecases_JSON.open())
    assert is_valid

    titles = []
    gtm_ids = []

    cards_data = json.load(usecases_JSON.open())
    for card in cards_data["cards"]:
        card_is_valid = validate_use_case_JSON(card)
        assert card_is_valid
        gtm_ids.append(card.get("gtmId"))
        titles.append(card.get("title"))
        validate_images_exist(card.get("imageName"))
        validate_config_file_exists(card.get("defaultConfig"))
        validate_details_file_exists(card.get("detailsFileName"))

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
        print("error format:", err)
        return False
    return True


def validate_unique(field_values):
    assert len(field_values) == len(set(field_values))


def validate_images_exist(image_name):
    dir_path = "use_cases/images"
    split_name = image_name.split(".")
    two_x = split_name[0] + "@2x." + split_name[1]
    three_x = split_name[0] + "@3x." + split_name[1]

    assert (Path(__file__).parent / dir_path / image_name).is_file()
    assert (Path(__file__).parent / dir_path / two_x).is_file()
    assert (Path(__file__).parent / dir_path / three_x).is_file()


def validate_config_file_exists(config_path):
    if config_path:
        assert (Path(__file__).parent / config_path).is_file()


def validate_details_file_exists(fileName):
    dir_path = "use_cases/details"
    if fileName:
        assert (Path(__file__).parent / dir_path / fileName).is_file()
