#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.11"
# dependencies = ["shapely", "pypopulation", "pycountry_convert"]
# ///

import json
import os
import random

import pypopulation
import pycountry_convert as pc

from shapely.geometry import Point, shape


def random_point_in_country(country_name: str) -> tuple:
    """
    Getting coordinates of a random point inside a country chosen by its name
    Parameters:
    ==========
    :param country_name: a country name
    Returns:
    ==========
    :return: a tuple with coordinates y and x
    """

    with open(os.path.join(os.path.dirname(__file__), "countries.geo.json")) as f:
        data = json.loads(f.read())
        #        print(country_name)
        matched_countries = [
            country
            for country in data["features"]
            if country["properties"]["name"] in country_name
        ]
        if not matched_countries:
            print(f"No country found for {country_name}")
            return 0, 0

        country = matched_countries[0]
        feature = shape(country["geometry"])
        minx, miny, maxx, maxy = feature.bounds

        while True:
            pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if feature.contains(pnt):
                return float(pnt.y), float(pnt.x)


def get_countries() -> list[str]:
    with open(os.path.join(os.path.dirname(__file__), "countries.geo.json")) as f:
        data = json.loads(f.read())
        return [
            (country["properties"]["name"], country["id"])
            for country in data["features"]
        ]


def country_to_continent(country_name: str) -> str:
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(
        country_continent_code
    )
    return country_continent_name


# TODO: This uses the countries available in the file, which is not the same as pycountry.countries
# It also is pretty rudimentary and gets the list of coordinates based on the
def main():
    DEFAULT_MERCHANT_COUNT = 10
    MIN_WEIGHT = 1_000_000  # Weight by half a million
    MAX_MERCHANT_COUNT = 500
    print("country,latitude,longitude")
    for country_name, country_id in get_countries():
        population = pypopulation.get_population(country_id)

        population = pypopulation.get_population(country_id)
        continent = country_to_continent(country_name)
        print(country_name, continent)


#        merchant_count = DEFAULT_MERCHANT_COUNT
#        if population:
#            merchant_count = int(population / MIN_WEIGHT * DEFAULT_MERCHANT_COUNT)
#            merchant_count = min(merchant_count, MAX_MERCHANT_COUNT)
#        #    print(f"{country_name}: {merchant_count}")
#        for _ in range(merchant_count):
#            point = random_point_in_country(country_name)
#            print(f"{country_name},{point[0]},{point[1]}")


if __name__ == "__main__":
    main()
