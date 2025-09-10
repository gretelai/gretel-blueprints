import os

import gretel_client.data_designer.columns as C
import gretel_client.data_designer.params as P
import pandas as pd
from gretel_client.data_designer import DataDesigner
from gretel_client.data_designer.preview import PreviewResults
from gretel_client.navigator_client import Gretel

# The Gretel object is the SDK's main entry point for interacting with Gretel's API.
gretel = Gretel(api_key="prompt")

country_seed_data = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "country-seed-data.csv")
)


def create_cardholder_aidd() -> DataDesigner:
    # Create a new Data Designer instance for Cardholder dataset
    cardholder_aidd = gretel.data_designer.new(model_suite="apache-2.0")
    cardholder_aidd.with_seed_dataset(
        dataset=country_seed_data,
        sampling_strategy="shuffle",
        with_replacement=True,
    )
    cardholder_aidd.with_person_samplers(
        {
            "card_holder": P.PersonSamplerParams(),
        },
    )

    # Add cardholder_id as a unique identifier
    cardholder_aidd.add_column(
        name="age",
        type="expression",
        expr="{{ card_holder.age }}",
    )

    # Add country
    cardholder_aidd.add_column(
        name="country",
        type="expression",
        expr="{{ _country }}",
    )
    # Add credit limit
    cardholder_aidd.add_column(
        C.SamplerColumn(
            name="credit_limit",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=1000, high=50000),
            convert_to="int",
        )
    )

    # Add credit score
    cardholder_aidd.add_column(
        C.SamplerColumn(
            name="credit_score",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=300, high=850),
            convert_to="int",
        )
    )

    # Add historical spend statistics
    cardholder_aidd.add_column(
        C.SamplerColumn(
            name="avg_1day_spend",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=10, high=500),
            convert_to="float",
        )
    )

    cardholder_aidd.add_column(
        C.SamplerColumn(
            name="avg_7day_spend",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=50, high=2000),
            convert_to="float",
        )
    )

    cardholder_aidd.add_column(
        C.SamplerColumn(
            name="avg_30day_spend",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=200, high=8000),
            convert_to="float",
        )
    )

    cardholder_aidd.add_column(
        C.SamplerColumn(
            name="txn_count_30days",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=5, high=100),
            convert_to="int",
        )
    )

    # Add risk flags
    cardholder_aidd.add_column(
        C.SamplerColumn(
            name="prior_fraud_flag",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["Yes", "No"], weights=[0.05, 0.95]  # 5% have prior fraud
            ),
        )
    )

    cardholder_aidd.add_column(
        C.SamplerColumn(
            name="high_risk_location",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["Yes", "No"], weights=[0.1, 0.9]  # 10% in high risk locations
            ),
        )
    )

    cardholder_aidd.validate()

    return cardholder_aidd


def intl_merchant_aidd() -> DataDesigner:
    # Create a new Data Designer instance for Merchant dataset
    merchant_aidd = gretel.data_designer.new(model_suite="apache-2.0")
    merchant_aidd.with_seed_dataset(
        dataset=country_seed_data,
        sampling_strategy="shuffle",
        with_replacement=True,
    )

    # Add merchant_id as a unique identifier
    merchant_aidd.add_column(
        C.SamplerColumn(
            name="merchant_id", type=P.SamplerType.UUID, params=P.UUIDSamplerParams()
        )
    )

    # Add merchant name using LLM
    merchant_aidd.add_column(
        C.LLMTextColumn(
            name="merchant_name",
            prompt=(
                "Generate a realistic merchant name for a business. It should be a company name that could exist in the real world. "
                "The merchant will be located in {{ country }} , and in the category/subcategory of {{ merchant_category }}/{{ merchant_subcategory }}"
                "Respond with only the merchant name, no other text."
            ),
            system_prompt=(
                "You are a helpful assistant that generates realistic merchant names. "
                "You respond with only the merchant name, no other text."
            ),
        )
    )

    # Add location coordinates
    merchant_aidd.add_column(
        name="latitude",
        type="expression",
        expr="{{ _latitude }} ",
    )

    merchant_aidd.add_column(
        name="longitude",
        type="expression",
        expr="{{ _longitude }} ",
    )

    merchant_aidd.add_column(
        name="country",
        type="expression",
        expr="{{ _country }} ",
    )

    # Add merchant category
    merchant_aidd.add_column(
        C.SamplerColumn(
            name="merchant_category",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=[
                    "retail",
                    "food_beverage",
                    "travel",
                    "entertainment",
                    "healthcare",
                    "automotive",
                    "professional_services",
                ]
            ),
        )
    )

    # Add subcategory based on main category
    merchant_aidd.add_column(
        C.SamplerColumn(
            name="merchant_subcategory",
            type=P.SamplerType.SUBCATEGORY,
            params=P.SubcategorySamplerParams(
                category="merchant_category",
                values={
                    "retail": [
                        "electronics",
                        "clothing",
                        "home_garden",
                        "sports",
                        "jewelry",
                    ],
                    "food_beverage": [
                        "restaurants",
                        "fast_food",
                        "grocery",
                        "coffee_shops",
                        "bars",
                    ],
                    "travel": [
                        "hotels",
                        "airlines",
                        "car_rental",
                        "travel_agencies",
                        "cruises",
                    ],
                    "entertainment": [
                        "movies",
                        "gaming",
                        "sports_events",
                        "concerts",
                        "museums",
                    ],
                    "healthcare": [
                        "hospitals",
                        "pharmacies",
                        "dental",
                        "vision",
                        "specialists",
                    ],
                    "automotive": [
                        "dealerships",
                        "repair_shops",
                        "gas_stations",
                        "parts_stores",
                        "car_wash",
                    ],
                    "professional_services": [
                        "legal",
                        "accounting",
                        "consulting",
                        "real_estate",
                        "insurance",
                    ],
                },
            ),
        )
    )

    # Add global risk score
    merchant_aidd.add_column(
        C.SamplerColumn(
            name="global_risk_score",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=1, high=100),
            convert_to="int",
        )
    )

    merchant_aidd.validate()
    return merchant_aidd


def bank_aidd() -> DataDesigner:
    # Create a new Data Designer instance for Bank dataset
    bank_aidd = gretel.data_designer.new(model_suite="apache-2.0")
    bank_aidd.with_seed_dataset(
        dataset=country_seed_data,
        sampling_strategy="shuffle",
        with_replacement=True,
    )

    # Add bank_id as a unique identifier
    bank_aidd.add_column(
        C.SamplerColumn(
            name="bank_id", type=P.SamplerType.UUID, params=P.UUIDSamplerParams()
        )
    )

    # Add bank name using LLM
    bank_aidd.add_column(
        C.LLMTextColumn(
            name="bank_name",
            prompt=(
                "Generate a realistic bank name. It should be a financial institution name that could exist in the real world. "
                "It will be located in the region {{ region }} and the country {{ country }}"
                "Respond with only the bank name, no other text."
            ),
            system_prompt=(
                "You are a helpful assistant that generates realistic bank names. "
                "You respond with only the bank name, no other text."
            ),
        )
    )

    bank_aidd.add_column(
        name="country",
        type="expression",
        expr="{{ _country }} ",
    )

    # TODO: Should we set the region in the country seed data?
    bank_aidd.add_column(
        C.LLMTextColumn(
            name="region",
            prompt=(
                'Given the following regions: "North America", "Europe", "Asia Pacific", "Latin America", '
                '"Middle East", "Africa"'
                "Oceania"
                "Respond with only the region name, no other text."
                "Country: {{ country }}"
            ),
            system_prompt=(
                "You are a helpful assistant that classifies countries by region. "
                "You respond with only the region, no other text."
            ),
        )
    )

    # Add risk score
    bank_aidd.add_column(
        C.SamplerColumn(
            name="risk_score",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=1, high=100),
            convert_to="int",
        )
    )

    # Add market share percentage
    bank_aidd.add_column(
        C.SamplerColumn(
            name="market_share_percent",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=0.1, high=25.0),
            convert_to="float",
        )
    )

    # Add compliance flags
    bank_aidd.add_column(
        C.SamplerColumn(
            name="aml_compliance_flag",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["compliant", "under_review", "non_compliant"],
                weights=[0.85, 0.1, 0.05],
            ),
        )
    )

    bank_aidd.add_column(
        C.SamplerColumn(
            name="kyc_compliance_flag",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["compliant", "under_review", "non_compliant"],
                weights=[0.9, 0.08, 0.02],
            ),
        )
    )

    bank_aidd.add_column(
        C.SamplerColumn(
            name="regulatory_oversight_flag",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["none", "low", "medium", "high"], weights=[0.3, 0.4, 0.2, 0.1]
            ),
        )
    )

    bank_aidd.validate()
    return bank_aidd


def create_combined_cross_join(
    bank_preview: PreviewResults, cardholder_preview: PreviewResults
) -> pd.DataFrame:
    # Use previews to create seeds for dependent columns
    bank_seed_data = bank_preview.dataset.df
    cardholder_seed_data = cardholder_preview.dataset.df

    # Option 2: Cross join (all combinations)
    bank_seed_data_temp = bank_seed_data.copy()
    cardholder_seed_data_temp = cardholder_seed_data.copy()
    bank_seed_data_temp["key"] = 1
    cardholder_seed_data_temp["key"] = 1
    combined_cross_join = bank_seed_data_temp.merge(
        cardholder_seed_data_temp, on="key"
    ).drop("key", axis=1)
    return combined_cross_join


def card_aidd(combined_cross_join: pd.DataFrame) -> DataDesigner:
    # Create a new Data Designer instance for Card dataset
    card_aidd = gretel.data_designer.new(model_suite="apache-2.0")
    card_aidd.with_seed_dataset(
        combined_cross_join, sampling_strategy="shuffle", with_replacement=True
    )

    # Add card_id as a unique identifier
    card_aidd.add_column(
        C.SamplerColumn(
            name="card_id", type=P.SamplerType.UUID, params=P.UUIDSamplerParams()
        )
    )

    # Add cardholder_id reference
    card_aidd.add_column(
        name="cardholder_id_fk",
        type="expression",
        expr="{{ cardholder_id }} ",
    )

    # Add issuer bank ID
    card_aidd.add_column(
        name="bank_id_fk",
        type="expression",
        expr="{{ bank_id }} ",
    )

    # Add card type
    card_aidd.add_column(
        C.SamplerColumn(
            name="card_type",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["credit", "debit", "prepaid"],
                weights=[0.6, 0.35, 0.05],  # Most are credit cards
            ),
        )
    )

    # Add expiry date (simplified as months from now)
    card_aidd.add_column(
        C.SamplerColumn(
            name="expiry_months_from_now",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=1, high=60),
            convert_to="int",
        )
    )

    # Add CVV check result
    card_aidd.add_column(
        C.SamplerColumn(
            name="cvv_check_result",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["pass", "fail", "not_provided"], weights=[0.85, 0.1, 0.05]
            ),
        )
    )

    # Add activation date (simplified as days ago)
    card_aidd.add_column(
        C.SamplerColumn(
            name="activation_days_ago",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=1, high=3650),  # Up to 10 years
            convert_to="int",
        )
    )

    # Add card status
    card_aidd.add_column(
        C.SamplerColumn(
            name="card_status",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["active", "blocked", "suspended"], weights=[0.92, 0.06, 0.02]
            ),
        )
    )

    card_aidd.validate()
    return card_aidd


def transaction_aidd(
    card_preview: PreviewResults, merchant_preview: PreviewResults
) -> DataDesigner:
    """Create Transaction dataset linking Cards and Merchants through transaction events"""
    # Create a new Data Designer instance for Transaction dataset
    transaction_aidd = gretel.data_designer.new(model_suite="apache-2.0")

    # Create seed data from card and merchant previews for cross join
    card_seed_data = card_preview.dataset.df
    merchant_seed_data = merchant_preview.dataset.df

    # Create cross join for card-merchant combinations
    card_seed_data["key"] = 1
    merchant_seed_data["key"] = 1
    card_merchant_cross_join = card_seed_data.merge(merchant_seed_data, on="key").drop(
        "key", axis=1
    )

    transaction_aidd.with_seed_dataset(
        dataset=card_merchant_cross_join,
        sampling_strategy="shuffle",
        with_replacement=True,
    )

    # Add transaction_id as a unique identifier
    transaction_aidd.add_column(
        C.SamplerColumn(
            name="txn_id", type=P.SamplerType.UUID, params=P.UUIDSamplerParams()
        )
    )

    # Add card_id reference from seed
    transaction_aidd.add_column(
        name="card_id",
        type="expression",
        expr="{{ card_id }}",
    )

    # Add merchant_id reference from seed
    transaction_aidd.add_column(
        name="merchant_id",
        type="expression",
        expr="{{ merchant_id }}",
    )

    # Add transaction amount
    transaction_aidd.add_column(
        C.SamplerColumn(
            name="amount_usd",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=1.0, high=5000.0),
            convert_to="float",
        )
    )

    # Add timestamp (simplified as days ago)
    transaction_aidd.add_column(
        C.SamplerColumn(
            name="days_ago",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=0, high=365),
            convert_to="int",
        )
    )

    # Add MCC (Merchant Category Code) - derived from merchant category
    transaction_aidd.add_column(
        C.SamplerColumn(
            name="mcc",
            type=P.SamplerType.SUBCATEGORY,
            params=P.SubcategorySamplerParams(
                category="merchant_category",
                values={
                    "retail": [
                        "5311",
                        "5651",
                        "5712",
                        "5941",
                        "5944",
                    ],  # Electronics, clothing, home, etc.
                    "food_beverage": [
                        "5812",
                        "5814",
                        "5411",
                        "5732",
                        "5813",
                    ],  # Restaurant, fast food, etc.
                    "travel": [
                        "7011",
                        "4511",
                        "7512",
                        "4722",
                        "4411",
                    ],  # Hotels, airlines, car rental, etc.
                    "entertainment": [
                        "7832",
                        "7993",
                        "7941",
                        "7922",
                        "8412",
                    ],  # Movies, gaming, sports, etc.
                    "healthcare": [
                        "8062",
                        "5912",
                        "8021",
                        "8043",
                        "8011",
                    ],  # Hospitals, pharmacy, dental, etc.
                    "automotive": [
                        "5511",
                        "7538",
                        "5541",
                        "5533",
                        "7542",
                    ],  # Dealerships, repair, gas, etc.
                    "professional_services": [
                        "8111",
                        "8931",
                        "8999",
                        "6513",
                        "6300",
                    ],  # Legal, accounting, etc.
                },
            ),
        )
    )

    # Add transaction channel
    transaction_aidd.add_column(
        C.SamplerColumn(
            name="channel",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["chip", "swipe", "online", "contactless"],
                weights=[0.45, 0.25, 0.25, 0.05],
            ),
        )
    )

    # Add currency
    transaction_aidd.add_column(
        C.SamplerColumn(
            name="currency",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=[
                    "USD",
                    "EUR",
                    "GBP",
                    "CAD",
                    "JPY",
                    "AUD",
                ],  # TODO: More currency types?
                weights=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05],
            ),
        )
    )

    # Add conversion rate (1.0 for USD, varied for others)
    # TODO: Should this be mapped?
    transaction_aidd.add_column(
        C.SamplerColumn(
            name="conversion_rate",
            type=P.SamplerType.UNIFORM,
            params=P.UniformSamplerParams(low=0.5, high=2.0),
            convert_to="float",
        )
    )

    # Add fraud flag (binary)
    transaction_aidd.add_column(
        C.LLMTextColumn(
            name="is_fraud",
            prompt=(
                "Say whether or not a transaction is fraudulent, returning either only a 1 or a 0, based on the provided data"
                "Currency: {{currency}}, Channel: {{channel}}, Amount_usd: {{amount_usd}}"
                "Respond with only the bank name, no other text."
            ),
            system_prompt=(
                "You are a helpful assistant that determines if a transaction is fraudulent. "
                "You respond with only a 1 or a 0."
            ),
        )
    )

    # Add device context
    transaction_aidd.add_column(
        C.SamplerColumn(
            name="device_id", type=P.SamplerType.UUID, params=P.UUIDSamplerParams()
        )
    )

    transaction_aidd.add_column(
        C.SamplerColumn(
            name="session_id", type=P.SamplerType.UUID, params=P.UUIDSamplerParams()
        )
    )

    transaction_aidd.add_column(
        C.SamplerColumn(
            name="app_version",
            type=P.SamplerType.CATEGORY,
            params=P.CategorySamplerParams(
                values=["1.0.0", "1.1.0", "1.2.0", "2.0.0", "2.1.0", "2.2.0"],
                weights=[0.1, 0.1, 0.15, 0.25, 0.25, 0.15],
            ),
        )
    )

    # Add network context
    # TODO: Is this a stupid thing to generate
    transaction_aidd.add_column(
        C.LLMTextColumn(
            name="ip_address",
            prompt=(
                "Given the following country, generate a valid corresponding public IP address"
                "Respond with only the IP address, no other text."
                "Country: {{ country }}"
            ),
            system_prompt=(
                "You are a helpful assistant that creates IP addresses based on a country. "
                "You respond with only the ip_address, no other text."
            ),
        )
    )

    # TODO: Should this be a generated thing?
    transaction_aidd.add_column(
        C.LLMTextColumn(
            name="user_agent_hashed",
            prompt=(
                "Generate a realistic SHA-256 hash string (64 hexadecimal characters) "
                "that could represent a hashed user agent string. "
                "Respond with only the hash, no other text."
            ),
            system_prompt=(
                "You are a helpful assistant that generates realistic hash strings. "
                "You respond with only a 64-character hexadecimal hash, no other text."
            ),
        )
    )

    transaction_aidd.validate()
    return transaction_aidd


cardholder_preview = create_cardholder_aidd().preview()
print(cardholder_preview)
merchant_preview = intl_merchant_aidd().preview()
print(merchant_preview)
bank_preview = bank_aidd().preview()
combined_cross_join = create_combined_cross_join(
    bank_preview=bank_preview, cardholder_preview=cardholder_preview
)
card_preview = card_aidd(combined_cross_join).preview()
print(card_preview)

# Preview transaction dataset
transaction_preview = transaction_aidd(card_preview, merchant_preview).preview()
print("=== TRANSACTION DATASET PREVIEW ===")
print(transaction_preview)
