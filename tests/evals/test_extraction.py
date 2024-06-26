# ruff: noqa: E501
from typing import Sequence

import pytest
from dydantic import create_model_from_schema
from langchain.chat_models import init_chat_model
from langsmith import expect, test, traceable
from typing_extensions import TypedDict

from trustcall import ExtractionInputs, ExtractionOutputs, create_extractor


class Inputs(TypedDict, total=False):
    system_prompt: str
    input_str: str
    current_value: dict
    error_handling: list


@traceable
async def predict_with_model(
    model_name: str, inputs: Inputs, tool_def: dict
) -> ExtractionOutputs:
    messages = [
        (
            "system",
            "Extract the relevant user preferences from the conversation."
            + inputs.get("system_prompt", ""),
        ),
        ("user", inputs["input_str"]),
    ]
    llm = init_chat_model(model_name, temperature=0)
    extractor = create_extractor(llm, tools=[tool_def])
    existing = inputs.get("current_value", {})
    extractor_inputs: dict = {"messages": messages}
    if existing:
        extractor_inputs["existing"] = {tool_def["name"]: existing}
    return await extractor.ainvoke(ExtractionInputs(**extractor_inputs))


class ContainsStr:
    def __init__(self, substr):
        self.substr = substr

    def __eq__(self, other):
        if not isinstance(other, str):
            return False
        return self.substr in other


class AnyStr(str):
    def __init__(self, matches: Sequence[str]):
        self.matches = matches

    def __hash__(self):
        return hash(tuple(self.matches))


extract_user_pref_schema = {
    "name": "extract_user_preferences_v1",
    "description": "A function that extracts personal preferences of the user in the categories 'Points of Interest', 'Navigation and Routing', 'Vehicle Settings and Comfort', 'Entertainment and Media'.",
    "parameters": {
        "$defs": {
            "DanceHall": {
                "properties": {
                    "preferred_dancing_network": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Charging Network",
                    },
                    "preferred_type_of_dancing_while_traveling": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Type Of Charging While Traveling",
                    },
                    "preferred_type_of_dancing_at_everyday_points": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Type Of Charging At Everyday Points",
                    },
                    "dancing_station_onsite_amenities": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Charging Station Onsite Amenities",
                    },
                },
                "title": "DanceHall",
                "type": "object",
            },
            "ClimateControl": {
                "properties": {
                    "preferred_temperature": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Temperature",
                    },
                    "fan_speed_preferences": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Fan Speed Preferences",
                    },
                    "airflow_direction_preferences": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Airflow Direction Preferences",
                    },
                    "seat_heating_preferences": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Seat Heating Preferences",
                    },
                },
                "title": "ClimateControl",
                "type": "object",
            },
            "EntertainmentAndMedia": {
                "properties": {
                    "music": {"$ref": "#/$defs/Music"},
                    "radio_and_podcast": {"$ref": "#/$defs/RadioAndPodcast"},
                },
                "required": ["music", "radio_and_podcast"],
                "title": "EntertainmentAndMedia",
                "type": "object",
            },
            "FireStation": {
                "properties": {
                    "preferred_fire_station": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Fire Station",
                    },
                    "willingness_to_donate": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Willingness To Pay Extra For Green Fuel",
                    },
                    "bingo_expertise": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Price Sensitivity For Fuel",
                    },
                },
                "title": "FireStation",
                "type": "object",
            },
            "GroceryShopping": {
                "properties": {
                    "preferred_supermarket_chain": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Supermarket Chain",
                    },
                    "preference_for_local_markets_farms_or_supermarket": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preference For Local Markets Farms Or Supermarket",
                    },
                },
                "title": "GroceryShopping",
                "type": "object",
            },
            "LightingAndAmbience": {
                "properties": {
                    "interior_lighting_brightness_preferences": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Interior Lighting Brightness Preferences",
                    },
                    "interior_lighting_ambient_preferences": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Interior Lighting Ambient Preferences",
                    },
                    "interior_lighting_color_preferences": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Interior Lighting Color Preferences",
                    },
                },
                "title": "LightingAndAmbience",
                "type": "object",
            },
            "Music": {
                "properties": {
                    "favorite_genres": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Favorite Genres",
                    },
                    "favorite_artists_or_bands": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Favorite Artists Or Bands",
                    },
                    "favorite_songs": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Favorite Songs",
                    },
                    "preferred_music_streaming_service": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Music Streaming Service",
                    },
                },
                "title": "Music",
                "type": "object",
            },
            "NavigationAndRouting": {
                "properties": {
                    "routing": {"$ref": "#/$defs/Routing"},
                    "traffic_and_conditions": {"$ref": "#/$defs/TrafficAndConditions"},
                    "parking": {"$ref": "#/$defs/Parking"},
                },
                "required": ["routing", "traffic_and_conditions", "parking"],
                "title": "NavigationAndRouting",
                "type": "object",
            },
            "OutputFormat": {
                "properties": {
                    "sentence_preference_revealed": {
                        "title": "Sentence Preference Revealed",
                        "type": "string",
                    },
                    "preference": {"title": "Preference", "type": "string"},
                },
                "required": ["sentence_preference_revealed", "preference"],
                "title": "OutputFormat",
                "type": "object",
            },
            "Parking": {
                "properties": {
                    "preferred_parking_type": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Parking Type",
                    },
                    "price_sensitivity_for_paid_parking": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Price Sensitivity For Paid Parking",
                    },
                    "distance_willing_to_walk_from_parking_to_destination": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Distance Willing To Walk From Parking To Destination",
                    },
                    "preference_for_covered_parking": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preference For Covered Parking",
                    },
                    "need_for_handicapped_accessible_parking": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Need For Handicapped Accessible Parking",
                    },
                    "preference_for_parking_with_security": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preference For Parking With Security",
                    },
                },
                "title": "Parking",
                "type": "object",
            },
            "PointsOfInterest": {
                "properties": {
                    "restaurant": {"$ref": "#/$defs/Restaurant"},
                    "fire_station": {"$ref": "#/$defs/FireStation"},
                    "dancing_station": {"$ref": "#/$defs/DanceHall"},
                    "grocery_shopping": {"$ref": "#/$defs/GroceryShopping"},
                },
                "required": [
                    "restaurant",
                    "fire_station",
                    "dancing_station",
                    "grocery_shopping",
                ],
                "title": "PointsOfInterest",
                "type": "object",
            },
            "RadioAndPodcast": {
                "properties": {
                    "preferred_radio_station": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Radio Station",
                    },
                    "favorite_podcast_genres": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Favorite Podcast Genres",
                    },
                    "favorite_podcast_shows": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Favorite Podcast Shows",
                    },
                    "general_news_source": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "General News Source",
                    },
                },
                "title": "RadioAndPodcast",
                "type": "object",
            },
            "Restaurant": {
                "properties": {
                    "favourite_cuisine": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Favourite Cuisine",
                    },
                    "preferred_restaurant_type": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Restaurant Type",
                    },
                    "fast_food_preference": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Fast Food Preference",
                    },
                    "desired_price_range": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Desired Price Range",
                    },
                    "dietary_preference": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Dietary Preference",
                    },
                    "preferred_payment_method": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Preferred Payment Method",
                    },
                },
                "title": "Restaurant",
                "type": "object",
            },
            "Routing": {
                "properties": {
                    "avoidance_of_specific_road_types": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Avoidance Of Specific Road Types",
                    },
                    "priority_for_shortest_time_or_shortest_distance": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Priority For Shortest Time Or Shortest Distance",
                    },
                    "tolerance_for_traffic": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Tolerance For Traffic",
                    },
                },
                "title": "Routing",
                "type": "object",
            },
            "TrafficAndConditions": {
                "properties": {
                    "traffic_information_source_preferences": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Traffic Information Source Preferences",
                    },
                    "willingness_to_take_longer_route_to_avoid_traffic": {
                        "anyOf": [
                            {
                                "items": {"$ref": "#/$defs/OutputFormat"},
                                "type": "array",
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "title": "Willingness To Take Longer Route To Avoid Traffic",
                    },
                },
                "title": "TrafficAndConditions",
                "type": "object",
            },
            "UserPreferences": {
                "properties": {
                    "points_of_interest": {"$ref": "#/$defs/PointsOfInterest"},
                    "routing": {"$ref": "#/$defs/NavigationAndRouting"},
                    "comfort_settings": {"$ref": "#/$defs/VehicleSettingsAndComfort"},
                    "entertainment_and_media": {
                        "$ref": "#/$defs/EntertainmentAndMedia"
                    },
                },
                "required": [
                    "points_of_interest",
                    "routing",
                    "comfort_settings",
                    "entertainment_and_media",
                ],
                "title": "UserPreferences",
                "type": "object",
            },
            "VehicleSettingsAndComfort": {
                "properties": {
                    "climate_control": {"$ref": "#/$defs/ClimateControl"},
                    "lighting_and_ambience": {"$ref": "#/$defs/LightingAndAmbience"},
                },
                "required": ["climate_control", "lighting_and_ambience"],
                "title": "VehicleSettingsAndComfort",
                "type": "object",
            },
        },
        "properties": {
            "pertinent_user_preferences": {"$ref": "#/$defs/UserPreferences"}
        },
        "required": ["pertinent_user_preferences"],
        "title": "PreferencesFunctionOutput",
        "type": "object",
    },
}


french_user_profile = {
    "name": "UserProfile",
    "description": "",
    "parameters": {
        "properties": {
            "name": {
                "description": "The user's name",
                "title": "Name",
                "type": "string",
            },
            "age": {"description": "The user's age", "title": "Age", "type": "integer"},
            "interests": {
                "description": "The user's interests",
                "items": {"type": "string"},
                "title": "Interests",
                "type": "array",
            },
            "relationships": {
                "additionalProperties": {"type": "string"},
                "description": "The user's friends, family, pets,and other relationships.",
                "title": "Relationships",
                "type": "object",
            },
            "other_useful_info": {
                "description": "Any other details you would like to know about the user.",
                "items": {"type": "string"},
                "title": "Other Useful Info",
                "type": "array",
            },
            "ai_instructions": {
                "description": "Prompt instructions that you as the AI assistant should always retain in subsequent conversations.",
                "items": {"type": "string"},
                "title": "Ai Instructions",
                "type": "array",
            },
        },
        "required": [
            "name",
            "age",
            "interests",
            "relationships",
            "other_useful_info",
            "ai_instructions",
        ],
        "type": "object",
    },
}

user_memory_schema = {
    "name": "UserMemory",
    "description": "",
    "parameters": {
        "properties": {
            "name": {
                "description": "The user's name",
                "title": "Name",
                "type": "string",
            },
            "age": {"description": "The user's age", "title": "Age", "type": "integer"},
            "interests": {
                "description": "The user's interests",
                "items": {"type": "string"},
                "title": "Interests",
                "type": "array",
            },
            "relationships": {
                "additionalProperties": {"type": "string"},
                "description": "The user's friends, family, pets,and other relationships.",
                "title": "Relationships",
                "type": "object",
            },
            "ai_instructions": {
                "description": "Prompt instructions that you as the AI assistant should always retain in subsequent conversations.",
                "items": {"type": "string"},
                "title": "Ai Instructions",
                "type": "array",
            },
        },
        "required": ["name", "age", "interests", "relationships", "ai_instructions"],
        "title": "UserMemory",
        "type": "object",
    },
}


@pytest.mark.asyncio_cooperative
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "inputs, tool_def",
    [
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=6748736f-00db-41d7-bbff-c54a6a118dfb>Can you help me find a place to eat, but make sure it's an Italian restaurant, I just love pasta dishes.</user>\n<ai>Certainly! There is a well-rated Italian restaurant called 'La Dolce Vita' about 10 minutes from our current location in Mariani. Would you like directions to there?</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=62e662f9-f404-4a79-8efc-81c8bf57045a>What's interesting around Mariani these days?</user>\n<ai>In Mariani, you can explore the Mariani Junction, nearby tea gardens, and some local heritage sites. Would you like directions to any of these places?</ai>\n<user id=62e662f9-f404-4a79-8efc-81c8bf57045a>Actually, I'm more in the mood for a meal. Could you find me a restaurant around here?</user>\n<ai>Sure, there are several restaurants in the area. Are you looking for any specific type of cuisine or dining experience?</ai>\n<user id=62e662f9-f404-4a79-8efc-81c8bf57045a>Well, I've always had an appreciation for a quiet place with a refined atmosphere, if you get my drift.</user>\n<ai>Understood. I recommend the La Bella Vita, which is a fine dining restaurant nearby known for its elegant setting and gourmet cuisine. Shall I navigate us there?</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=613f0d47-568a-4484-9012-ee5bdedbce04>Navigate to Applebees</user>\n<ai>Setting navigation to Applebees in Sherwood. You'll be there in 15 minutes, and traffic is light. Enjoy your meal!</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=b9059c2f-0fe3-4a66-8c06-225953f2fa11>I could really go for some Italian food right now.</user>\n<ai>I located several Italian restaurants nearby. The closest one is La Bella Vita, 4 kilometers away. Would you like to navigate there?</ai>\n<user id=b9059c2f-0fe3-4a66-8c06-225953f2fa11>Yes, that sounds perfect. Please set it as our destination.</user>\n<ai>Destination set to La Bella Vita. The route is clear with minimal traffic. We should arrive in approximately 10 minutes. Would you like to start the navigation?</ai>\n<user id=b9059c2f-0fe3-4a66-8c06-225953f2fa11>Start the navigation and can you also adjust the temperature to something warmer?</user>\n<ai>Navigation started, and the cabin temperature is now set to a warmer setting. Is there anything else I can assist you with on the way?</ai>\n<user id=b9059c2f-0fe3-4a66-8c06-225953f2fa11>Maybe some Italian music to set the mood?</user>\n<ai>Playing classic Italian music. Enjoy the drive and your meal at La Bella Vita.</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=d655271e-bd1c-4df6-9686-402a002c9dc3>Find me a place to buy fresh apples</user>\n<ai>There are several grocery stores nearby. The closest one is 'Meyve Emporium' which specializes in fresh fruits.</ai>\n<user id=d655271e-bd1c-4df6-9686-402a002c9dc3>No supermarkets, I prefer local market stands</user>\n<ai>Understood. The nearest local market stand with fresh apples is 'Akhisar Farmers' Market', 10 minutes away from our current location.</ai>\n<user id=d655271e-bd1c-4df6-9686-402a002c9dc3>Set navigation to that market</user>\n<ai>Navigation to 'Akhisar Farmers' Market' is now set. Would you like to start the route guidance?</ai>\n<user id=d655271e-bd1c-4df6-9686-402a002c9dc3>Yes, and can you check for traffic on the way?</user>\n<ai>Sure, the route to 'Akhisar Farmers' Market' is clear with no traffic delays. Starting route guidance now.</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=34c2567e-34ec-404a-b8e9-f86b5084ced5>Set the interior lighting to something warm, I can't stand these cool tones anymore.</user>\n<ai>Adjusting the interior lighting to a warm ambience. You should notice a cozier atmosphere now.</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=0a1708e8-1956-4f6b-aa23-f1698ecd6427>Make the lights yellow inside</user>\n<ai>Certainly, changing the interior lighting to yellow now for a warm and cozy atmosphere.</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=34dc9d76-c707-42f4-9cda-8879e06e0570>Adjust interior lighting.</user>\n<ai>Sure, I can change the brightness or color. Which would you like to adjust?</ai>\n<user id=34dc9d76-c707-42f4-9cda-8879e06e0570>Brightness, set it low.</user>\n<ai>Setting the interior lighting to low brightness now.</ai>\n<user id=34dc9d76-c707-42f4-9cda-8879e06e0570>Keep this brightness level for my future rides.</user>\n<ai>I'll ensure the lighting is low for the duration of this ride, but I'm unable to store settings for future trips.</ai>\n<user id=34dc9d76-c707-42f4-9cda-8879e06e0570>That's fine, just make sure it stays this way now.</user>\n<ai>Understood. The low brightness setting will be maintained. If you need any more adjustments, just let me know.</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=82ee6962-2fd4-4add-a2dc-d34ae8e21da1>How's the traffic ahead on this route?</user>\n<ai>The traffic in Mariani is currently flowing smoothly with no delays on your route.</ai>\n<user id=82ee6962-2fd4-4add-a2dc-d34ae8e21da1>Great, can we put on some music? I don't like driving in silence.</user>\n<ai>Of course, would you like to choose a radio station or should I select a playlist for you?</ai>\n<user id=82ee6962-2fd4-4add-a2dc-d34ae8e21da1>Hmm, let's go with a playlist, something upbeat and cheerful, you know the kind that's always on the radio.</user>\n<ai>I have a Pop Hits playlist that fits that description. Shall I play it now?</ai>\n<user id=82ee6962-2fd4-4add-a2dc-d34ae8e21da1>That sounds perfect, pop music always lifts my spirits.</user>\n<ai>Playing the Pop Hits playlist. Enjoy your drive with some uplifting tunes!</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=18b7ac21-fb6c-4503-84f7-8fa04982f896>Start navigation to Applebees</user>\n<ai>Navigating to Applebees on Maple Street, Sherwood. It will take approximately 15 minutes to get there.</ai>\n<user id=18b7ac21-fb6c-4503-84f7-8fa04982f896>Any traffic on the route</user>\n<ai>The route to Applebees is clear with no reported traffic delays.</ai>\n<user id=18b7ac21-fb6c-4503-84f7-8fa04982f896>Play something upbeat</user>\n<ai>Sure, starting your Upbeat Playlist now.</ai>\n<user id=18b7ac21-fb6c-4503-84f7-8fa04982f896>Set temperature to 70 degrees</user>\n<ai>Setting the cabin temperature to 70 degrees.</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=5ff7412a-3140-438b-8430-c96dc0c7da05>I've been craving some good Italian food lately, can you suggest a nice Italian restaurant nearby?</user>\n<ai>Certainly, there's an Italian restaurant called La Cucina, 4 kilometers away with excellent reviews. Shall I navigate there?</ai>\n<user id=5ff7412a-3140-438b-8430-c96dc0c7da05>La Cucina sounds perfect, please turn on the navigation.</user>\n<ai>Navigation to La Cucina is now active. You will arrive in approximately 12 minutes. Enjoy your meal!</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nOnly extract pertinent user preferences, no temporal desires. Extract same user preferences only once independent if the user has said it multiple times.\n</instructions>",
                "current_value": {},
            },
            extract_user_pref_schema,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=1098f610-90a2-4ae7-abb6-096061b4713a>Hello</user>\n<ai>Bonjour Kevin! Let's begin with a simple question to gauge your understanding of French. Please fill in the gap:\n\nJe __________ français. \na) parles\nb) parler\nc) parle</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nRemember that this schema must be useful for prompt engineering an AI assistant.\n</instructions>",
                "current_value": {},
            },
            french_user_profile,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<user id=a4206dff-4858-48ef-8944-7a3a71a77c8b>Hello</user>\n<ai>Bonjour Kevin! Let's start with some basic questions to assess your French proficiency. \n\n**Fill in the blank:** Je mange des pâtes parce que j'ai ____. \n\nA) soif  \nB) faim  \nC) froid  \nD) chaud</ai>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nRemember that this schema must be useful for prompt engineering an AI assistant.\n</instructions>",
                "current_value": {},
            },
            french_user_profile,
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<system>You are a helpful AI assistant</system>\n<user id=f2485662-64c1-4865-b77f-6bdf3ae07f85>I'm Joe and I'm 40 years old, have two cats named Sherlock and Mycroft, and I'm into drones.</user>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nRemember that this schema must be useful for prompt engineering an AI assistant.\n</instructions>",
                "current_value": {},
            },
            user_memory_schema,
        ),
    ],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o",
        "claude-3-5-sonnet-20240620",
        "accounts/fireworks/models/firefunction-v2",
    ],
)
async def test_schema_extracted(model_name: str, inputs: dict, tool_def: dict):
    await check_schema_extracted(model_name, inputs, tool_def)


@test
async def check_schema_extracted(model_name: str, inputs: dict, tool_def: dict):
    model = create_model_from_schema(tool_def["parameters"])
    result = await predict_with_model(model_name, inputs, tool_def)
    expect(result["messages"][0].tool_calls[0]["args"]).against(model.model_validate)
    return result


fitness_log_entry = {
    "name": "FitnessLogEntry",
    "description": "",
    "parameters": {
        "$defs": {
            "Activity": {
                "properties": {
                    "name": {
                        "default": "",
                        "description": "Name of the activity.",
                        "title": "Name",
                        "type": "string",
                    },
                    "duration_seconds": {
                        "default": 0,
                        "description": "Duration of the activity in seconds.",
                        "title": "Duration Seconds",
                        "type": "number",
                    },
                    "intensity": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Intensity of the activity.",
                        "enum": ["High", "Moderate", "Low", "None"],
                        "title": "Intensity",
                    },
                    "notes": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Notes for the activity.",
                        "title": "Notes",
                    },
                },
                "required": ["intensity"],
                "title": "Activity",
                "type": "object",
            },
            "ActivityTrainingWorkout": {
                "description": "An activity training workout session.\nCan consist of activities like:\n- HIIT\n- Cardio\n- Yoga, Pilates or Flexibility workouts",
                "properties": {
                    "duration": {
                        "default": 0,
                        "description": "Duration of the workout in hours.",
                        "title": "Duration",
                        "type": "number",
                    },
                    "title": {
                        "default": "",
                        "description": "A title for the activity/workout. It should be a generated succinct identifier.",
                        "title": "Title",
                        "type": "string",
                    },
                    "category": {
                        "default": "",
                        "description": "Categorize the activity/workout.",
                        "title": "Category",
                        "type": "string",
                    },
                    "cycles": {
                        "default": [],
                        "description": "List of cycles done during the workout.",
                        "items": {"$ref": "#/$defs/Cycle"},
                        "title": "Cycles",
                        "type": "array",
                    },
                },
                "title": "ActivityTrainingWorkout",
                "type": "object",
            },
            "Cycle": {
                "properties": {
                    "activities": {
                        "description": "List of activities done in the cycle.",
                        "items": {"$ref": "#/$defs/Activity"},
                        "title": "Activities",
                        "type": "array",
                    },
                    "rounds": {
                        "default": 1,
                        "description": "Number of times this cycle happened back-to-back.",
                        "title": "Rounds",
                        "type": "integer",
                    },
                },
                "required": ["activities"],
                "title": "Cycle",
                "type": "object",
            },
            "Exercise": {
                "description": "The exercise done during strengh trainging.\nOne exercise is composed of sets.",
                "properties": {
                    "exercise_name": {
                        "description": "Name of the exercise done.",
                        "title": "Exercise Name",
                        "type": "string",
                    },
                    "sets": {
                        "anyOf": [
                            {"items": {"$ref": "#/$defs/Set"}, "type": "array"},
                            {"type": "null"},
                        ],
                        "default": None,
                        "description": "List of sets done for this exercise.",
                        "title": "Sets",
                    },
                    "notes": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Notes for the exercise.",
                        "title": "Notes",
                    },
                },
                "required": ["exercise_name"],
                "title": "Exercise",
                "type": "object",
            },
            "Set": {
                "description": "A set for a single exericise.\nA set is composed of reps.",
                "properties": {
                    "reps": {
                        "default": None,
                        "description": "Number of reps in the set.",
                        "title": "Reps",
                        "type": "integer",
                    },
                    "weight": {
                        "default": None,
                        "description": "Weight used for the set.",
                        "title": "Weight",
                        "type": "number",
                    },
                    "weight_units": {
                        "default": None,
                        "description": "Units for weight.",
                        "enum": ["lbs", "kg"],
                        "title": "Weight Units",
                        "type": "string",
                    },
                    "notes": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Notes for the set.",
                        "title": "Notes",
                    },
                },
                "title": "Set",
                "type": "object",
            },
            "StrenghTrainingWorkout": {
                "description": "A strengh / resistance training workout. A typical gym workout with weights and machines.",
                "properties": {
                    "duration": {
                        "default": 0,
                        "description": "Duration of the workout in hours.",
                        "title": "Duration",
                        "type": "number",
                    },
                    "title": {
                        "default": "",
                        "description": "A title for the activity/workout. It should be a generated succinct identifier.",
                        "title": "Title",
                        "type": "string",
                    },
                    "category": {
                        "default": "",
                        "description": "Categorize the activity/workout.",
                        "title": "Category",
                        "type": "string",
                    },
                    "exercises": {
                        "default": [],
                        "description": "List of exercises done during the resistance training workout.",
                        "items": {"$ref": "#/$defs/Exercise"},
                        "title": "Exercises",
                        "type": "array",
                    },
                },
                "title": "StrenghTrainingWorkout",
                "type": "object",
            },
        },
        "properties": {
            "workout": {
                "anyOf": [
                    {"$ref": "#/$defs/StrenghTrainingWorkout"},
                    {"$ref": "#/$defs/ActivityTrainingWorkout"},
                ],
                "description": "The activity/workout/training done.",
                "title": "Workout",
            },
            "notes": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "defailt": None,  # weird spelling mistake in the data.
                "description": "General notes about the workout session",
                "title": "Notes",
            },
        },
        "required": ["workout", "notes"],
        "type": "object",
    },
}

core_belief_schema = {
    "name": "CoreBelief",
    "description": "",
    "parameters": {
        "properties": {
            "belief": {
                "default": "",
                "description": "The belief the user has about the world, themselves, or anything else.",
                "title": "Belief",
                "type": "string",
            },
            "why": {
                "description": "Why the user believes this.",
                "title": "Why",
                "type": "string",
            },
            "context": {
                "description": "The raw context from the conversation that leads you to conclude that the user believes this.",
                "title": "Context",
                "type": "string",
            },
        },
        "required": ["why", "context"],
        "type": "object",
    },
}

formative_event_schema = {
    "name": "FormativeEvent",
    "description": "",
    "parameters": {
        "properties": {
            "event": {
                "default": "",
                "description": "The event that occurred. Must be important enough to be formative for the user.",
                "title": "Event",
                "type": "string",
            },
            "impact": {
                "default": "",
                "description": "How this event influenced the user.",
                "title": "Impact",
                "type": "string",
            },
        },
        "type": "object",
    },
}


academic_rubric = {
    "name": "AcademicRubric",
    "description": "The rubric for an academic assignment.",
    "parameters": {
        "properties": {
            "name": {
                "description": "The user's name",
                "title": "Name",
                "type": "string",
            },
            "grade": {
                "description": "The user's grade",
                "title": "Grade",
                "type": "string",
            },
            "grocery_list": {
                "description": "The user's grocery list",
                "items": {"type": "string"},
                "title": "Grocery List",
                "type": "array",
            },
        },
        "required": ["name", "grade", "grocery_list"],
        "title": "UserMemory",
        "type": "object",
    },
}


# @pytest.mark.asyncio_cooperative
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "inputs, tool_def, expected",
    [
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<system>You are a helpful AI assistant</system>\n<user id=f2485662-64c1-4865-b77f-6bdf3ae07f85>I'm Jeremiah and I'm 27 years old, have one cat named Morgan and a dog named Ehud. Used to have a cat named Watson, but he passed away :(; and I'm into pasta making.</user>\n</conversation>\n</conversations>",
                "system_prompt": "\n## Additional Instructions\n    \n<instructions>\nRemember that this schema must be useful for prompt engineering an AI assistant.\n</instructions>",
                "current_value": {
                    "name": "Joe",
                    "age": 40,
                    "interests": ["drones"],
                    "relationships": {"cats": "Sherlock and Mycroft"},
                    "ai_instructions": [],
                },
            },
            user_memory_schema,
            {
                "name": "Jeremiah",
                "age": 27,
                "interests": ["pasta making"],
                "relationships": {
                    AnyStr(("cat", "cats")): ContainsStr("Morgan"),
                    AnyStr(("dogs", "dog")): ContainsStr("Ehud"),
                },
                "ai_instructions": [],
            },
        ),
        (
            {
                "input_str": "<conversations>\n<conversation idx=0>\n<system>You are a helpful AI assistant</system>\n<user id=f2485662-64c1-4865-b77f-6bdf3ae07f85I>How do you exit vim? not sure how it 's different from emacs but i'm hella stuck</user>\n</conversation>\n</conversations>",
                "system_prompt": "",
                "current_value": {
                    "name": "笋岗",
                    "grade": "8th",
                    "grocery_list": ["pomelo", "milk", "seitan", "aubergine"],
                },
            },
            academic_rubric,
            {
                "name": "笋岗",
                "grade": "8th",
                "grocery_list": ["pomelo", "milk", "seitan", "aubergine"],
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o",
        "claude-3-5-sonnet-20240620",
        "accounts/fireworks/models/firefunction-v2",
    ],
)
async def test_extraction_updates(
    model_name: str, inputs: dict, tool_def: dict, expected: dict
):
    """Check that the schema is updated correctly."""
    await check_extraction_updates(model_name, inputs, tool_def, expected)


@test
async def check_extraction_updates(
    model_name: str, inputs: dict, tool_def: dict, expected: dict
):
    """Check that the schema is updated correctly."""
    result = await predict_with_model(model_name, inputs, tool_def)
    model = create_model_from_schema(tool_def["parameters"])
    predicted = result["messages"][0].tool_calls[0]["args"]
    assert not isinstance(predicted, Exception)
    expect(predicted).against(model.model_validate)
    for key, value in expected.items():
        pred = predicted[key]
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_key, AnyStr):
                    if not any(pred.get(opt) == sub_value for opt in sub_key.matches):
                        raise AssertionError(
                            f"Expected {sub_key} in {pred} to equal {sub_value}"
                        )
                else:
                    expect(pred[sub_key]).to_equal(sub_value)
        else:
            expect(predicted[key]).to_equal(value)
