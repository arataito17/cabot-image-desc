# Copyright (c) 2025  Carnegie Mellon University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys
import logging
import pytest
import os
import json
from pymongo import MongoClient
from bson import ObjectId
from fastapi.testclient import TestClient

sys.path.append('/app')  # this should not be changed, this is a path in the docker container
os.environ["API_KEY"] = "test_api_key"
os.environ["USERNAMES"] = "test_user"
os.environ["PASSWORDS"] = "test_password"
os.environ["MONGODB_HOST"] = "mongodb://mongo-test:27017/"
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "__DUMMY_OPENAI_API_KEY__")
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "__DUMMY_HF_TOKEN__")
os.environ["TRANSFORMERS_CACHE"] = "/mnt/arata/models"
os.environ["HF_HOME"] = "/mnt/arata/models"

from server.app import app   # noqa: E402, needs to be loaded after the environment variables are set

client = TestClient(app, follow_redirects=False)

# Set up MongoDB connection for tests
mongodb_host = os.getenv('MONGODB_HOST', 'mongodb://mongo-test:27017/')
mongodb_name = os.getenv('MONGODB_NAME', 'geo_image_db')
client_db = MongoClient(mongodb_host)
db = client_db[mongodb_name]
image_collection = db['images']


# Fixture to set up and tear down the database before and after each test
@pytest.fixture(autouse=True)
def insert_dummy_data():
    # Load dummy data from test.json
    with open('/test/data/test.json') as f:
        dummy_data = json.load(f)
        # Convert _id to ObjectId
        dummy_data["_id"] = ObjectId(dummy_data["_id"])

    # Insert dummy data before each test
    result = image_collection.insert_one(dummy_data)

    yield result.inserted_id  # return the inserted id, which is used in the tests

    # Clean up after each test
    image_collection.delete_many({})


# Fixture to log in before each test that requires authentication
@pytest.fixture
def login():
    response = client.post("/login", data={"username": os.environ["USERNAMES"], "password": os.environ["PASSWORDS"]})
    assert response.status_code == 303
    for name, value in response.cookies.items():
        client.cookies.set(name, value)
    return response


# Fixture to set the log level to INFO for each test
@pytest.fixture(autouse=True)
def set_log_level(caplog):
    caplog.set_level(logging.INFO)


# Fixture to set up headers for API key authentication
@pytest.fixture
def api_key_headers():
    return {"x-api-key": os.environ["API_KEY"]}

# Do not change code style, especially two lines between functions


# Test with a user/pass login cookies


# Test the login endpoint
def test_login():
    response = client.post("/login", data={"username": os.environ["USERNAMES"], "password": os.environ["PASSWORDS"]})
    assert response.status_code == 303
    assert "token" in response.cookies

    # Set the cookies directly on the client instance
    for name, value in response.cookies.items():
        client.cookies.set(name, value)

    # Follow the redirect
    redirect_url = response.headers["Location"]
    response = client.get(redirect_url)
    assert response.status_code == 200


# Test the logout endpoint
def test_logout(login):
    # Logout
    response = client.get("/logout")
    assert response.status_code == 303
    assert "token" not in response.cookies


# Test the read_location endpoint
def test_read_location(login, insert_dummy_data):
    # Read location
    response = client.get(f"/locations/{insert_dummy_data}")
    assert response.status_code == 200
    assert "description" in response.json()


# Test the read_locations_by_lat_lng endpoint
def test_read_locations_by_lat_lng(login):
    # Read locations by latitude and longitude
    response = client.get("/locations?lat=35.62414&lng=139.7754&distance=1000")
    assert response.status_code == 200
    assert len(response.json()) > 0


# Test the update_description endpoint
def test_update_description(login, insert_dummy_data):
    # Update description
    response = client.post(f"/update_description?id={insert_dummy_data}", data={"description": "New description"})
    assert response.status_code == 200


# Test the update_floor endpoint
def test_update_floor(login, insert_dummy_data):
    # Update floor
    response = client.post(f"/update_floor?id={insert_dummy_data}", data={"floor": 2})
    assert response.status_code == 200


# Test the add_tag endpoint
def test_add_tag(login, insert_dummy_data):
    # Add tag
    response = client.post(f"/add_tag?id={insert_dummy_data}", data={"tag": "new_tag"})
    assert response.status_code == 200


# Test the clear_tag endpoint
def test_clear_tag(login, insert_dummy_data):
    # Clear tag
    response = client.post(f"/clear_tag?id={insert_dummy_data}")
    assert response.status_code == 200


# Test with an api key


# Test the read_locations_by_lat_lng endpoint
def test_read_locations_by_lat_lng_with_api_key(api_key_headers):
    # Read locations by latitude and longitude
    response = client.get("/locations?lat=35.62414&lng=139.7754&distance=1000", headers=api_key_headers)
    assert response.status_code == 200
    assert len(response.json()) > 0


# Test the read_description_by_lat_lng endpoint
def test_read_description_by_lat_lng_with_api_key_ja(api_key_headers):
    # Read description by latitude and longitude
    response = client.get("/description?lat=35.62414&lng=139.7754&floor=0&rotation=0.0&max_count=10&max_distance=100", headers=api_key_headers)
    assert response.status_code == 200


# Test read_description_by_lat_lng_with_image endpoint using dummy image from test.json
def test_read_description_by_lat_lng_with_image_ja(api_key_headers, insert_dummy_data):
    with open('/test/data/test.json') as f:
        dummy = json.load(f)
    image_payload = [{"position": "front", "image_uri": dummy["image"]}]
    params = {
        "lat": 35.62414166666667,
        "lng": 139.77542222222223,
        "floor": 1,
        "rotation": 0,
        "max_count": 10,
        "max_distance": 15,
        "length_index": 0,
        "distance_to_travel": 100,
    }
    response = client.post("/description_with_live_image", headers=api_key_headers, params=params, json=image_payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "description" in json_response


# Test stop_reason endpoint using dummy image from test.json
def test_stop_reason_with_image_ja(api_key_headers, insert_dummy_data):
    with open('/test/data/test.json') as f:
        dummy = json.load(f)
    image_payload = [{"position": "front", "image_uri": dummy["image"]}]
    params = {
        "lat": 35.62414166666667,
        "lng": 139.77542222222223,
        "floor": 1,
        "rotation": 0,
        "max_count": 10,
        "max_distance": 100,
        "length_index": 0,
        "distance_to_travel": 100,
    }
    response = client.post("/stop_reason", headers=api_key_headers, params=params, json=image_payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "description" in json_response


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_read_description_by_lat_lng_with_api_key_lang(api_key_headers, lang):
    response = client.get(
        f"/description?lat=35.62414&lng=139.7754&floor=0&rotation=0.0&max_count=10&max_distance=100&lang={lang}",
        headers=api_key_headers
    )
    assert response.status_code == 200
    json_response = response.json()
    assert "lang" in json_response, "Expected 'lang' key in response JSON"
    assert json_response["lang"] == lang or json_response["lang"] == "dummy_value", f"Expected language '{lang}' in response JSON"
    assert "translated" in json_response, "Expected 'translated' key in response JSON"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_read_description_by_lat_lng_with_image_lang(api_key_headers, insert_dummy_data, lang):
    with open('/test/data/test.json') as f:
        dummy = json.load(f)
    image_payload = [{"position": "front", "image_uri": dummy["image"]}]
    params = {
        "lat": 35.62414166666667,
        "lng": 139.77542222222223,
        "floor": 1,
        "rotation": 0,
        "max_count": 10,
        "max_distance": 15,
        "length_index": 0,
        "distance_to_travel": 100,
        "lang": lang
    }
    response = client.post("/description_with_live_image", headers=api_key_headers, params=params, json=image_payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "lang" in json_response, "Expected 'lang' key in response JSON"
    assert json_response["lang"] == lang or json_response["lang"] == "dummy_value", f"Expected language '{lang}' in response JSON"
    assert "translated" in json_response, "Expected 'translated' key in response JSON"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_stop_reason_with_image_lang(api_key_headers, insert_dummy_data, lang):
    with open('/test/data/test.json') as f:
        dummy = json.load(f)
    image_payload = [{"position": "front", "image_uri": dummy["image"]}]
    params = {
        "lat": 35.62414166666667,
        "lng": 139.77542222222223,
        "floor": 1,
        "rotation": 0,
        "max_count": 10,
        "max_distance": 100,
        "length_index": 0,
        "distance_to_travel": 100,
        "lang": lang
    }
    response = client.post("/stop_reason", headers=api_key_headers, params=params, json=image_payload)
    assert response.status_code == 200
    json_response = response.json()
    assert "lang" in json_response, "Expected 'lang' key in response JSON"
    assert json_response["lang"] == lang or json_response["lang"] == "dummy_value", f"Expected language '{lang}' in response JSON"
    assert "translated" in json_response, "Expected 'translated' key in response JSON"
