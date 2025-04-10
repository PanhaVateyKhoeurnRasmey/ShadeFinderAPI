import sys
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import base64
import os
from main import app
import warnings

client = TestClient(app)

def read_image(image_path:str) -> bytes:
     with open(image_path, "rb") as image_file:
          return image_file.read()
     
def test_print():
    print("this is successful!")

def test_predict_endpoint():
    image_path = "test_image.png"
    image_data = read_image(image_path)
    response = client.post(
        "/predict/", 
        files={"file": ("test_image.png", image_data, "image/png")}
    )
    print("response:\n", response)
    assert response.status_code == 200
    response_data = response.json()
    print(response_data)
    assert 'undertone' in response_data
    assert 'tone' in response_data
    assert isinstance(response_data['undertone'], dict)  
    assert isinstance(response_data['tone'], dict)     