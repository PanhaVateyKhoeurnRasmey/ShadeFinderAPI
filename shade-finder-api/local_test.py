import base64
import json
from lambda_function import lambda_handler

with open("test_image.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

event = {
    "body": json.dumps({
        "img": encoded_string
    })
}
context = {}

with open("event.json", "w") as json_file:
    json.dump(event, json_file, indent=2)

response = lambda_handler(event, context)
print(json.dumps(response, indent=2))
