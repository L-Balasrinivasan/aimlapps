import requests

# Specify the API endpoint URL
url = "http://localhost:8000/check-similarity/"

# Prepare the data for the POST request
files = {
    "file1": ("image1.jpg", open("image_1.png", "rb")),
    "file2": ("image2.jpg", open("image_2.jpg", "rb")),
}
data = {"threshold": 0.7}

# Send the POST request with files
response = requests.post(url, files=files, data=data)

# Print the response
print(response.json())
