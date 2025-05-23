import requests

url = "" # （API）
api_key = ""  # （KEY）


# example
file_paths = [
    "./image/6.png",
    "./image/7.png",
    "./image/8.png"
]
question = "Describe the three images in detail." # (Question)


'''
# text example
file_paths = []
question = "describe beijing"
'''

files = [('files', open(file_path, 'rb')) for file_path in file_paths]
data = {
    'question': question,
    'api_key': api_key
}

try:
    response = requests.post(url, files=files, data=data)
    if response.status_code == 200:
        print("Response:", response.json().get("response", "No response key found in the JSON."))
    else:
        print("Error:", response.status_code, response.text)
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")