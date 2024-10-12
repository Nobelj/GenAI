import requests

url = "http://localhost:3000/api/custom_generate"


def send_to_suno(
    prompt: str,
    tags: str,
    title: str,
) -> list:
    data = {
        "prompt": prompt,
        "tags": tags,
        "title": title,
        "make_instrumental": "false",
        "wait_audio": "true",
    }

    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.status_code, response.text)

    response_data = response.json()
    audio_urls = [response_data[0]["audio_url"], response_data[1]["audio_url"]]
    return audio_urls
