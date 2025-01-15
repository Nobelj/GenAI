# The document contains the following information:

# * **Patient's Name**: Kamlesh
# * **Age**: 35 years, male
# * **Address**: H. Shankar nag Colony Raipur
# * **Contact Number**: 9160521341
# * **Doctor's Name**: Dr. Satyadev Singh MBBS, M.D.
# * **Registration Number**: CMC 4973
# * **Address**: 27 Soma Colony Raipur
# * **Contact Number**: 9230523491
# * **Date**: 21-05-2016
# * **Medication Prescribed**:
# 	+ Tablet Ciprofloxacin (500 mg): 1 tablet in the morning and 1 tablet in the evening for 5 days
# 	+ Tablet Paracetamol (500 mg): take 1 tablet whenever fever is more than 100°F to the maximum of 4 tablets in a day

# This information appears to be a prescription note for a patient named Kamlesh, written by Dr. Satyadev Singh.


import subprocess
import requests


api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJIYXBweSAiLCJVc2VyTmFtZSI6IkhhcHB5ICIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxODcyNTcyNTI0MTY5MDA3NDM4IiwiUGhvbmUiOiI5MTM5NDQ5MyIsIkdyb3VwSUQiOiIxODcyNTcyNTI0MTYwNjE4ODMwIiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMDEtMDYgMjE6MTE6MzYiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.KhYUJBKe-sXvo6pwxKcKH5yIRmftnXvr-8n8WCfQmBel9G0iNkrXb0qd-lmcFiDoDXsbjibk_wA6VamsW1sOcZIop_tt0VgnSfrRmN3noPfXu6GFya4q7OhfUoC8-Hs8y153KjlW-OymGTPHVRFyeaG26pTHYoy4e-4Az3Udp-phQCo5m_Szc6X8hyNIz8TdKZ_0VMZL8ZWt5hXO76JAMzwjvEGXUCdO66QbcCHmLIh8jakrPfWFRXVO1i-dL5sbCSP60vELl2oe0pWM9egrnPKrRy1MSSOxX_Mjyf2j9J4pVpcxIeOhX-VYN4UnfbZ3H0Ua6ETw3fIItiqahfCpiA"


def upload(file_name, file_path):
    url = "https://api.minimax.chat/v1/music_upload"
    payload = {"purpose": "song"}
    files = [("file", (file_name, open(file_path, "rb"), "audio/mpeg"))]
    headers = {
        "authorization": "Bearer " + api_key,
    }

    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    print(response.headers.get("Trace-Id"))
    print(response.text)
    return response


def generate_audiobytes(refer_voice, refer_instrumental, lyrics, refer_vocal=""):
	url = "https://api.minimax.chat/v1/music_generation"
	# refer_vocal = '请输入您的复刻音色ID'

	payload = {
		"refer_voice": refer_voice,
		"refer_instrumental": refer_instrumental,
		# 'refer_vocal': refer_vocal
		"lyrics": lyrics,
		"model": "music-01",
		"audio_setting": '{"sample_rate":44100,"bitrate":256000,"format":"mp3"}',
	}

	headers = {
		"authorization": "Bearer " + api_key,
	}

	response = requests.post(url, headers=headers, data=payload)
	audio_hex = response.json()["data"]["audio"]
	audio_bytes = bytes.fromhex(audio_hex)
	return audio_bytes
