import requests
from suno import Suno, ModelVersions

# client = Suno(
#     cookie="__client=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNsaWVudF8ycDNrTXpYdWtLMTJqN3dEOUpvbmxNVDVnN2IiLCJyb3RhdGluZ190b2tlbiI6ImRoZXpzM3VhZWo5bWFndmJsYmVkbXF6Mm0yaG9neXpmcXJkZDJrODEifQ.H0-wsq1OuaCmJSnwb5ADPFHCgtZJziB6bk6_--o54wDX15ry2iMvglfxWCiebdrrnhwWfgXektvw1jcTiryaUnEi4Jbe5lzWeSo_p4lyYMutvyBYxRjiVsN4L_lzbsWikhiamR5Bl7TeyhMjM8AsDoG13Cb2sT13idz4QE4gwdvNyIxTYuq_RUcICoQbEx6SgAv4JEOIygiT9TtxkNBlUqJQOYlLNmprRI-NgtPycsGh4VHEnlgSuFQcsTNyW2ZbXSrfK889VUXL4BIbRqAVg7rbPkTgJXH9wpruRjD8BxAI4ZfbE9Mo3udbX8f9WYHUJu6gRypFG43dtQMDnTuqFA; __client_uat=1732002114; __client_uat_U9tcbTPE=1732002114; __cf_bm=1yJOKcy3DpZt.xRyUg2bW0TUTwXtkDwF3Rq0._iUySI-1735895564-1.0.1.1-J51.xH2uPrCWCokjZSI4MncFS2sqvKzfRUHg0j9JvAmd6liDjrDYSM0ejQsQ6h3dvXHiB6PJq7gRPexA3wQgXw; _cfuvid=ATGVXDu_SxxNjNWs48NAEXSdZOU9ww18QbwC.xBZWuc-1735895564481-0.0.1.1-604800000; ajs_anonymous_id=561fdf81-1594-4525-b40d-74015e861248",
#     model_version=ModelVersions.CHIRP_V3_5,
# )


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
    return ["", ""]
    response_data = client.generate(
        prompt=prompt,
        tags=tags,
        is_custom=True,
        title=title,
        make_instrumental=False,
        wait_audio=True,
    )
    audio_urls = [response_data[0].audio_url, response_data[1].audio_url]
    return audio_urls
