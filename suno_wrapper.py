import requests
from suno import Suno, ModelVersions

client = Suno(
    cookie="js_anonymous_id=561fdf81-1594-4525-b40d-74015e861248; __stripe_mid=b91b7377-7da2-42f8-8997-cdfb3b250f281a3e0f; __client=eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImNsaWVudF8ybGVJdWVCMzJMM0NMRmY0ZlY1dnBROTBmazIiLCJyb3RhdGluZ190b2tlbiI6ImZwdGFqNGJ4dGF4MDBsc2ZwdjQzcWJ6bTUzbmhrM2M5Y296Nzk4cmMifQ.wlSdpaNRHy74qYtx3w17fFqTWXUtLvJ4mVigLfyZ6wIuzpUoR4Pp2Oh7GrKpMWeluEH8947MKM675PCL1mAxyM5-q2g5MUVJ5M8cjZZszBxOInH7SZ6eKm7UdwtesaGK3M234yfXRrX4zOu7AxpesTsrb4P3A-RaXS3SyqZDXT23aLlWXXdyYSncoeKyXtYdgYFTjC7Rc8MkMWyGXv0aeWWivXBhwrXOk4s724DwZuMSX5ec1yUqfUGt-fM0N5EhpB23RqRjE32DFaES0jqYwsoWhP76IR8QtOnivBKnUIy1LISzwdQvIQgnr9IyxVmL-TSJRs6LyNkvzrrM1BuQfQ; __client_uat=1725534343; __cf_bm=vXV6W460_Ng9dM6kA7ERc432JTG.rk_IGqK7iIh4O34-1728056817-1.0.1.1-juqe2qBcHLEZLZfxQ5I2rHo8gSM7UlyTqt2jkP8atghTb2c742UR9.axFdAgY9h5O1LYhoJuaa5BboGrSWkcxg; _cfuvid=gBAFhhHR1e.x53cOZElGbNmWqStC2S4IsDYszuq0Yi4-1728056817810-0.0.1.1-604800000; __client_uat_U9tcbTPE=1725534343",
    model_version=ModelVersions.CHIRP_V3_5,
)


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
