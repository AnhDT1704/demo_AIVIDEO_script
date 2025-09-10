import requests

url = "https://api.blaze.vn/v1/stt/transcribe?enable_segments=true"

files = [
    (
        'audio_file',
        (
            'Bản ghi Mới 119 (mp3cut (mp3cut.net) (1).mp3', 
            open("D:/audiotest.m4a", 'rb'), 
            'audio/mpeg'
        )
    )
]

headers = {
    'Authorization': 'Bearer 220a427773d646ef18f7e2e52e649336991462b0'
}

response = requests.post(url, headers=headers, files=files)

print(response.text)
