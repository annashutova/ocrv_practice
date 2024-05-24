from gradio_client import Client, file


def transcribe_with_api(client: Client, input_wav_file: str, output_txt_file: str):
    result = client.predict(
        inputs=file(input_wav_file),
        task="transcribe",
        api_name="/predict"
    )

    with open(output_txt_file, "w") as f:
        print(f"▼ Transcription of {input_wav_file}\n")
        f.write(result)
