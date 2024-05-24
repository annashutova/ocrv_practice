from gradio_client import Client, file


def transcribe_with_api(client: Client, input_wav_file: str, output_txt_file: str):
    """
    Transcribes audio wav file using gradio client with model.

    Args:
        client (Client): gradio client.
        input_wav_file (str): path to input audio wav file.
        output_txt_file (str): path to output txt file.

    Returns:
        None
    """
    result = client.predict(
        inputs=file(input_wav_file),
        task="transcribe",
        api_name="/predict"
    )

    # Export to output_txt_file
    with open(output_txt_file, "w") as f:
        print(f"▼ Transcription of {input_wav_file}\n")
        f.write(result)
