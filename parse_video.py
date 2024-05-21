from pydub import AudioSegment
from pydub.silence import split_on_silence
import pathlib
import whisper


CURRENT_DIR = pathlib.Path(__file__).parent.resolve()

MIN_SEGMENT_DURATION = 1 * 1000  # 1 second
MAX_SEGMENT_DURATION = 10 * 1000  # 10 seconds
MIN_SILENCE_LEN = 500  # 0.5 seconds
SILENCE_THRESH = -20
# SILENCE_THRESH_DW = -35
# MIN_SILENCE_LEN_DW = 300


def segment_audio_file(path_to_file: str, file_format: str) -> None:
    """
    Segments an audio file into chunks based on silence and export to separate mp3 files.

    Args:
        path_to_file (str): The path to the audio file without the file extension.
        file_format (str): The format of the audio file (e.g., 'mp4').

    Returns:
        None
    """
    # Load the audio file
    mp4_audio = AudioSegment.from_file(f'{path_to_file}.{file_format}', format=file_format)

    # Split the audio on silence
    chunks: list[AudioSegment] = split_on_silence(
        mp4_audio,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
    )

    # filtered_chunks = list(filter(is_filtered_by_duration, chunks))

    # Load the Whisper model
    model = whisper.load_model("base")

    # Export each filtered by length audio
    for i, chunk in enumerate(chunks):
        if is_filtered_by_duration(chunk):
            print(f"Chunk length: {len(chunk)} ms")
            output_mp3_file = f"{CURRENT_DIR}/audio_segments/chunk{i}.mp3"
            print("Exporting file", output_mp3_file)
            chunk.export(output_mp3_file, format="mp3")

            # Transcribe audio file and export
            transcription = model.transcribe(output_mp3_file, language='ru')
            output_txt_file = f"{CURRENT_DIR}/text_segments/chunk{i}.txt"

            with open(output_txt_file, "w") as f:
                print(f"▼ Transcription of {output_txt_file}\n")
                f.write(transcription['text'])


def is_filtered_by_duration(audio_segment: AudioSegment) -> bool:
    """
    Checks whether AudioSegment duration lies between MIN_SEGMENT_DURATION and MAX_SEGMENT_DURATION.

    Args:
        audio_segment (AudioSegment): AudioSegment to check.

    Returns:
        bool
    """
    return MIN_SEGMENT_DURATION <= len(audio_segment) <= MAX_SEGMENT_DURATION


if __name__ == '__main__':
    test_file_path = f'{CURRENT_DIR}/friends_audio'
    file_format = 'mp4'

    segment_audio_file(test_file_path, file_format)
