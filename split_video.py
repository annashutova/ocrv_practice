import os

from pydub import AudioSegment
from pydub.silence import split_on_silence

import config


CURRENT_DIR = os.getcwd()


def segment_audio_file(
        path_to_file: str,
        file_format: str,
        export_audio_format: str,
        min_seg_dur: int = 1000,
        max_seg_dur: int = 10000,
        min_silence_len: int = 1000,
        silence_thresh: int = -30
) -> None:
    """
    Segments an audio file into chunks based on silence and export to separate mp3 files.

    Args:
        path_to_file (str): The path to the audio file without the file extension.
        file_format (str): The format of the audio file (e.g., 'mp4').

    Returns:
        None
    """
    base_export_name = os.path.basename(path_to_file)

    # Load the audio file
    mp4_audio = AudioSegment.from_file(f'{path_to_file}.{file_format}', format=file_format)

    # Split the audio on silence
    chunks: list[AudioSegment] = split_on_silence(
        mp4_audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=300  # 0.3 seconds
    )

    # Check the existence of directories
    audio_export_dir = f'{CURRENT_DIR}/audio_segments/{base_export_name}'
    txt_export_dir = f'{CURRENT_DIR}/text_segments/{base_export_name}'

    if not os.path.exists(audio_export_dir):
        os.mkdir(audio_export_dir)
    if not os.path.exists(txt_export_dir):
        os.mkdir(txt_export_dir)

    # Export each filtered by length audio
    for i, chunk in enumerate(chunks):
        if is_filtered_by_duration(chunk, min_seg_dur, max_seg_dur):
            print(f"Chunk length: {len(chunk)} ms")
            output_audio_file = f"{audio_export_dir}/{base_export_name}_chunk{i}.{export_audio_format}"
            print("Exporting file", output_audio_file)
            chunk.export(output_audio_file, format=export_format)

            # output_txt_file = f"{txt_export_dir}/{base_export_name}_chunk{i}.txt"


def is_filtered_by_duration(
        audio_segment: AudioSegment,
        min_seg_dur: int = 1000,
        max_seg_dur: int = 10000,
) -> bool:
    """
    Checks whether AudioSegment duration lies between MIN_SEGMENT_DURATION and MAX_SEGMENT_DURATION.

    Args:
        audio_segment (AudioSegment): AudioSegment to check.

    Returns:
        bool
    """
    return min_seg_dur <= len(audio_segment) <= max_seg_dur


if __name__ == '__main__':
    test_file_path = f'{CURRENT_DIR}/videos/славик_и_димон'
    file_format = 'mp4'
    export_format = 'mp3'

    segment_audio_file(
        test_file_path,
        file_format,
        export_format,
        min_seg_dur=config.MIN_SEG_DUR_DEFAULT,
        max_seg_dur=config.MAX_SEG_DUR_DEFAULT,
        min_silence_len=config.MIN_SILENCE_LEN_SD,
        silence_thresh=config.SILENCE_THRESH_SD,
    )
