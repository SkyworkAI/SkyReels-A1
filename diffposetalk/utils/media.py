import shlex
import subprocess
from pathlib import Path


def combine_video_and_audio(video_file, audio_file, output, quality=17, copy_audio=True):
    audio_codec = '-c:a copy' if copy_audio else ''
    cmd = f'ffmpeg -i {video_file} -i {audio_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p ' \
          f'{audio_codec} -fflags +shortest -y -hide_banner -loglevel error {output}'
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def combine_frames_and_audio(frame_files, audio_file, fps, output, quality=17):
    cmd = f'ffmpeg -framerate {fps} -i {frame_files} -i {audio_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p ' \
          f'-c:a copy -fflags +shortest -y -hide_banner -loglevel error {output}'
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def convert_video(video_file, output, quality=17):
    cmd = f'ffmpeg -i {video_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p ' \
          f'-fflags +shortest -y -hide_banner -loglevel error {output}'
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def reencode_audio(audio_file, output):
    cmd = f'ffmpeg -i {audio_file} -y -hide_banner -loglevel error {output}'
    assert subprocess.run(shlex.split(cmd)).returncode == 0


def extract_frames(filename, output_dir, quality=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -i {filename} -qmin 1 -qscale:v {quality} -y -start_number 0 -hide_banner -loglevel error ' \
          f'{output_dir / "%06d.jpg"}'
    assert subprocess.run(shlex.split(cmd)).returncode == 0
