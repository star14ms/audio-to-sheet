import subprocess


def download_audio_from_youtube(url, verbose=True):
  subprocess.call([
    "yt-dlp", 
    "-f", "251",
    "-o", "./data/audio/%(title)s.%(ext)s",
    url,
  ], stderr=subprocess.DEVNULL if not verbose else None)


if __name__ == '__main__':
  url = input("Enter the YouTube URL: ")
  download_audio_from_youtube(url)