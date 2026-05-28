from trackit.datasets.base.tool.decode_video_to_path import decode_video_file
import os


def decode_AIC19_Track1_videos_to_image(path: str, destination: str):
    labels = os.listdir(path)
    labels.sort()

    os.makedirs(destination, exist_ok=True)

    for label in labels:
        label_path = os.path.join(path, label)
        videos = os.listdir(label_path)
        videos.sort()

        destination_label_path = os.path.join(destination, label)
        os.makedirs(destination_label_path, exist_ok=True)

        for video in videos:
            video_path = os.path.join(label_path, video)
            pos = video.rfind('.')
            assert pos != -1
            destination_video_path = os.path.join(destination_label_path, video[:pos])

            os.makedirs(destination_video_path, exist_ok=True)

            decode_video_file(video_path, destination_video_path)
