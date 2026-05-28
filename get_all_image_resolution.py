from trackit.data.source.dataset.builder import build_dataset_as_data_source
from trackit.core.runtime.utils.custom_yaml_loader import load_yaml
from tqdm import tqdm
import math

if __name__ == "__main__":
    config = load_yaml("config/_dataset/train.yaml")
    datasets = build_dataset_as_data_source({'type': 'dataset', 'parameters': config})
    frame_sizes = {}
    for dataset in datasets:
        for sequence in tqdm(dataset):
            for sample in sequence:
                frame_size = tuple(sample.get_frame_size())
                if frame_sizes.get(frame_size) is None:
                    frame_sizes[frame_size] = 0
                frame_sizes[frame_size] += 1
    # print sorted frame sizes
    sorted_frame_sizes = sorted(frame_sizes.items(), key=lambda x: x[1], reverse=True)
    total_sequences = sum(count for _, count in sorted_frame_sizes)
    for frame_size, count in sorted_frame_sizes:
        gcd = math.gcd(frame_size[0], frame_size[1])
        print(f"Frame size: {frame_size}, Count: {count}, Aspect ratio: {frame_size[0] // gcd}:{frame_size[1] // gcd}, percentage: {count / total_sequences:.2%}")

    # print aspect ratios sorted by count
    aspect_ratios = {}
    for frame_size, count in sorted_frame_sizes:
        gcd = math.gcd(frame_size[0], frame_size[1])
        aspect_ratio = f"{frame_size[0] // gcd}:{frame_size[1] // gcd}"
        if aspect_ratios.get(aspect_ratio) is None:
            aspect_ratios[aspect_ratio] = 0
        aspect_ratios[aspect_ratio] += count
    sorted_aspect_ratios = sorted(aspect_ratios.items(), key=lambda x: x[1], reverse=True)
    total_aspect_ratios = sum(count for _, count in sorted_aspect_ratios)
    for aspect_ratio, count in sorted_aspect_ratios:
        print(f"Aspect ratio: {aspect_ratio}, Count: {count}, percentage: {count / total_aspect_ratios:.2%}")

