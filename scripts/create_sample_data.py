"""Script to generate sample images and sample_dataset.jsonl for testing/demonstration."""

import json
import os
from PIL import Image, ImageDraw


def create_sample_data() -> None:
    """Generates synthetic geometric/color images and a JSONL dataset file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    img_dir = os.path.join(data_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    samples = [
        {
            "filename": "red_square.jpg",
            "bg": (240, 240, 240),
            "draw": lambda d: d.rectangle([64, 64, 160, 160], fill=(220, 40, 40)),
            "question": "What shape and color is in the center of the image?",
            "answer": "There is a red square in the center of the image.",
        },
        {
            "filename": "blue_circle.jpg",
            "bg": (240, 240, 240),
            "draw": lambda d: d.ellipse([64, 64, 160, 160], fill=(30, 100, 220)),
            "question": "What object is visible in the image?",
            "answer": "A blue circle is displayed against a light background.",
        },
        {
            "filename": "green_box.jpg",
            "bg": (240, 240, 240),
            "draw": lambda d: d.rectangle([40, 80, 184, 144], fill=(34, 160, 60)),
            "question": "Describe the main feature in this picture.",
            "answer": "The image shows a horizontal green rectangle.",
        },
        {
            "filename": "yellow_star.jpg",
            "bg": (30, 30, 40),
            "draw": lambda d: d.polygon([(112, 40), (130, 90), (184, 90), (140, 122), (156, 174), (112, 142), (68, 174), (84, 122), (40, 90), (94, 90)], fill=(240, 200, 20)),
            "question": "What is the bright object on the dark background?",
            "answer": "It is a bright yellow star centered on a dark background.",
        },
    ]

    jsonl_records = []
    for s in samples:
        img_path = os.path.join(img_dir, s["filename"])
        rel_path = f"data/images/{s['filename']}"

        # Draw and save image
        img = Image.new("RGB", (224, 224), color=s["bg"])
        draw = ImageDraw.Draw(img)
        s["draw"](draw)
        img.save(img_path, quality=95)

        jsonl_records.append({
            "image": rel_path,
            "question": s["question"],
            "answer": s["answer"],
        })

    jsonl_path = os.path.join(data_dir, "sample_dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Created {len(samples)} sample images in {img_dir}")
    print(f"Created sample dataset in {jsonl_path}")


if __name__ == "__main__":
    create_sample_data()
