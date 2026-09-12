"""Create one shareable image containing all blinded A/B comparisons."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CANVAS_BACKGROUND = "#EEF2F6"
CARD_BACKGROUND = "#FFFFFF"
TEXT_COLOR = "#172033"
MUTED_COLOR = "#526074"
ACCENT_COLOR = "#244E7A"
BORDER_COLOR = "#CBD5E1"


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def fit_on_white(source: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "white")
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    panel.paste(fitted, offset)
    return panel


def read_subjects(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def create_collage(packet_dir: Path, subjects_path: Path, output_path: Path) -> None:
    subjects = read_subjects(subjects_path)
    if len(subjects) != 8:
        raise ValueError(f"Expected 8 subjects, found {len(subjects)}.")

    columns = 2
    rows = 4
    outer_margin = 48
    column_gap = 28
    row_gap = 28
    title_height = 122
    footer_height = 76
    card_width = 1060
    card_height = 720
    card_padding = 24
    card_header_height = 66
    image_gap = 18
    image_label_height = 42
    image_width = (card_width - 2 * card_padding - image_gap) // 2
    image_height = card_height - card_header_height - image_label_height - 2 * card_padding

    canvas_width = outer_margin * 2 + columns * card_width + column_gap
    canvas_height = (
        outer_margin * 2
        + title_height
        + rows * card_height
        + (rows - 1) * row_gap
        + footer_height
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), CANVAS_BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(42, bold=True)
    subtitle_font = load_font(23)
    subject_font = load_font(27, bold=True)
    label_font = load_font(29, bold=True)
    footer_font = load_font(21)

    draw.text(
        (outer_margin, outer_margin),
        "Blinded A/B Image Comparison",
        fill=TEXT_COLOR,
        font=title_font,
    )
    draw.text(
        (outer_margin, outer_margin + 58),
        "For each person, compare A and B without trying to identify the generating method.",
        fill=MUTED_COLOR,
        font=subtitle_font,
    )

    image_dir = packet_dir / "images"
    for index, subject in enumerate(subjects):
        row, column = divmod(index, columns)
        card_x = outer_margin + column * (card_width + column_gap)
        card_y = outer_margin + title_height + row * (card_height + row_gap)
        card_box = (card_x, card_y, card_x + card_width, card_y + card_height)
        draw.rounded_rectangle(
            card_box,
            radius=18,
            fill=CARD_BACKGROUND,
            outline=BORDER_COLOR,
            width=2,
        )

        heading = f"{subject['subject_id']}  {subject['name']}"
        draw.text(
            (card_x + card_padding, card_y + 18),
            heading,
            fill=TEXT_COLOR,
            font=subject_font,
        )

        for label_index, label in enumerate(("A", "B")):
            source = image_dir / f"{subject['subject_id']}_{label}.png"
            if not source.exists():
                raise FileNotFoundError(source)
            image_x = card_x + card_padding + label_index * (image_width + image_gap)
            label_y = card_y + card_header_height
            image_y = label_y + image_label_height
            label_box = (image_x, label_y, image_x + image_width, image_y)
            draw.rounded_rectangle(label_box, radius=9, fill=ACCENT_COLOR)
            label_bbox = draw.textbbox((0, 0), label, font=label_font)
            label_text_width = label_bbox[2] - label_bbox[0]
            label_text_height = label_bbox[3] - label_bbox[1]
            draw.text(
                (
                    image_x + (image_width - label_text_width) / 2,
                    label_y + (image_label_height - label_text_height) / 2 - label_bbox[1],
                ),
                label,
                fill="white",
                font=label_font,
            )

            panel = fit_on_white(source, (image_width, image_height))
            canvas.paste(panel, (image_x, image_y))
            draw.rectangle(
                (image_x, image_y, image_x + image_width, image_y + image_height),
                outline=BORDER_COLOR,
                width=2,
            )

    footer_y = canvas_height - outer_margin - footer_height + 18
    draw.text(
        (outer_margin, footer_y),
        "Record A, B, or Tie for each subject in the accompanying rating form.",
        fill=MUTED_COLOR,
        font=footer_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    print(f"Reviewer collage: {output_path.resolve()}")
    print(f"Canvas: {canvas_width}x{canvas_height}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", default="evaluation/blind_packet")
    parser.add_argument("--subjects", default="evaluation/subjects.csv")
    parser.add_argument(
        "--output",
        default="evaluation/blind_packet/reviewer_comparison.png",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    create_collage(
        Path(arguments.packet_dir),
        Path(arguments.subjects),
        Path(arguments.output),
    )
