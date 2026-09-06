"""Render individual coloring-book pages and a combined A4 PDF book."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
import textwrap
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph


class Concatenator:
    def __init__(self):
        self.title_font = "Helvetica-Bold"
        self.text_font = "Helvetica"
        try:
            pdfmetrics.registerFont(TTFont("Arial", "arial.ttf"))
            pdfmetrics.registerFont(TTFont("Arial-Bold", "arialbd.ttf"))
            self.title_font = "Arial-Bold"
            self.text_font = "Arial"
        except Exception:
            # Helvetica is available in ReportLab and works in a fresh Colab.
            pass

    @staticmethod
    def _ensure_pdf_path(output_path):
        path = Path(output_path)
        if path.suffix.lower() != ".pdf":
            path = path.with_suffix(".pdf")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _draw_page(self, document, page):
        page_width, page_height = A4
        margin = 50
        title = page.get("title", "")

        if title:
            document.setFont(self.title_font, 24)
            document.drawCentredString(page_width / 2, page_height - margin - 20, title)

        image = ImageReader(page["image_path"])
        image_width, image_height = image.getSize()
        max_image_width = page_width - (2 * margin)
        max_image_height = page_height * 0.55
        aspect = image_height / float(image_width)
        draw_width = min(max_image_width, max_image_height / aspect)
        draw_height = draw_width * aspect
        image_x = (page_width - draw_width) / 2
        image_y = page_height - margin - 60 - draw_height
        document.drawImage(
            image,
            image_x,
            image_y,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
            mask="auto",
        )

        text = page.get("text", "")
        if not isinstance(text, str) or not any(char.isalpha() for char in text):
            raise ValueError("A real biography is required; placeholder text cannot be printed.")
        paragraph = Paragraph(escape(text), ParagraphStyle(
            "biography", fontName=self.text_font, fontSize=12, leading=16,
        ))
        text_top = image_y - 24
        _, text_height = paragraph.wrap(max_image_width, text_top - 78)
        if text_height > text_top - 78:
            raise ValueError("Biography does not fit the page; refusing to silently truncate it.")
        paragraph.drawOn(document, margin, text_top - text_height)

        attribution_parts = []
        if page.get("source_url"):
            attribution_parts.append("Biography source: English Wikipedia")
        if page.get("attribution"):
            attribution_parts.append(f"Portrait: {page['attribution']}")
        attribution_parts.append("Line art generated with AI; verify before publication.")

        document.setFont(self.text_font, 7)
        footer_y = 42
        for line in textwrap.wrap(" | ".join(attribution_parts), width=130)[:3]:
            document.drawString(margin, footer_y, line)
            footer_y -= 9

        if page.get("source_url"):
            document.linkURL(
                page["source_url"],
                (margin, 36, page_width - margin, 60),
                relative=0,
            )

    def create_page(
        self,
        image_path,
        text,
        output_path,
        title="",
        attribution=None,
        source_url=None,
    ):
        page = {
            "image_path": image_path,
            "text": text,
            "title": title,
            "attribution": attribution,
            "source_url": source_url,
        }
        return self.create_book([page], output_path)

    def create_book(self, pages, output_path):
        """Create one multi-page PDF from page dictionaries."""
        output_path = self._ensure_pdf_path(output_path)
        if not pages:
            return False
        temporary = tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".pdf", delete=False)
        temporary_path = Path(temporary.name)
        temporary.close()
        document = canvas.Canvas(str(temporary_path), pagesize=A4)
        document.setTitle(pages[0].get("title", "AI Coloring Book") if len(pages) == 1 else "AI Coloring Book")
        document.setAuthor("AI Coloring Book research prototype")
        document.setSubject("AI-generated biographical coloring-book pages")
        try:
            for index, page in enumerate(pages):
                self._draw_page(document, page)
                if index < len(pages) - 1:
                    document.showPage()
            document.save()
            os.replace(temporary_path, output_path)
            return True
        except Exception as exc:
            print(f"Error creating PDF {output_path}: {exc}")
            return False
        finally:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_image = "images/Marie_Curie_output.png"
    if Path(test_image).exists():
        Concatenator().create_page(
            image_path=test_image,
            text="Marie Curie was a pioneering physicist and chemist who studied radioactivity.",
            output_path="output/test_concatenator_output.pdf",
            title="Marie Curie",
        )
