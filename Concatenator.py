from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import textwrap

class Concatenator:
    def __init__(self):
        # ReportLab doesn't require pre-loading standard fonts like Helvetica
        pass

    def create_page(self, image_path, text, output_path, title=""):
        """
        Combines the generated coloring image and summary text into a single A4 PDF.
        
        :param image_path: Path to the generated coloring page image.
        :param text: The summary text to include.
        :param output_path: Path to save the final PDF document.
        :param title: Optional title (e.g., the person's name) to put at the top.
        """
        # Ensure the output is a .pdf
        if not output_path.lower().endswith(".pdf"):
            output_path = output_path.rsplit(".", 1)[0] + ".pdf"

        c = canvas.Canvas(output_path, pagesize=A4)
        page_width, page_height = A4
        
        margin = 50
        
        # Register a unicode-compatible font like Arial
        try:
            pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
            pdfmetrics.registerFont(TTFont('Arial-Bold', 'arialbd.ttf'))
            title_font = 'Arial-Bold'
            text_font = 'Arial'
        except Exception:
            # Fallback to standard Helvetica if Arial isn't found
            title_font = 'Helvetica-Bold'
            text_font = 'Helvetica'
        
        # Draw Title
        if title:
            c.setFont(title_font, 24)
            title_width = c.stringWidth(title, title_font, 24)
            title_x = (page_width - title_width) / 2
            c.drawString(title_x, page_height - margin - 20, title)
            
        # Draw Image
        try:
            img = ImageReader(image_path)
            img_width, img_height = img.getSize()
            
            # Restrict image to fit within margins and take up about 55% of the page height
            max_img_width = page_width - (2 * margin)
            max_img_height = page_height * 0.55
            
            aspect = img_height / float(img_width)
            draw_width = min(max_img_width, max_img_height / aspect)
            draw_height = draw_width * aspect
            
            img_x = (page_width - draw_width) / 2
            img_y = page_height - margin - 60 - draw_height
            
            c.drawImage(img, img_x, img_y, width=draw_width, height=draw_height)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return False

        # Draw Summary Text
        c.setFont(text_font, 12)
        text_y = img_y - 40 # Start text a bit below the image
        
        # Wrap text. For Arial/Helvetica 12, roughly 85 characters fit nicely across A4 with 50pt margins
        wrapped_text = textwrap.wrap(text, width=85)
        
        for line in wrapped_text:
            if text_y < margin:
                break # Prevent drawing off the bottom of the page
            c.drawString(margin, text_y, line)
            text_y -= 16 # Line height increment

        c.save()
        return True

if __name__ == "__main__":
    import os
    # Example values for testing
    test_image = "images/Marie_Curie_output.png"
    test_text = "Marie Curie was a Polish and French chemist and physicist who shared the 1903 Nobel Prize with her husband Pierre for discovering radioactivity. She was the first woman to win a Nobel Prize and the first to win it twice, discovering radium and polonium. She was the first woman to become a professor at the University of Paris and named the element polonium after her native country. While studying in Paris, she taught her daughters the Polish language and named the element polonium after Poland. She passed away in France in 1934 at 66, and was the first woman to be entombed on her own merits in the Paris Pantheon."
    test_output = "output/test_concatenator_output.pdf"
    test_title = "Marie Curie"

    os.makedirs("output", exist_ok=True)
    
    if os.path.exists(test_image):
        print(f"Testing Concatenator with {test_image}...")
        concatenator = Concatenator()
        success = concatenator.create_page(test_image, test_text, test_output, test_title)
        
        if success:
            print(f"Test successful! PDF saved to {test_output}")
        else:
            print("Test failed during PDF creation.")
    else:
        print(f"Test image not found at '{test_image}'.")
        print("Please run main.py first to generate an image or update the 'test_image' path in this block.")
