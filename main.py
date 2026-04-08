from Fetcher import get_person_info
from Summarizer import Summarizer
from Generator import ColoringPageGenerator


def main():
    # Get person info
    person_info = get_person_info("Marie Curie", fuzzy_search=True, save_folder="images")

    # Summarize the text
    summarizer = Summarizer()
    summary = summarizer.summarize(person_info["summary"])
    print(summary)
    
    # Generate coloring page
    generator = ColoringPageGenerator()
    generator.process_image(
        person_info["image_path"],
        "images/" + person_info["title"] + "_output.png",
    )

if __name__ == "__main__":
    main()