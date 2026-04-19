from Fetcher import get_person_info
from Summarizer import Summarizer
from Generator import ColoringPageGenerator


def main():
    names = ["Marie Curie", "Max Planck"]  # Add the names you want to process to this list

    summarizer = Summarizer()
    generator = ColoringPageGenerator()

    for name in names:
        print(f"\n--- Processing {name} ---")
        try:
            # Get person info
            person_info = get_person_info(name, fuzzy_search=True, save_folder="images")

            if not person_info:
                print(f"Could not fetch info for {name}, skipping...")
                continue

            # Summarize the text
            summary = summarizer.summarize(person_info["summary"])
            print(f"Summary:\n{summary}")
            
            # Generate coloring page
            output_path = "images/" + person_info["title"].replace(" ", "_") + "_output.png"
            generator.process_image(person_info["image_path"], output_path)
            print(f"Successfully created coloring page at {output_path}")

        except Exception as e:
            print(f"An error occurred while processing {name}: {e}")

if __name__ == "__main__":
    main()