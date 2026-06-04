from Fetcher import get_person_info
from Summarizer import Summarizer
from Generator import ColoringPageGenerator
from Concatenator import Concatenator


def main():
    names = ["Marie Curie"]  # Add the names you want to process to this list

    summarizer = Summarizer()
    generator = ColoringPageGenerator()
    concatenator = Concatenator()

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
            print(f"Summary generated for {name}.")
            
            # Generate coloring page
            clean_name = person_info["title"].replace(" ", "_")
            image_output_path = f"images/{clean_name}_output.png"
            generator.process_image(person_info["image_path"], image_output_path)
            print(f"Successfully created coloring page at {image_output_path}")

            # Concatenate image and text
            os.makedirs("output", exist_ok=True)
            final_page_path = f"output/{clean_name}.pdf"
            print("Combining image and text into final page...")
            success = concatenator.create_page(
                image_path=image_output_path,
                text=summary,
                output_path=final_page_path,
                title=person_info["title"]
            )
            
            if success:
                print(f"Final coloring book page saved to {final_page_path}")

        except Exception as e:
            print(f"An error occurred while processing {name}: {e}")

if __name__ == "__main__":
    main()