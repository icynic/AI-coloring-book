import wikipediaapi
import requests
import urllib.parse
import os

# Define the User-Agent globally so we can use it across functions
USER_AGENT = "AIColoringBook"

def get_person_info(query, fuzzy_search=True, save_folder=None):
    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")
    title = query

    if fuzzy_search:
        # 1. Search using the raw Wikimedia API
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        }

        response = requests.get(
            search_url, params=params, headers={"User-Agent": USER_AGENT}
        )
        results = response.json()

        # Extract the first title from the results
        if results[1]:
            title = results[1][0]
        else:
            print("Error: No page found.")
            return {"title": title, "summary": None, "image_url": None, "image_path": None}

    # 2. Get the page summary using wikipediaapi
    page = wiki.page(title)
    if not page.exists():
        print("Error: No page found.")
        return {"title": title, "summary": None, "image_url": None, "image_path": None}
    
    summary = page.summary

    # 3. Get the portrait/main image using Wikipedia's REST API
    image_url = None
    formatted_title = urllib.parse.quote(title.replace(" ", "_"))
    rest_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_title}"
    
    rest_response = requests.get(rest_url, headers={"User-Agent": USER_AGENT})
    
    if rest_response.status_code == 200:
        page_data = rest_response.json()
        if "originalimage" in page_data:
            image_url = page_data["originalimage"]["source"]
        elif "thumbnail" in page_data:
            image_url = page_data["thumbnail"]["source"]

    image_path = None
    if save_folder:
        image_path = _download_image(image_url, title, save_folder=save_folder)

    return {
        "title": title,
        "summary": summary,
        "image_url": image_url,
        "image_path": image_path
    }

def _download_image(image_url, title, save_folder=None):
    if not image_url:
        print("No image URL found to download.")
        return None

    # Parse the URL to get the file extension (e.g., .jpg, .png)
    parsed_url = urllib.parse.urlparse(image_url)
    _, ext = os.path.splitext(parsed_url.path)
    if not ext:
        ext = ".jpg"

    # Create a safe filename (replace spaces and slashes)
    safe_title = title.replace(" ", "_").replace("/", "_")
    filename = f"{safe_title}{ext}"
    
    if save_folder and save_folder != ".":
        os.makedirs(save_folder, exist_ok=True)
        filepath = os.path.join(save_folder, filename)
    else:
        filepath = filename

    # print(f"Downloading image to: {filepath}")
    
    try:
        response = requests.get(image_url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        
        with open(filepath, 'wb') as file:
            file.write(response.content)

        print(f"Saved image to: {filepath}")
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")
        return None

if __name__ == "__main__":
    result = get_person_info("Marie Curie", # Required
                            fuzzy_search=True, # Optional
                            save_folder="images") # Optional, only saves if folder is provided
    
    print(result['title'])
    print(result['summary'])
    print(result['image_url'])
    print(result['image_path'])