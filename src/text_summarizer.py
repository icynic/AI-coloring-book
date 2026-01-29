"""
Wikipedia lead-paragraph fetcher + summarizer pipeline (uses Ollama).
- Attempts direct fetch, has options to fetch content via Jina ai proxy, rest summary, or wiki api fallback however these options usually blocked (403) and fixed yet.
"""

from typing import Optional, Tuple, Dict, Any
import unicodedata
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
import re, time, random, sys, logging
from datetime import datetime
from pathlib import Path
import ollama
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent.parent 
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True) 
LOG_FILE = LOG_DIR / "wikipedia_text_summarizer.log" 
LOG_LEVEL = "DEBUG"
HTML_PREVIEW_LEN = 50000   # how many chars of HTML to log for debugging
REQUEST_TIMEOUT = 15
MAX_ATTEMPTS = 2
BACKOFF_FACTOR = 2.0     # exponential backoff multiplier
USE_JINA_FALLBACK = False    # non-API proxy fallback (r.jina.ai)
USE_REST_FALLBACK = False  # set to allow Wikimedia REST summary fallback  
USE_WAPI_FALLBACK = False    # set Wikipedia API fallback
USER_AGENTS = [
    # A small rotation of user agents to reduce simple blocks
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]
 
def setup_logging(log_file: str = LOG_FILE, log_level: str = LOG_LEVEL):
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    logging.basicConfig(level=level, format=fmt, handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding='utf-8')
    ])
    return logging.getLogger(__name__)

logger = setup_logging()

def make_session(retries: int = 2, backoff_factor: float = 0.5, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(['GET', 'POST'])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def _clean_paragraph(p: Tag) -> Optional[str]:
    """
    Returns cleaned text content of a <p> Tag (or None if empty after cleaning).
    Cleaning steps:
     - removes reference <sup> tags (citations)
     - removes IPA / pronunciation / small/noise spans (common classes on Wikipedia)
     - removes images, tables, script/style
     - collapse whitespace and strip
    """
    # a copy is parsed so we can safely decompose elements without mutating original soup
    tmp = BeautifulSoup(str(p), "html.parser")

    # remove common noisy elements (superscripts, IPA/pronunciation spans, edit/anchor spans, images, etc.)
    selectors_to_remove = [
        "sup",                 # citations like [1]
        "span.IPA",            # IPA spans
        "span.nopopups",
        "span.nowrap",
        "span.rt-commentedText",
        "span.mw-editsection", # edit links
        "small",
        "img",
        "table",
        "script",
        "style",
    ]
    for sel in selectors_to_remove:
        for node in tmp.select(sel):
            node.decompose()

    # also remove stray reference anchors sometimes outside <sup>
    for a in tmp.find_all("a", class_="reference"):
        a.decompose()

    # Get text and normalize whitespace
    text = tmp.get_text(" ", strip=True)
    if not text:
        return None

    # Remove multiple spaces/newlines and normalize Unicode spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove leading/trailing bracketed reference-like leftovers like [1] if any remained
    text = re.sub(r"^\[\d+\]\s*", "", text)
    text = re.sub(r"\s*\[\d+\]$", "", text)

    return text or None

def extract_wikipedia_lead(html: str, min_length: int = 30, logger: Optional[logging.Logger] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    - html: entire HTML of a Wikipedia page
    - min_length: minimum number of characters for a paragraph to be considered a valid lead
    - logger: optional logging.Logger for debug messages

    The function:
    - extracts the page title from <h1 id="firstHeading"> if available
    - finds the content container (mw-parser-output)
    - iterates direct child <p> tags (then deeper if needed)
    - removes reference tags and pronunciation/IPA spans and other noisy tags
    - returns the first paragraph whose cleaned text length >= min_length
        
    Returns: (cleaned paragraph or None, page title or None)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    soup = BeautifulSoup(html, "html.parser")

    # 1) page title
    title_tag = soup.find(id="firstHeading")
    title = title_tag.get_text(strip=True) if title_tag else None

    # 2) content container (Wikipedia normally uses this)
    content = soup.select_one("#mw-content-text .mw-parser-output") or soup.select_one("div.mw-parser-output")
    if content is None:
        logger.debug("mw-parser-output not found; falling back to first <p> in document")
        first_p = soup.find("p")
        if first_p:
            cleaned = _clean_paragraph(first_p)
            return (cleaned if cleaned and len(cleaned) >= min_length else None, title)
        return (None, title)

    # 3) try direct children first (most reliable for lead paragraph location)
    for p in content.find_all("p", recursive=False):
        cleaned = _clean_paragraph(p)
        if cleaned and len(cleaned) >= min_length:
            return (cleaned, title)

    # 4) fall back to any <p> inside content
    for p in content.find_all("p"):
        cleaned = _clean_paragraph(p)
        if cleaned and len(cleaned) >= min_length:
            return (cleaned, title)

    # 5) nothing suitable found
    logger.debug("no suitable lead paragraph found")
    return (None, title)


class WikipediaFetcher:
    def __init__(self, use_jina_fallback: bool = USE_JINA_FALLBACK, use_rest_fallback: bool = USE_REST_FALLBACK, use_Wapi_fallback: bool = USE_WAPI_FALLBACK, proxies: Optional[Dict[str,str]] = None):
        self.session = make_session(retries=1, backoff_factor=0.3)
        self.use_jina_fallback = use_jina_fallback
        self.use_rest_fallback = use_rest_fallback
        self.use_Wapi_fallback = use_Wapi_fallback
        self.proxies = proxies

    def _build_headers(self) -> Dict[str,str]:
        ua = random.choice(USER_AGENTS)
        headers = {
            "User-Agent": ua,
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        return headers

    def _normalize_name(self, name: str) -> str:
        """
        normalizes a name string into a Wikipedia-friendly format
        """
        name = name.strip()
        # Normalize unicode (keeps accents like é), standardize apostrophes & turns separators into spaces
        name = unicodedata.normalize("NFC", name)
        name = name.replace("’", "'").replace("‘", "'").replace("`", "'")
        name = re.sub(r"[_\-–—/]+", " ", name)

        # everything except letters, numbers, spaces, and apostrophes is removed
        name = re.sub(r"[^\w\s']", "", name, flags=re.UNICODE)

        parts = []
        for word in name.split():
            # Capitalize word but keep internal apostrophes
            word = word.capitalize()
            # apostrophes encoded
            word = word.replace("'", "%27")
            parts.append(word)

        return "_".join(parts)
            
    def fetch_direct(self, title: str) -> Tuple[Optional[str], Optional[int], Optional[Dict[str,str]]]:
        """Tries direct GET from en.wikipedia.org."""
        safe_title = title.replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{safe_title}"
        headers = self._build_headers()
        logger.info(f"Attempting direct fetch: {url}")
        logger.debug(f"Request headers: {headers}")

        attempt = 0
        wait = 2.0
        while attempt < MAX_ATTEMPTS:
            attempt += 1
            try:
                logger.info(f"Direct fetch attempt {attempt}/{MAX_ATTEMPTS}")
                r = self.session.get(url, headers=headers, timeout=REQUEST_TIMEOUT, proxies=self.proxies)
                logger.info(f"HTTP Status: {r.status_code}")
                logger.debug(f"Response headers: {dict(r.headers)}")
                # log trimmed html preview for debugging
                logger.info(f"HTML preview (first {HTML_PREVIEW_LEN} chars):\n{r.text[:HTML_PREVIEW_LEN]!s}\n")
                if r.status_code == 200:
                    return r.text, r.status_code, dict(r.headers)
                if r.status_code == 403:
                    logger.warning("Received 403 from Wikipedia (blocked).")
                    # Exponential backoff then retry
                    time.sleep(wait)
                    wait *= BACKOFF_FACTOR
                    continue
                # For other 4xx/5xx, let retry logic in session handle some
                r.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"Direct fetch error: {e}")
                time.sleep(wait)
                wait *= BACKOFF_FACTOR
        logger.error("Direct fetch failed after retries")
        return None, None, None

    def fetch_via_jina(self, title: str) -> Tuple[Optional[str], Optional[int], Optional[Dict[str,str]]]:
        """
        Uses the jina.ai proxy: r.jina.ai/http://en.wikipedia.org/wiki/TITLE
        because this often returns a pre-rendered HTML and can bypass trivial blocks.
        """
        safe_title = title.replace(" ", "_")
        url = f"https://r.jina.ai/http://en.wikipedia.org/wiki/{safe_title}"
        headers = self._build_headers()
        logger.info(f"Attempting jina.ai proxy fetch: {url}")
        logger.debug(f"Request headers: {headers}")

        try:
            r = self.session.get(url, headers=headers, timeout=REQUEST_TIMEOUT, proxies=self.proxies)
            logger.info(f"HTTP Status (jina): {r.status_code}")
            logger.debug(f"Response headers (jina): {dict(r.headers)}")
            logger.info(f"HTML preview (jina, first {HTML_PREVIEW_LEN} chars):\n{r.text[:HTML_PREVIEW_LEN]!s}\n")
            if r.status_code == 200:
                return r.text, r.status_code, dict(r.headers)
        except Exception as e:
            logger.error(f"Jina fetch error: {e}")
        return None, None, None

    def fetch_rest_summary(self, title: str) -> Tuple[Optional[str], Optional[int], Optional[Dict[str,str]]]:
        """
        Optional: Wikimedia REST summary endpoint returns JSON with plain extract summary.
        This is the official REST endpoint (not the search API) and is fairly reliable.
        """
        safe_title = title.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_title}"
        headers = {"User-Agent": random.choice(USER_AGENTS), "Accept": "application/json"}
        logger.info(f"Attempting Wikimedia REST summary: {url}")
        try:
            r = self.session.get(url, headers=headers, timeout=REQUEST_TIMEOUT, proxies=self.proxies)
            logger.info(f"REST status: {r.status_code}")
            logger.debug(f"REST response headers: {dict(r.headers)}")
            if r.status_code == 200:
                data = r.json()
                extract = data.get("extract")
                if extract:
                    # return extract as a pseudo-paragraph + title
                    title_from = data.get("title", safe_title)
                    return extract, r.status_code, dict(r.headers)
            else:
                logger.warning(f"REST summary returned {r.status_code}")
        except Exception as e:
            logger.error(f"REST fetch error: {e}")
        return None, None, None
 
    def fetch_api_summary(self, title: str, lang: str = "en") -> Tuple[Optional[str], Optional[int], Optional[Dict[str, str]]]:
        """
        Fetch the plaintext intro (extract) for a Wikipedia article using the MediaWiki action API.

        Returns:
            (extract_text or None, http_status_code or None, response_headers or None)

        Notes:
        - Uses action=query&prop=extracts with exintro & explaintext to get the plain intro paragraph.
        - Follows redirects (redirects=1).
        - Uses a requests.Session() (self.session if available) and attaches retries.
        """ 
        session = getattr(self, "session", None)
        if session is None:
            session = make_session()

        safe_title = title.strip()
        api_url = f"https://{lang}.wikipedia.org/w/api.php"

        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "titles": safe_title,
            "redirects": "1",
            "formatversion": "2",  # easier to parse: pages as list
        }

        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "application/json",
        }

        logger.info(f"Attempting Wikipedia API summary: {api_url} (title={safe_title})")
        try:
            r = session.get(api_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT, proxies=getattr(self, "proxies", None))
            logger.info(f"Wikipedia API status: {r.status_code}")
            logger.debug(f"Wikipedia API response headers: {dict(r.headers)}")

            data = r.json()
            if "error" in data:
                logger.warning(f"Wikipedia API returned error: {data['error']}")
                return None, r.status_code, dict(r.headers)

            # With formatversion=2, pages is a list
            query = data.get("query", {})
            pages = query.get("pages", [])

            if not pages:
                logger.warning("Wikipedia API returned no pages.")
                return None, r.status_code, dict(r.headers)

            # Use the first page in result (titles param is a single title normally)
            page = pages[0]

            if page.get("missing"):
                logger.warning(f"Wikipedia page missing: {safe_title}")
                return None, r.status_code, dict(r.headers)

            # Disambiguation detection (pageprops.disambiguation is common)
            pageprops = page.get("pageprops", {})
            if pageprops and pageprops.get("disambiguation"):
                logger.info(f"Page is a disambiguation page: {page.get('title')}") 
                return None, r.status_code, dict(r.headers)

            extract = page.get("extract")
            if extract:
                return extract, r.status_code, dict(r.headers)
            else: 
                logger.info(f"No extract found for page: {page.get('title')}")
                return None, r.status_code, dict(r.headers)

        except requests.exceptions.RequestException as re:
            logger.error(f"Wikipedia API request error: {re}")
        except ValueError as ve:
            logger.error(f"JSON decode error from Wikipedia API: {ve}")
        except Exception as e:
            logger.exception(f"Unexpected error fetching Wikipedia API summary: {e}")
 
        return None, None, None

    def fetch(self, title: str) -> Tuple[Optional[str], Optional[str]]:
        """Main fetch orchestration function: tries direct, then jina proxy & optionally REST summary."""
        logger.info(f"Fetching Wikipedia lead paragraph for: {title!r}")
        title = self._normalize_name(title)

        # 1) direct
        html, status, headers = self.fetch_direct(title)
        if html:
            txt, page_title = extract_wikipedia_lead(html, min_length=10, logger=logger)
            if txt:
                return txt, page_title
            else:
                # If HTML fetched but no paragraph found, we still log and try fallback
                logger.warning("Direct fetch returned HTML but no suitable paragraph extracted. Will try fallback(s).")

        # 2) jina.ai proxy fallback (non-API)
        if self.use_jina_fallback:
            html, status, headers = self.fetch_via_jina(title)
            if html:
                txt, page_title = extract_wikipedia_lead(html, min_length=50, logger=logger)
                if txt:
                    logger.info("Successfully extracted paragraph via jina.ai proxy fallback")
                    return txt, page_title
                else:
                    logger.warning("jina.ai proxy returned HTML but no suitable paragraph extracted.")

        # 3) optional REST summary fallback (opt-in)
        if self.use_rest_fallback:
            summary, status, headers = self.fetch_rest_summary(title)
            if summary:
                logger.info("Using REST summary fallback (extract).")
                # REST summary is plain text so we return as-is
                return summary, title.replace("_", " ")
        
        # 4) Wikipedia API
        if self.use_Wapi_fallback:
            summary, status, headers = self.fetch_api_summary(title)
            if summary:
                logger.info("Using Wikipedia API summary fallback (extract).") 
                return summary, title.replace("_", " ")

        logger.error("All fetch strategies failed or returned no paragraph")
        return None, None

class WikipediaSummarizer:
    def __init__(self, model_name: str = "mistral", **fetcher_opts):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.fetcher = WikipediaFetcher(**fetcher_opts)
        self.logger.info(f"Using Ollama model: {self.model_name}")

    def text_summarize(self, text: str, max_sentences: int = 5) -> str:
        if not text:
            return "No text to summarize."
        prompt = f"""Please summarize the following text about the person in a easy to understand and interesting way.
Make exactly {max_sentences} well structured sentences in a single paragraph.

Text:
{text}

Summary:"""
        try:
            resp = ollama.generate(model=self.model_name, prompt=prompt, options={"temperature": 0.7, "num_predict": 400})
            summary = resp.get("response", "").strip()
            # ensures it ends with a period ;)
            if summary and not summary.endswith("."):
                summary += "."
            return summary
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "Unable to generate summary: LLM error."

    def process_name(self, name: str) -> Dict[str, Any]:
        logger.info("="*60)
        logger.info(f"So... {name!r}")
        
        text, page_title = self.fetcher.fetch(name)
        if not text:
            logger.error(f"Could not fetch a lead paragraph for: {name!r}")
            return {"success": False, "error": "fetch_failed", "name": name, "timestamp": datetime.utcnow().isoformat()}

        logger.info(f"Fetched content for page: {page_title!r}")
        logger.debug(f"Original preview (first 400 chars): {text[:400]!r}")

        summary = self.text_summarize(text, max_sentences=5)

        return {
            "success": True,
            "name": name,
            "page_title": page_title,
            "original_text": text,
            "final_summary": summary,
            "model_used": self.model_name,
            "timestamp": datetime.utcnow().isoformat()
        }
 

def main():
    logger.info("Wikipedia Summarizer starting..")
    summarizer = WikipediaSummarizer(model_name="mistral", use_jina_fallback=USE_JINA_FALLBACK, use_rest_fallback=USE_REST_FALLBACK)

    try:
        while True:
            print("\nEnter a name (or 'quit'):")
            name = input("> ").strip()
            if not name:
                continue
            if name.lower() in ("quit", "exit", "q"):
                logger.info("User requested exit.")
                break
            result = summarizer.process_name(name)
            if not result["success"]:
                print("ERROR:", result.get("error"))
            else:
                print("\nPage title:", result["page_title"])
                print("\nOriginal (lead paragraph):\n", result["original_text"])
                print("\nSummarized paragraph:\n", result["final_summary"])
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Shutting down.")

if __name__ == "__main__":
    main()
