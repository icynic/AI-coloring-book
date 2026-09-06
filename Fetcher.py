"""Fetch biographical text, a lead image, and reproducibility metadata."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import os
import re
import time
import urllib.parse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from source_text import select_article_text


WIKIMEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = os.environ.get(
    "AICOLORINGBOOK_USER_AGENT",
    "AIColoringBook/1.0 (academic research project; contact via project repository)",
)
REQUEST_TIMEOUT = 30
REQUEST_DELAY_SECONDS = float(os.environ.get("AICOLORINGBOOK_REQUEST_DELAY", "0.75"))
_last_request_started = 0.0


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=5,
        connect=3,
        read=3,
        status=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def _get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """Issue a rate-limited GET; the session adapter handles transient retries."""
    global _last_request_started
    wait_seconds = REQUEST_DELAY_SECONDS - (time.monotonic() - _last_request_started)
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    _last_request_started = time.monotonic()
    return session.get(url, **kwargs)


def _empty_result(title: str) -> dict:
    return {
        "title": title,
        "summary": None,
        "image_url": None,
        "image_path": None,
        "page_id": None,
        "revision_id": None,
        "revision_timestamp": None,
        "page_url": None,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "image_title": None,
        "image_license": None,
        "image_license_url": None,
        "image_artist": None,
        "image_credit": None,
        "source_text_sha256": None,
        "image_sha256": None,
        "image_sha1": None,
        "image_timestamp": None,
        "image_user": None,
    }


def _plain_text(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip() or None


def _extmetadata_value(metadata: dict, key: str) -> str | None:
    entry = metadata.get(key) or {}
    return _plain_text(entry.get("value"))


def _fetch_article_text(session, revision_id, page_id):
    """Read the saved revision, not a potentially newer live lead extract."""
    if type(revision_id) is not int or revision_id < 1 or type(page_id) is not int or page_id < 1:
        raise ValueError("A saved page ID and revision ID are required for reproducible text recovery.")
    response = _get(session, WIKIMEDIA_API_URL, params={
        "action": "parse", "oldid": revision_id, "prop": "text|revid",
        "disableeditsection": 1, "disablelimitreport": 1,
        "format": "json", "formatversion": 2,
    }, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    if payload.get("error"):
        raise ValueError(f"Wikipedia parse failed: {payload['error']}")
    parsed = payload.get("parse") or {}
    if parsed.get("revid") != revision_id or parsed.get("pageid") != page_id:
        raise ValueError("Wikipedia returned a different page or revision; refusing to replace the source.")
    article_html = parsed.get("text")
    if not isinstance(article_html, str):
        raise ValueError("Wikipedia did not return article HTML.")
    return select_article_text(article_html) | {
        "text_retrieved_at": datetime.now(timezone.utc).isoformat(),
        "text_source_url": f"https://en.wikipedia.org/w/index.php?oldid={revision_id}",
    }


def refresh_person_text(source):
    """Text-only recovery; never request an image or change portrait metadata."""
    # Bound transient retries; a 429 is reported immediately instead of waiting
    # on repeated Retry-After delays. Rerun later if Wikimedia rate-limits this IP.
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, connect=2, read=1, status=2, backoff_factor=1,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}), respect_retry_after_header=False,
        )))
        text = _fetch_article_text(session, source.get("revision_id"), source.get("page_id"))
    return {**source, **text}


def _fetch_page_metadata(session: requests.Session, title: str) -> dict:
    """Return the exact Wikipedia revision and Wikimedia image attribution."""
    response = _get(
        session,
        WIKIMEDIA_API_URL,
        params={
            "action": "query",
            "prop": "info|revisions|pageimages",
            "inprop": "url",
            "rvprop": "ids|timestamp",
            "piprop": "name|original|thumbnail",
            "pithumbsize": 1600,
            "redirects": 1,
            "titles": title,
            "format": "json",
            "formatversion": 2,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return {}

    page = pages[0]
    revision = (page.get("revisions") or [{}])[0]
    image_title = page.get("pageimage")
    result = {
        "title": page.get("title", title),
        "page_id": page.get("pageid"),
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
        "page_url": page.get("fullurl"),
        "image_title": image_title,
        "image_url": (page.get("original") or page.get("thumbnail") or {}).get("source"),
    }
    result.update(_fetch_article_text(session, result["revision_id"], result["page_id"]))

    if not image_title:
        return result

    image_response = _get(
        session,
        WIKIMEDIA_API_URL,
        params={
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|sha1|timestamp|user",
            "titles": f"File:{image_title}",
            "format": "json",
            "formatversion": 2,
        },
        timeout=REQUEST_TIMEOUT,
    )
    image_response.raise_for_status()
    image_pages = image_response.json().get("query", {}).get("pages", [])
    image_info = ((image_pages[0].get("imageinfo") or [{}])[0]) if image_pages else {}
    metadata = image_info.get("extmetadata") or {}

    result["image_url"] = image_info.get("url") or result["image_url"]
    result.update(
        {
            "image_license": _extmetadata_value(metadata, "LicenseShortName")
            or _extmetadata_value(metadata, "UsageTerms"),
            "image_license_url": _extmetadata_value(metadata, "LicenseUrl"),
            "image_artist": _extmetadata_value(metadata, "Artist"),
            "image_credit": _extmetadata_value(metadata, "Credit"),
            "image_sha1": image_info.get("sha1"),
            "image_timestamp": image_info.get("timestamp"),
            "image_user": image_info.get("user"),
        }
    )
    return result


def get_person_info(query, fuzzy_search=True, save_folder=None):
    """Fetch bounded biographical prose from the article and its main image.

    Metadata is returned with every sample so experiments can be traced to an
    exact Wikipedia revision and the source image can be attributed correctly.
    """
    session = _build_session()
    title = query

    try:
        if fuzzy_search:
            response = _get(
                session,
                WIKIMEDIA_API_URL,
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            results = response.json()
            if len(results) < 2 or not results[1]:
                print(f"Error: no Wikipedia page found for {query!r}.")
                return _empty_result(title)
            title = results[1][0]

        metadata = _fetch_page_metadata(session, title)
        if not metadata.get("page_id"):
            print(f"Error: no Wikipedia page found for {title!r}.")
            return _empty_result(title)

        title = metadata.get("title", title)
        result = _empty_result(title)
        result.update(metadata)
        if result.get("summary"):
            result["source_text_sha256"] = hashlib.sha256(
                result["summary"].encode("utf-8")
            ).hexdigest()

        if save_folder:
            result["image_path"] = _download_image(
                result.get("image_url"), title, save_folder=save_folder, session=session
            )
            if result["image_path"]:
                with open(result["image_path"], "rb") as image_file:
                    result["image_sha256"] = hashlib.sha256(image_file.read()).hexdigest()
        return result
    except (requests.RequestException, ValueError) as exc:
        print(f"Failed to fetch {query!r}: {exc}")
        result = _empty_result(title)
        result["error"] = str(exc)
        return result


def _download_image(image_url, title, save_folder=None, session=None):
    if not image_url:
        print("No image URL found to download.")
        return None

    parsed_url = urllib.parse.urlparse(image_url)
    _, ext = os.path.splitext(parsed_url.path)
    ext = ext if ext and len(ext) <= 6 else ".jpg"
    safe_title = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")
    filename = f"{safe_title}{ext}"
    filepath = os.path.join(save_folder, filename) if save_folder else filename
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    try:
        client = session or _build_session()
        response = _get(client, image_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        with open(filepath, "wb") as file:
            file.write(response.content)
        print(f"Saved image to: {filepath}")
        return filepath
    except requests.RequestException as exc:
        print(f"Failed to download image: {exc}")
        return None


if __name__ == "__main__":
    person = get_person_info("Marie Curie", fuzzy_search=True, save_folder="images")
    for key, value in person.items():
        print(f"{key}: {value}")
