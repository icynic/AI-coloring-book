"""Deterministic, model-free selection of bounded Wikipedia prose."""

from html.parser import HTMLParser
import hashlib
import re

from summary_validation import split_source_sentences


SOURCE_POLICY_VERSION = 2
SOURCE_MAX_WORDS = 800
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "wbr"}
_SKIP_TAGS = {"table", "script", "style", "sup", "ul", "ol", "dl", "figure"}
_SKIP_CLASSES = {"infobox", "navbox", "sidebar", "hatnote", "metadata", "reflist", "reference",
                 "toc", "thumb", "mw-editsection", "noprint", "shortdescription"}
_EXCLUDED_SECTION = re.compile(
    r"^(references|notes|footnotes|citations|sources|bibliography|further reading|external links|see also)$", re.I
)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ArticleParagraphs(HTMLParser):
    """Keep paragraphs with their heading path; exclude navigation and references."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.headings = []
        self.paragraph = None
        self.heading = None
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        skipped = bool(self.stack and self.stack[-1][1]) or tag in _SKIP_TAGS or bool(classes & _SKIP_CLASSES)
        if tag in _VOID_TAGS:
            if tag == "br" and self.paragraph is not None and not skipped:
                self.paragraph.append(" ")
            return
        self.stack.append((tag, skipped))
        if not skipped:
            if tag == "p":
                self.paragraph = []
            elif re.fullmatch(r"h[2-6]", tag):
                self.heading = (int(tag[1]), [])

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data):
        if self.stack and self.stack[-1][1]:
            return
        if self.heading is not None:
            self.heading[1].append(data)
        elif self.paragraph is not None:
            self.paragraph.append(data)

    def handle_endtag(self, tag):
        skipped = bool(self.stack and self.stack[-1][1])
        if not skipped and tag == "p" and self.paragraph is not None:
            text = re.sub(r"\s+", " ", "".join(self.paragraph)).strip()
            if text and not any(_EXCLUDED_SECTION.fullmatch(title) for _, title in self.headings):
                self.paragraphs.append({"section": " / ".join(title for _, title in self.headings) or "Lead",
                                        "text": text, "paragraph_index": len(self.paragraphs)})
            self.paragraph = None
        elif not skipped and self.heading is not None and tag == f"h{self.heading[0]}":
            level, chunks = self.heading
            title = re.sub(r"\s+", " ", "".join(chunks)).strip()
            self.headings = [(n, value) for n, value in self.headings if n < level] + [(level, title)]
            self.heading = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break


def _priority(paragraph):
    text = paragraph["text"]
    section = paragraph["section"]
    if re.search(r"\bmarburg\b", text, re.I):
        return 0
    if section == "Lead":
        return 1
    if re.search(r"\b(published|discovered|developed|awarded|prize|nobel|contribution)\b", text, re.I):
        return 2
    if re.search(r"life|biograph|career|education|work|research|achievement", section, re.I):
        return 3
    return 4


def select_article_text(article_html, max_words=SOURCE_MAX_WORDS):
    if max_words < 1:
        raise ValueError("Source word budget must be positive.")
    parser = ArticleParagraphs()
    parser.feed(article_html)
    parser.close()
    passages = []
    seen = set()
    remaining = max_words
    lead_remaining = min(180, max_words)
    for paragraph in sorted(parser.paragraphs, key=lambda item: (_priority(item), item["paragraph_index"])):
        # Unclassified sections are a fallback for sparse articles, not padding
        # once enough biographical prose has already been found.
        if _priority(paragraph) == 4 and max_words - remaining >= 200:
            continue
        # Select complete sentences, never cut a word or a sentence mid-way.
        budget = min(200, remaining, lead_remaining if paragraph["section"] == "Lead" else max_words)
        selected = []
        for sentence in split_source_sentences(paragraph["text"]):
            count = len(sentence.split())
            if sentence.casefold() in seen or count > budget:
                continue
            selected.append(sentence)
            seen.add(sentence.casefold())
            budget -= count
            remaining -= count
            if paragraph["section"] == "Lead":
                lead_remaining -= count
        if selected:
            passages.append({**paragraph, "text": " ".join(selected)})
    passages.sort(key=lambda item: item["paragraph_index"])
    text = "\n\n".join(item["text"] for item in passages)
    if not text:
        raise ValueError("The article contains no usable prose within the source word budget.")
    return {
        # Keep this legacy key as the actual Qwen input, not merely the lead.
        "summary": text,
        "lead_summary": "\n\n".join(item["text"] for item in parser.paragraphs if item["section"] == "Lead"),
        "source_text_sha256": text_sha256(text),
        "source_word_count": len(text.split()),
        "source_policy_version": SOURCE_POLICY_VERSION,
        "source_text_kind": "selected_article_paragraphs",
        "source_max_words": max_words,
        "source_passages": passages,
    }
