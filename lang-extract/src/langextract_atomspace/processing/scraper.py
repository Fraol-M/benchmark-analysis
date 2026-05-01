import re
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; LangExtractAtomSpace/0.1; +https://metta-lang.dev/)"
)
_URL_MAX_PARAGRAPHS = 12
_URL_MAX_CHARS = 8_000
_URL_SAFE_MAX_WORKERS = 1


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _limit_url_paragraphs(
    paragraphs: list[str],
    *,
    max_paragraphs: int,
    max_chars: int,
) -> tuple[list[str], bool]:
    limited: list[str] = []
    used_chars = 0

    for paragraph in paragraphs:
        if len(limited) >= max_paragraphs:
            return limited, True

        added_chars = len(paragraph) + (2 if limited else 0)
        if used_chars + added_chars > max_chars:
            if not limited:
                limited.append(paragraph[:max_chars].rstrip())
            return limited, True

        limited.append(paragraph)
        used_chars += added_chars

    return limited, False


def extract_text_from_url(
    url: str,
    timeout: int = 20,
    *,
    max_paragraphs: int = _URL_MAX_PARAGRAPHS,
    max_chars: int = _URL_MAX_CHARS,
) -> dict[str, Any]:
    """
    Fetch a public webpage and extract readable text content for downstream
    processing. Supports public http/https HTML pages.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only public http/https URLs are supported right now.")
    if not parsed.netloc:
        raise ValueError("The URL is missing a hostname.")

    request = Request(
        url,
        headers={
            "User-Agent": _DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            html_bytes = response.read()
    except HTTPError as exc:
        raise ValueError(f"Website returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise ValueError(f"Could not fetch URL: {exc.reason}") from exc

    if "html" not in content_type.lower():
        raise ValueError("This URL does not look like an HTML webpage.")

    html = html_bytes.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "iframe", "form"]):
        tag.decompose()
    for tag_name in ["header", "footer", "nav", "aside"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = _clean_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""

    main_root = soup.find("article") or soup.find("main") or soup.body or soup
    paragraphs: list[str] = []
    for tag in main_root.find_all(["h1", "h2", "h3", "p", "li"]):
        text = _clean_whitespace(tag.get_text(" ", strip=True))
        if len(text) >= 40:
            paragraphs.append(text)

    if not paragraphs:
        body_text = _clean_whitespace(main_root.get_text("\n", strip=True))
        paragraphs = [chunk.strip() for chunk in body_text.split("\n") if len(chunk.strip()) >= 40]

    deduped: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        if paragraph not in seen:
            deduped.append(paragraph)
            seen.add(paragraph)

    original_text = "\n\n".join(deduped)
    limited_paragraphs, truncated = _limit_url_paragraphs(
        deduped,
        max_paragraphs=max_paragraphs,
        max_chars=max_chars,
    )
    extracted_text = "\n\n".join(limited_paragraphs)
    if not extracted_text.strip():
        raise ValueError("Could not extract readable text from that page.")

    return {
        "url": url,
        "title": title,
        "text": extracted_text,
        "paragraphs": limited_paragraphs,
        "content_type": content_type,
        "paragraph_count": len(limited_paragraphs),
        "char_count": len(extracted_text),
        "original_paragraph_count": len(deduped),
        "original_char_count": len(original_text),
        "truncated": truncated,
    }
