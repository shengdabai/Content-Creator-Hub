import base64
import ipaddress
import mimetypes
import re
import socket
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse
from urllib.request import getproxies_environment, proxy_bypass_environment

import urllib3
from bs4 import BeautifulSoup

ALLOWED_SCHEMES = {"http", "https"}
BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
MAX_PAGE_BYTES = 2_000_000


def normalize_text(text: str) -> str:
    if not text:
        return ""
    clean = text.replace("\r", "\n")
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    return clean.strip()


def _is_public_ip(ip_text: str) -> bool:
    ip = ipaddress.ip_address(ip_text)
    return ip.is_global and not ip.is_multicast


def _validated_target(url: str):
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError("Only http and https URLs are allowed")
    if not parsed.netloc:
        raise ValueError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise ValueError("Authenticated URLs are not allowed")

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname or hostname in BLOCKED_HOSTS:
        raise ValueError("URL hostname is not allowed")
    # Accessing port also validates malformed/out-of-range values.
    port = parsed.port or (443 if parsed.scheme.lower() == 'https' else 80)

    try:
        resolved = {info[4][0] for info in socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)}
    except socket.gaierror as exc:
        raise ValueError("Unable to resolve URL hostname") from exc

    if not resolved or any(not _is_public_ip(ip) for ip in resolved):
        raise ValueError("URL resolves to a non-public address")

    return parsed, port, sorted(resolved)[0]


def _env_proxy(parsed):
    """Return the environment proxy URL for this target, or None (honours NO_PROXY)."""
    proxies = getproxies_environment()
    if proxy_bypass_environment(parsed.hostname or '', proxies):
        return None
    return proxies.get(parsed.scheme.lower()) or proxies.get('all') or None


def validate_public_http_url(url: str) -> str:
    return _validated_target(url)[0].geturl()


def extract_web_content(url: str, timeout_seconds: int = 15) -> Dict[str, str]:
    parsed, port, address = _validated_target(url)
    options = {'timeout': urllib3.Timeout(total=timeout_seconds), 'maxsize': 1}
    proxy_url = _env_proxy(parsed)
    pool = None
    response = None
    try:
        if proxy_url:
            # Proxy mode: the host was already checked locally; the proxy does its own DNS.
            # Redirects are not followed, so every non-2xx hop is rejected below.
            pool = urllib3.ProxyManager(proxy_url, **options)
            response = pool.urlopen('GET', parsed.geturl(), headers={'User-Agent': 'ContentCreatorHub/1.0'},
                                    redirect=False, retries=False, preload_content=False)
        else:
            # Connect to the checked IP, without a second DNS lookup or environment proxy.
            # HTTPS still verifies the certificate and sends SNI for the original host.
            if parsed.scheme.lower() == 'https':
                pool = urllib3.HTTPSConnectionPool(address, port, server_hostname=parsed.hostname,
                                                 assert_hostname=parsed.hostname, **options)
            else:
                pool = urllib3.HTTPConnectionPool(address, port, **options)
            target = parsed.path or '/'
            if parsed.query:
                target += '?' + parsed.query
            response = pool.urlopen('GET', target, headers={'Host': parsed.netloc, 'User-Agent': 'ContentCreatorHub/1.0'},
                                    redirect=False, retries=False, preload_content=False)
        if not 200 <= response.status < 300:
            raise ValueError('URL returned an error or redirect')
        body = response.read(MAX_PAGE_BYTES + 1, decode_content=True)
        if len(body) > MAX_PAGE_BYTES:
            raise ValueError('Web page is too large')
    finally:
        if response is not None:
            response.close()
        if pool is not None:
            pool.close()
    soup = BeautifulSoup(body, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "footer", "nav"]):
        tag.extract()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else ""

    article = soup.find("article")
    if article:
        content = article.get_text("\n", strip=True)
    else:
        blocks = soup.find_all(["h1", "h2", "h3", "p", "li"])
        content = "\n".join(b.get_text(" ", strip=True) for b in blocks)

    return {
        "title": normalize_text(title)[:300],
        "text": normalize_text(content)[:12000],
    }


def image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"
