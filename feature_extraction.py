"""
Feature Extraction Module for Phishing Website Detection
Analyzes URL characteristics purely locally using string & regex analysis.
Does NOT send network requests or crawl target websites.
"""

import re
from urllib.parse import urlparse

# Exact list of features in order used for ML model training and inference
FEATURE_NAMES = [
    'url_length',
    'num_dots',
    'num_subdomains',
    'num_hyphens',
    'num_special_chars',
    'has_at_symbol',
    'has_ip_address',
    'use_https',
    'num_digits',
    'num_params',
    'url_depth',
    'has_suspicious_keyword'
]

# Keywords commonly found in phishing/spoofed URLs
SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'bank', 'update', 'account', 'secure', 'webscr', 
    'cmd', 'signin', 'banking', 'paypal', 'ebay', 'amazon', 'confirm', 
    'credential', 'pay', 'security', 'wallet', 'checkpoint', 'service', 
    'validation', 'auth', 'recovery', 'pass', 'logon'
]


def has_ip(domain: str) -> int:
    """Check if domain or URL contains an IP address (IPv4)."""
    # Regex for standard IPv4 address
    ip_pattern = r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'
    # Also check if domain contains IP anywhere
    ip_regex = r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}'
    if re.match(ip_pattern, domain) or re.search(ip_regex, domain):
        return 1
    return 0


def count_subdomains(domain: str) -> int:
    """Calculate the number of subdomains in a given hostname/domain."""
    # Remove port if exists
    domain = domain.split(':')[0]
    # Handle IP address domain
    if re.match(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$', domain):
        return 0
    
    parts = [p for p in domain.split('.') if p]
    if len(parts) <= 2:
        return 0
    # e.g., sub1.sub2.example.com -> parts = 4 -> subdomains = 2
    return len(parts) - 2


def extract_feature_dict(url: str) -> dict:
    """
    Extract numerical security features from a URL string.
    Returns a dictionary of raw feature names to numerical values.
    """
    if not url:
        url = ""

    # Ensure url has scheme for parsing if missing
    parsed_url = urlparse(url if '://' in url else 'http://' + url)
    domain = parsed_url.netloc or parsed_url.path.split('/')[0]

    url_lower = url.lower()

    # 1. URL Length
    url_length = len(url)

    # 2. Number of Dots
    num_dots = url.count('.')

    # 3. Number of Subdomains
    num_subdomains = count_subdomains(domain)

    # 4. Number of Hyphens
    num_hyphens = url.count('-')

    # 5. Number of Special Characters (@, ?, =, %, !, _, ~, &, #, +)
    special_chars = set("@?=%!_~&#+")
    num_special_chars = sum(1 for char in url if char in special_chars)

    # 6. Presence of @ symbol
    has_at_symbol = 1 if '@' in url else 0

    # 7. Presence of IP Address
    has_ip_address = has_ip(domain)

    # 8. HTTPS Usage
    use_https = 1 if url_lower.startswith('https://') else 0

    # 9. Number of Digits
    num_digits = sum(1 for char in url if char.isdigit())

    # 10. Number of Query Parameters
    num_params = url.count('&') + url.count('=')

    # 11. URL Path Depth
    path_parts = [p for p in parsed_url.path.split('/') if p]
    url_depth = len(path_parts)

    # 12. Suspicious Keywords
    has_suspicious_keyword = 1 if any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS) else 0

    return {
        'url_length': url_length,
        'num_dots': num_dots,
        'num_subdomains': num_subdomains,
        'num_hyphens': num_hyphens,
        'num_special_chars': num_special_chars,
        'has_at_symbol': has_at_symbol,
        'has_ip_address': has_ip_address,
        'use_https': use_https,
        'num_digits': num_digits,
        'num_params': num_params,
        'url_depth': url_depth,
        'has_suspicious_keyword': has_suspicious_keyword
    }


def extract_features(url: str) -> list:
    """
    Extract features from a URL and return a feature vector (list of values)
    matching the order in FEATURE_NAMES.
    """
    f_dict = extract_feature_dict(url)
    return [f_dict[name] for name in FEATURE_NAMES]


def get_human_readable_features(url: str) -> dict:
    """
    Convert raw feature values into user-friendly, formatted dictionary
    for rendering in the web dashboard interface.
    """
    f_dict = extract_feature_dict(url)
    return {
        "URL Length": f"{f_dict['url_length']} characters",
        "Number of Dots": f_dict['num_dots'],
        "Subdomains Count": f_dict['num_subdomains'],
        "Hyphens Count": f_dict['num_hyphens'],
        "Special Characters": f_dict['num_special_chars'],
        "Suspicious '@' Symbol": "Detected" if f_dict['has_at_symbol'] else "Not Detected",
        "IP Address Routing": "Detected" if f_dict['has_ip_address'] else "Not Detected",
        "HTTPS Encryption": "Detected (Secure)" if f_dict['use_https'] else "Not Detected (HTTP)",
        "Numeric Digits Count": f_dict['num_digits'],
        "Query Parameters Count": f_dict['num_params'],
        "URL Path Depth": f_dict['url_depth'],
        "Suspicious Keywords": "Detected" if f_dict['has_suspicious_keyword'] else "Not Detected"
    }
