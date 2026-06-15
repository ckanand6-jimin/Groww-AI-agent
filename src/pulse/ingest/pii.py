import re

# Simple PII patterns
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
PHONE_PATTERN = re.compile(r'\+?\d[\d -]{8,12}\d')
URL_TOKEN_PATTERN = re.compile(r'(https?://[^\s]+)')

def scrub_pii(text: str) -> str:
    """
    Remove common PII from the text:
    - Emails
    - Phone numbers
    - URLs (they might contain auth tokens or tracking parameters)
    """
    if not text:
        return text

    # Replace emails
    text = EMAIL_PATTERN.sub('[EMAIL]', text)
    
    # Replace URLs
    text = URL_TOKEN_PATTERN.sub('[URL]', text)
    
    # Replace phone numbers. Be cautious with simple digit strings,
    # so we require a certain length.
    text = PHONE_PATTERN.sub('[PHONE]', text)
    
    return text
