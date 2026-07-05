import os
import sys
import re
import json
import base64
import argparse
import urllib.request
import urllib.parse
from html.parser import HTMLParser
import zipfile
import shutil
import copy
import time
import xml.etree.ElementTree as ET
import socket
import struct

# --- Email Domain Validation (MX record check via DNS over socket) ---
def validate_email_domain(email):
    """Validates an email address format and checks if the domain has MX records.
    Returns (is_valid: bool, reason: str)."""
    if not email or not isinstance(email, str):
        return False, "Empty or invalid email"
    
    email = email.strip()
    
    # Basic format check
    email_regex = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_regex, email):
        return False, f"Invalid email format: {email}"
    
    domain = email.split('@')[1].lower()
    
    # Block obviously fake/placeholder domains
    fake_domains = ['example.com', 'test.com', 'placeholder.com', 'company.com', 'domain.com']
    if domain in fake_domains:
        return False, f"Placeholder domain detected: {domain}"
    
    # Check if domain resolves (MX or A record)
    try:
        # Try MX lookup first
        import subprocess
        result = subprocess.run(
            ['nslookup', '-type=MX', domain],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.lower()
        if 'mail exchanger' in output or 'mx preference' in output:
            return True, "MX record found"
        
        # Fallback: check if domain has any A record (some domains accept mail on A record)
        socket.getaddrinfo(domain, 25, socket.AF_INET)
        return True, "Domain resolves (A record)"
    except subprocess.TimeoutExpired:
        return False, f"DNS lookup timed out for {domain}"
    except (socket.gaierror, socket.herror, OSError):
        return False, f"Domain does not resolve: {domain}"
    except Exception as e:
        # If DNS check fails, be permissive — don't block on DNS errors
        return True, f"DNS check inconclusive ({e}), allowing"


# --- Load .env file (lightweight, no external dependency) ---
def _load_dotenv():
    """Reads key=value pairs from a .env file in the script's directory into os.environ."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key, value = key.strip(), value.strip()
            # Don't overwrite existing env vars (CLI/system takes priority)
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

# Gmail API imports
try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from email.message import EmailMessage
except ImportError:
    print("Gmail API libraries not fully installed. Run pip install google-api-python-client google-auth-oauthlib google-auth-httplib2")
    sys.exit(1)

# Google GenAI imports
try:
    from google import genai
    from google.genai import types
    from pydantic import BaseModel, Field
except ImportError:
    print("Google GenAI SDK not installed. Run pip install google-genai")
    sys.exit(1)

# Configure output to support UTF-8 characters on terminal
sys.stdout.reconfigure(encoding='utf-8')

# Define SCOPES for Gmail API
SCOPES = ['https://mail.google.com/']

# Global flag to track Gemini API quota exhaustion
gemini_quota_exceeded = False

# --- OpenRouter Free Tier Helper ---
def openrouter_chat(prompt, json_schema_properties=None, temperature=0.2):
    """Calls OpenRouter's free-tier API (OpenAI-compatible) with optional JSON schema enforcement.
    Uses the 'openrouter/free' router which auto-selects the best free model."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")

    messages = [{"role": "user", "content": prompt}]
    body = {
        "model": "openrouter/free",
        "messages": messages,
        "temperature": temperature,
    }

    # If a JSON schema is provided, enforce structured output
    if json_schema_properties:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": json_schema_properties,
                    "required": list(json_schema_properties.keys()),
                    "additionalProperties": False
                }
            }
        }

    data = json.dumps(body).encode('utf-8')
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Hitesh-singh67",
            "X-Title": "Outreach Pipeline"
        },
        method="POST"
    )

    backoff = 3
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
            content = result["choices"][0]["message"]["content"]
            # Try to parse JSON from the content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try extracting JSON from markdown code block
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
                if match:
                    return json.loads(match.group(1).strip())
                return content
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='ignore')
            if e.code in (429, 503) and attempt < 2:
                print(f"  [!] OpenRouter API busy ({e.code}). Retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                # Recreate the request (data stream is consumed)
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(body).encode('utf-8'),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/Hitesh-singh67",
                        "X-Title": "Outreach Pipeline"
                    },
                    method="POST"
                )
            else:
                raise RuntimeError(f"OpenRouter API error {e.code}: {error_body}")
    raise RuntimeError("OpenRouter: max retries exceeded")


# Define the MatchResult schema for structured JSON output from Gemini
class MatchResult(BaseModel):
    matches_stack: bool = Field(description="True if the job role or company matches the candidate's programming languages (TypeScript, Python, JavaScript, Java, C, C++).")
    reason: str = Field(description="A brief explanation of why this job matches or doesn't match the candidate's profile.")
    company_domain: str = Field(description="The inferred official website domain of the company (e.g. scale.com, stripe.com, tesla.com).")
    contact_email: str = Field(description="The suggested career/hiring contact email for this company domain (e.g. careers@company.com, jobs@company.com, hr@company.com, recruiting@company.com).")
    email_subject: str = Field(description="A personalized, low-friction, high-impact cold email subject line.")
    email_body: str = Field(description="A personalized cold email following the candidate's custom template structure (<350 words), aligning candidate's background with the company's stack.")

# Define Pydantic model for tailored resume content
class TailoredResume(BaseModel):
    skills_line: str = Field(description="The candidate's skills line reordered to put the most relevant languages/frameworks first for this company.")
    experience_bullets: list[str] = Field(description="An array of exactly 4 strings rewritten to emphasize aspects most relevant to this company's stack.")
    project1_description: list[str] = Field(description="An array of exactly 2 strings for SentinelLog-AI project rewritten to emphasize relevance.")
    project2_description: list[str] = Field(description="An array of exactly 2 strings for RAG Engine project rewritten to emphasize relevance.")

class GeminiQuotaExhaustedError(Exception):
    """Raised when Gemini daily API quota is fully exhausted."""
    pass

def gemini_generate_content(client, model, contents, config=None, max_attempts=5):
    """Wraps generate_content with robust handling of 429/503 rate limits and quotas.
    If rate-limited, parses the retry delay or sleeps for 30s before retrying.
    Raises GeminiQuotaExhaustedError immediately if daily limit is hit."""
    backoff = 5
    for attempt in range(max_attempts):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "resource_exhausted" in err_str.lower() or "503" in err_str
            
            if is_rate_limit:
                # Detect daily limit exhaustion (e.g., 'requestsperday' or 'Daily' in quotaId / error)
                if "requestsperday" in err_str.lower() or "perday" in err_str.lower() or "daily" in err_str.lower():
                    print("  [!] Gemini daily quota exhausted (GenerateRequestsPerDay).")
                    raise GeminiQuotaExhaustedError("Gemini daily quota exhausted")
                
                if attempt < max_attempts - 1:
                    # Try to parse the retry delay from the error message
                    # e.g., "Please retry in 26.36s" or "retryDelay: '26s'"
                    delay = backoff
                    match = re.search(r'retry in ([\d\.]+)\s*s', err_str, re.IGNORECASE)
                    if match:
                        delay = float(match.group(1)) + 1.0 # Add a small buffer
                    else:
                        match_info = re.search(r'retryDelay:\s*\'(\d+)s\'', err_str)
                        if match_info:
                            delay = float(match_info.group(1)) + 1.0
                        else:
                            # Default to 30s for resource_exhausted, 10s for 503
                            delay = 30.0 if "429" in err_str or "resource_exhausted" in err_str.lower() else 10.0
                    
                    print(f"  [!] Gemini rate-limited/busy. Sleeping for {delay:.1f}s before retrying (Attempt {attempt+1}/{max_attempts})...")
                    time.sleep(delay)
                    backoff *= 2
                else:
                    raise e
            else:
                raise e

def gemini_structured_chat(client, prompt, response_schema, temperature=0.2):
    """Calls Gemini 2.5 Flash to get structured output conforming to the response_schema."""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=temperature
    )
    response = gemini_generate_content(
        client=client,
        model="gemini-2.5-flash",
        contents=prompt,
        config=config
    )
    if not response or not response.text:
        return None
    
    text = response.text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code block if present
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        raise

def parse_markdown_table_row(line, has_salary=True):
    """Parses a markdown table row and extracts company name, company url, role, location, and apply link."""
    # Split line by '|'
    parts = [p.strip() for p in line.split('|')]
    if len(parts) < 2:
        return None
        
    # Clean up empty split elements at boundaries
    if parts[0] == '':
        parts = parts[1:]
    if parts and parts[-1] == '':
        parts = parts[:-1]
        
    if has_salary:
        if len(parts) < 6:
            return None
        comp_col = parts[0]
        role_col = parts[1]
        loc_col = parts[2]
        apply_col = parts[4]
    else:
        if len(parts) < 5:
            return None
        comp_col = parts[0]
        role_col = parts[1]
        loc_col = parts[2]
        apply_col = parts[3]

    # Parse company name and url
    # e.g., <a href="https://www.nvidia.com"><strong>NVIDIA</strong></a>
    company_name = "Unknown"
    company_url = None
    
    comp_url_match = re.search(r'href="([^"]+)"', comp_col)
    if comp_url_match:
        company_url = comp_url_match.group(1)
        
    comp_name_match = re.search(r'<strong>(.*?)</strong>', comp_col)
    if comp_name_match:
        company_name = comp_name_match.group(1)
    else:
        company_name = re.sub(r'<[^>]+>', '', comp_col).strip()

    # Parse role
    role_text = re.sub(r'<[^>]+>', '', role_col).strip()
    
    # Parse location
    location_text = re.sub(r'<[^>]+>', '', loc_col).strip()

    # Parse apply link
    # e.g., <a href="https://..."><img .../></a>
    apply_link = None
    apply_match = re.search(r'href="([^"]+)"', apply_col)
    if apply_match:
        apply_link = apply_match.group(1)

    # Skip closed positions or missing links
    if not apply_link or apply_link.strip() == "":
        return None
        
    company_text_lower = company_name.lower()
    role_text_lower = role_text.lower()
    if "🚫" in company_text_lower or "🚫" in role_text_lower or "closed" in role_text_lower:
        return None

    return {
        'company': company_name,
        'company_url': company_url,
        'role': role_text,
        'location': location_text,
        'apply_link': apply_link
    }

# Custom HTML parser to extract clean text from job description pages
class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.ignore = False

    def handle_starttag(self, tag, attrs):
        if tag in ['script', 'style', 'head', 'meta', 'link']:
            self.ignore = True

    def handle_endtag(self, tag):
        if tag in ['script', 'style', 'head', 'meta', 'link']:
            self.ignore = False

    def handle_data(self, data):
        if not self.ignore:
            t = data.strip()
            if t:
                self.text.append(t)

    def get_text(self):
        return " ".join(self.text)

def parse_docx_resume(docx_path):
    """Extracts raw text from a .docx file without external dependencies."""
    print(f"[*] Extracting resume content from: {docx_path}...")
    try:
        doc = zipfile.ZipFile(docx_path)
        xml_content = doc.read('word/document.xml')
        root = ET.fromstring(xml_content)
        paragraphs = []
        for p in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
            texts = [t.text for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if t.text]
            if texts:
                paragraphs.append(''.join(texts))
        full_text = '\n'.join(paragraphs)
        print(f"[+] Successfully read resume. Character length: {len(full_text)}")
        return full_text
    except Exception as e:
        print(f"[-] Error parsing resume docx: {e}")
        sys.exit(1)

def fetch_speedyapply_postings():
    """Scrapes speedyapply 2026 Software Engineering internships (USA and International)."""
    print("[*] Fetching active internships from speedyapply/2026-SWE-College-Jobs...")
    
    urls = [
        ("https://raw.githubusercontent.com/speedyapply/2026-SWE-College-Jobs/main/README.md", True),  # has_salary=True
        ("https://raw.githubusercontent.com/speedyapply/2026-SWE-College-Jobs/main/INTERN_INTL.md", False)  # has_salary=False
    ]
    
    all_postings = []
    
    for url, has_salary in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read().decode('utf-8')
        except Exception as e:
            print(f"[-] Failed to download postings from {url}: {e}")
            continue

        lines = content.split('\n')
        current_category = "SWE"
        
        for line in lines:
            line = line.strip()
            if line.startswith('###'):
                header = line.replace('#', '').strip().lower()
                if 'faang' in header:
                    current_category = 'SWE'
                elif 'quant' in header:
                    current_category = 'Quant'
                else:
                    current_category = 'SWE'
            
            if not line.startswith('|'):
                continue
                
            # Skip headers or dividers
            if 'company' in line.lower() or '---|---' in line:
                continue
                
            row_data = parse_markdown_table_row(line, has_salary)
            if row_data:
                row_data['category'] = current_category
                all_postings.append(row_data)
                
    print(f"[+] speedyapply active postings scraped: {len(all_postings)}")
    return all_postings

def search_india_internships(client):
    """Searches Google for active internship postings at Indian STARTUPS where founders/HR have public emails.
    Prioritizes seed/Series A companies, YC-backed startups, and small companies with accessible leadership."""
    print("[*] Searching for Indian startup internships with findable contacts using Gemini Search Grounding...")
    
    prompt = """
Search for 75 INDIAN STARTUPS and SMALL/MID-SIZE COMPANIES that are actively hiring Software Engineer Interns, AI/ML Interns, Data Analyst Interns, or Data Engineer Interns.

IMPORTANT — FOCUS ON STARTUPS, NOT LARGE CORPORATIONS:
- Prioritize seed-stage, Series A/B startups, YC-backed companies, and growing Indian tech companies.
- Look at: YC Startup Directory (India), Wellfound (formerly AngelList) India jobs, Internshala, Unstop, LinkedIn startup jobs.
- DO NOT include large enterprises (Microsoft, Google, Amazon, TCS, Infosys, Wipro, HCL, etc.) — they only use ATS portals and don't accept cold emails.
- Focus on companies where the CEO/CTO/HR person's email is likely publicly available (small teams, startup culture, public team pages).
- Include the company's official website URL so we can find their team/about page.

Target cities: Bangalore/Bengaluru, Mumbai, Delhi NCR (Gurugram/Gurgaon/Noida), Pune, Hyderabad, Chennai, and Remote (India).

For EACH company, also try to find:
- The name and title of the CEO/Founder/CTO
- Any publicly available email from their team page, LinkedIn, or GitHub

Format your response exactly as a JSON array of objects inside a single markdown code block, like this:
```json
[
  {
    "company": "Company Name",
    "company_url": "https://company.com",
    "role": "Role Title (e.g., Software Engineering Intern)",
    "location": "Location (e.g. Bangalore, India or Remote)",
    "apply_link": "https://apply-link-or-careers-page",
    "founder_name": "Founder/CEO name if found, or null",
    "founder_title": "CEO/CTO/Founder, or null",
    "founder_email": "their email if publicly available, or null"
  },
  ...
]
```
Do not output anything else.
"""
    try:
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
        response = gemini_generate_content(
            client=client,
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
            max_attempts=5
        )
        if not response:
            raise Exception("No response received from GenAI API.")
            
        text = response.text
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        json_str = match.group(1) if match else text
        postings_data = json.loads(json_str.strip())
        
        # Save successful results to cache for future fallback
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'india_postings_cache.json')
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(postings_data, f, indent=2)
            print(f"[+] Updated India postings cache with {len(postings_data)} fresh listings.")
        except Exception:
            pass
        
    except Exception as e:
        print(f"[-] Gemini search grounding for India startup jobs failed: {e}")
        print("[*] Falling back to local India postings cache...")
        
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'india_postings_cache.json')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    postings_data = json.load(f)
                print(f"[+] Loaded {len(postings_data)} India listings from local postings cache.")
            except Exception as cache_err:
                print(f"[-] Failed to load local postings cache: {cache_err}")
                return []
        else:
            print("[-] Local India postings cache file not found.")
            return []

    postings = []
    for item in postings_data:
        posting = {
            'category': 'AI/ML' if any(x in item.get('role', '').lower() for x in ['ai', 'ml', 'machine', 'learning']) else 'SWE',
            'company': item.get('company', 'Unknown'),
            'company_url': item.get('company_url', ''),
            'role': item.get('role', 'Intern'),
            'location': item.get('location', 'India'),
            'apply_link': item.get('apply_link', '')
        }
        # Carry forward any founder info discovered during search (will be used in contact discovery)
        if item.get('founder_name'):
            posting['founder_name'] = item['founder_name']
            posting['founder_title'] = item.get('founder_title', '')
            posting['founder_email'] = item.get('founder_email', '')
        postings.append(posting)
    print(f"[+] Found {len(postings)} India startup listings.")
    return postings

def fetch_job_description(url):
    """Tries to scrape the job page and extract raw text description."""
    if not url:
        return ""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as response:
            html = response.read().decode('utf-8', errors='ignore')
        extractor = TextExtractor()
        extractor.feed(html)
        text = extractor.get_text()
        return text[:4000]
    except Exception as e:
        return ""

def check_gmail_token_valid(non_interactive=False):
    """Pre-flight check: verifies Gmail token can be refreshed BEFORE burning API calls.
    Returns True if token is usable, False/exits if not."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.join(script_dir, 'token.json')
    credentials_path = os.path.join(script_dir, 'credentials.json')

    if not os.path.exists(token_path):
        if non_interactive:
            print("\n[-] FATAL: token.json not found and running in non-interactive mode.")
            print("[*] ACTION REQUIRED: Run the script locally once to authenticate, then update")
            print("    the GMAIL_TOKEN_JSON GitHub secret with the new token.json contents.")
            sys.exit(1)
        print("[*] No Gmail token found. Will authenticate during the send phase.")
        return True  # Will be handled by get_gmail_service

    try:
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if creds and creds.valid:
            print("[+] Gmail token is valid.")
            return True
        if creds and creds.expired and creds.refresh_token:
            print("[*] Gmail token expired. Testing refresh...")
            creds.refresh(Request())
            # Save refreshed token
            with open(token_path, 'w') as f:
                f.write(creds.to_json())
            print("[+] Gmail token refreshed successfully.")
            return True
    except Exception as e:
        print(f"[-] Gmail token refresh failed: {e}")

    if non_interactive:
        print("\n[-] FATAL: Gmail token is expired/revoked and cannot be refreshed in non-interactive mode.")
        print("[*] ACTION REQUIRED:")
        print("    1. Run locally:  python discover_and_outreach.py")
        print("    2. Complete the browser OAuth flow.")
        print("    3. Copy the new token.json contents into your GitHub repo secret GMAIL_TOKEN_JSON.")
        print("    4. The daily cron will then work automatically.")
        sys.exit(1)
    else:
        print("[*] Token needs re-authentication. Will handle during the send phase.")
        return True


def get_gmail_service(non_interactive=False):
    """Authenticates the user and returns the Gmail API service instance."""
    creds = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    token_path = os.path.join(script_dir, 'token.json')
    credentials_path = os.path.join(script_dir, 'credentials.json')

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print("[*] Gmail token expired. Attempting to refresh...")
                creds.refresh(Request())
            except Exception as e:
                print(f"[-] Failed to refresh Gmail credentials: {e}")
                print("[*] The token may have been revoked or expired. Re-authenticating...")
                creds = None
        
        if not creds:
            if non_interactive:
                print("\n[-] FATAL: Gmail credentials are expired/invalid in non-interactive mode.")
                print("[*] ACTION REQUIRED:")
                print("    1. Run locally:  python discover_and_outreach.py")
                print("    2. Complete the browser OAuth flow.")
                print("    3. Copy the new token.json contents into your GitHub repo secret GMAIL_TOKEN_JSON.")
                sys.exit(1)
                
            if not os.path.exists(credentials_path):
                print(f"\n[-] Error: '{credentials_path}' not found.")
                print("[*] Please download your OAuth 2.0 Client credentials from the Google Cloud Console.")
                print(f"[*] Save it as 'credentials.json' in: {script_dir} and try again.")
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
            
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def stage_gmail_draft(service, to_email, subject, body_text, resume_path=None):
    """Creates a pending draft in the user's Gmail inbox with resume attached."""
    try:
        message = EmailMessage()
        message.set_content(body_text)
        message["To"] = to_email
        message["Subject"] = subject
        
        # Attach resume if path provided and exists
        if resume_path:
            if os.path.exists(resume_path):
                print(f"     [+] Attaching resume: {resume_path}")
                import mimetypes
                mime_type, _ = mimetypes.guess_type(resume_path)
                if mime_type is None:
                    mime_type = 'application/octet-stream'
                main_type, sub_type = mime_type.split('/', 1)
                
                with open(resume_path, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(resume_path)
                    message.add_attachment(file_data, maintype=main_type, subtype=sub_type, filename=file_name)
            else:
                print(f"     [-] Warning: Resume path '{resume_path}' does not exist! Staging draft WITHOUT attachment.")
        
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"message": {"raw": encoded_message}}
        
        draft = service.users().drafts().create(userId="me", body=create_message).execute()
        return draft
    except Exception as e:
        print(f"[-] Failed to create Gmail draft for {to_email}: {e}")
        return None

def send_gmail_email(service, to_email, subject, body_text, resume_path=None):
    """Sends an email from the user's Gmail with resume attached."""
    try:
        message = EmailMessage()
        message.set_content(body_text)
        message["To"] = to_email
        message["Subject"] = subject
        
        # Attach resume if path provided and exists
        if resume_path:
            if os.path.exists(resume_path):
                print(f"     [+] Attaching resume: {resume_path}")
                import mimetypes
                mime_type, _ = mimetypes.guess_type(resume_path)
                if mime_type is None:
                    mime_type = 'application/octet-stream'
                main_type, sub_type = mime_type.split('/', 1)
                
                with open(resume_path, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(resume_path)
                    message.add_attachment(file_data, maintype=main_type, subtype=sub_type, filename=file_name)
            else:
                print(f"     [-] Warning: Resume path '{resume_path}' does not exist! Sending email WITHOUT attachment.")
        
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"raw": encoded_message}
        
        sent_msg = service.users().messages().send(userId="me", body=create_message).execute()
        return sent_msg
    except Exception as e:
        print(f"[-] Failed to send email to {to_email}: {e}")
        return None

def send_summary_to_candidate(service, matched_companies, was_sent):
    """Sends a summary of today's outreach activities to the candidate's own email."""
    try:
        from datetime import datetime
        to_email = "hiteshkumarsingh6395@gmail.com"
        date_str = datetime.now().strftime('%Y-%m-%d')
        subject = f"Daily Outreach Summary - {date_str}"
        
        body_lines = [
            "Hello Hitesh,",
            "",
            f"The Daily Outreach Pipeline has completed its run for today ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}).",
            "",
            f"Mode: {'Direct Email Sending' if was_sent else 'Draft Staging in Inbox'}",
            f"Total companies processed today: {len(matched_companies)}",
            "",
        ]
        
        if matched_companies:
            body_lines.append("Details of matches:")
            body_lines.append("=" * 40)
            for comp, info in matched_companies.items():
                body_lines.append(f"- Company: {comp}")
                body_lines.append(f"  Role: {info['role']}")
                body_lines.append(f"  Location: {info.get('location', 'N/A')}")
                contacts = info.get('all_contacts', [{'name': 'Primary', 'email': info['contact_email']}])
                body_lines.append(f"  Contacts ({len(contacts)}):")
                for c in contacts:
                    body_lines.append(f"    → {c.get('name', 'Unknown')} ({c.get('title', 'N/A')}) — {c['email']}")
                body_lines.append(f"  Official Apply Link: {info.get('apply_link', 'N/A')}")
                body_lines.append(f"  Status: {'SENT' if was_sent else 'STAGED AS DRAFT'}")
                body_lines.append("")
        else:
            body_lines.append("No job postings matched your profile or had contact emails discovered today.")
            
        body_lines.append("\nBest regards,")
        body_lines.append("Your Automated Outreach Assistant Bot")
        
        body_text = "\n".join(body_lines)
        
        message = EmailMessage()
        message.set_content(body_text)
        message["To"] = to_email
        message["Subject"] = subject
        
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"raw": encoded_message}
        
        service.users().messages().send(userId="me", body=create_message).execute()
        print("[+] Status summary email successfully sent to your inbox!")
    except Exception as e:
        print(f"[-] Failed to send status summary email to candidate: {e}")

def _discover_contact_via_openrouter(company, company_url=None):
    """Fallback: uses OpenRouter free tier to discover company contact emails when Gemini quota is exhausted.
    NOTE: OpenRouter models don't have live web search, so this relies on the model's training data knowledge.
    CRITICAL: Instructed to NEVER fabricate emails — return null if not confidently known."""
    url_hint = f" (possible URL/domain hint: {company_url})" if company_url else ""
    prompt = f"""Find the official website domain and REAL, VERIFIED email contacts for the company "{company}"{url_hint}.

CRITICAL RULES — READ CAREFULLY:
1. You do NOT have internet access. Only return emails you are CONFIDENT exist from your training data.
2. NEVER fabricate, guess, or construct emails. Do NOT make up emails like careers@company.com, jobs@company.com, or firstname@company.com unless you have seen them in your training data.
3. If you are not 100% sure a specific email address exists and is active, set ALL email fields to null.
4. Preferred contacts (in order): CEO/Founder personal email, CTO email, HR Head/Manager email, official careers/recruiting email that you KNOW exists.
5. Do NOT return legal, privacy, security, abuse, billing, or customer support emails.

I would rather get null than a made-up email that will bounce.

Respond with a JSON object with these keys:
- "company_domain": string or null
- "contacts": array of objects, each with "name" (string or null), "title" (string or null), "email" (string or null), "source" (string describing where you know this email from, or null)
- "contact_email": string or null (the best single email from contacts, for backward compatibility)
- "email_source_url": string or null
- "email_found_on_web": boolean (set to false since you don't have web access)
- "reasoning": string
"""
    schema_props = {
        "company_domain": {"type": ["string", "null"]},
        "contacts": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": ["string", "null"]},
            "title": {"type": ["string", "null"]},
            "email": {"type": ["string", "null"]},
            "source": {"type": ["string", "null"]}
        }}},
        "contact_email": {"type": ["string", "null"]},
        "email_source_url": {"type": ["string", "null"]},
        "email_found_on_web": {"type": "boolean"},
        "reasoning": {"type": "string"}
    }
    try:
        result = openrouter_chat(prompt, json_schema_properties=schema_props, temperature=0.1)
        if isinstance(result, dict):
            # Validate any returned emails before accepting
            contacts = result.get("contacts", [])
            validated_contacts = []
            for c in contacts:
                email = c.get("email")
                if email and isinstance(email, str) and "@" in email:
                    is_valid, reason = validate_email_domain(email)
                    if is_valid:
                        validated_contacts.append(c)
                    else:
                        print(f"    [!] OpenRouter email rejected: {email} ({reason})")
            result["contacts"] = validated_contacts
            # Update contact_email to first valid contact
            if validated_contacts:
                result["contact_email"] = validated_contacts[0]["email"]
            else:
                result["contact_email"] = None
            return result
        return None
    except Exception as e:
        print(f"  [!] OpenRouter contact discovery also failed: {e}")
        return None

def _is_generic_guessed_email(email):
    """Detects emails that were likely guessed/fabricated rather than found on the web.
    Returns True if the email looks like a generic pattern guess."""
    if not email:
        return False
    email_lower = email.lower().strip()
    # Common generic prefixes that AI models fabricate
    generic_prefixes = [
        'careers@', 'jobs@', 'hr@', 'recruiting@', 'talent@', 'hiring@',
        'apply@', 'recruitment@', 'joinourteam@', 'work@', 'opportunities@',
        'info@', 'contact@', 'hello@', 'support@', 'help@'
    ]
    for prefix in generic_prefixes:
        if email_lower.startswith(prefix):
            return True
    return False

def discover_company_contact(client, company, company_url=None):
    """Uses a local cache file first, then gemini-2.5-flash with Google Search grounding to discover 
    REAL people (CEO, CTO, HR) and their verified email addresses at the company.
    Falls back to OpenRouter free tier if Gemini quota is exhausted.
    Returns a dict with 'contacts' array and backward-compatible 'contact_email' field."""
    global gemini_quota_exceeded
    company_clean = str(company).strip().lower()
    
    # Check local cache first
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'company_contacts_cache.json')
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            if company_clean in cache:
                cached = cache[company_clean]
                cached_email = cached.get("contact_email") if cached else None
                
                # Invalidate cache entries with generic guessed emails — these are what caused bounces
                if cached_email and _is_generic_guessed_email(cached_email):
                    print(f"  └─ Found '{company}' in cache but email '{cached_email}' looks generic/guessed. Re-searching...")
                    # Remove stale entry so we re-search
                    del cache[company_clean]
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache, f, indent=2)
                elif cached_email and str(cached_email).strip().lower() not in ["null", "none", ""]:
                    # Validate cached email domain
                    is_valid, reason = validate_email_domain(cached_email)
                    if is_valid:
                        print(f"  └─ Found '{company}' details in local contact cache (MX verified).")
                        return cached
                    else:
                        print(f"  └─ Cached email '{cached_email}' failed MX check ({reason}). Re-searching...")
                        del cache[company_clean]
                        with open(cache_path, 'w', encoding='utf-8') as f:
                            json.dump(cache, f, indent=2)
                else:
                    print(f"  └─ Found '{company}' in cache but no verified email. Skipping.")
                    return cached
        except Exception as e:
            print(f"  [!] Failed to read contact cache: {e}")
            
    if gemini_quota_exceeded:
        # Fallback to OpenRouter instead of giving up entirely
        print(f"  └─ Gemini quota exhausted. Falling back to OpenRouter for '{company}' contact discovery...")
        data = _discover_contact_via_openrouter(company, company_url)
        if data:
            # Save to cache
            try:
                cache = {}
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache = json.load(f)
                cache[company_clean] = data
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, indent=2)
                print(f"  [+] Saved '{company}' details to local contact cache (via OpenRouter).")
            except Exception as cache_err:
                print(f"  [!] Failed to save contact to cache: {cache_err}")
        return data
            
    print(f"  └─ Searching for '{company}' contact details via Google Search Grounding...")
    
    url_hint = f" (possible URL/domain hint: {company_url})" if company_url else ""
    prompt = f"""
Search the web to find REAL PEOPLE and their VERIFIED email addresses at the company "{company}"{url_hint}.

YOUR PRIMARY GOAL: Find the actual email addresses of specific people at this company — NOT generic emails.

SEARCH STRATEGY (follow this order):
1. Search LinkedIn for "{company}" + "CEO" OR "Founder" OR "CTO" OR "HR Manager" OR "Head of People" OR "Talent Acquisition". Look for their personal email in LinkedIn bios, or linked personal websites.
2. Search the company's official website team/about page for founder and leadership emails.
3. Search Wellfound/AngelList, Crunchbase, or startup directories for founder contact info.
4. Search GitHub for the company's org or founder's profile — many developers list their email publicly.
5. Search press releases, blog posts, or news articles that mention specific people and their emails.
6. As a LAST RESORT, check if the company has a known, working careers@ or hr@ email that is EXPLICITLY listed on their website (not guessed).

CRITICAL RULES:
- Only return emails you FOUND EXPLICITLY on a real webpage. Cite the source URL.
- NEVER construct or guess an email address. For example, do NOT take someone's name and combine it with the company domain to create "firstname@company.com" — unless you found that exact email on a webpage.
- If the company is a large enterprise (e.g., Microsoft, Google, Amazon, TikTok, Meta, Apple) that ONLY uses ATS portals for recruiting, return null for all emails. These companies don't accept cold emails.
- Prefer personal emails of decision-makers (CEO, CTO, Founder, HR Head) over generic departmental emails.
- Return up to 3 contacts if found.

Format your response exactly as a JSON object inside a single markdown code block:
```json
{{
  "company_domain": "string or null",
  "contacts": [
    {{
      "name": "Person's full name or null",
      "title": "Their title (CEO, CTO, HR Manager, etc.) or null",
      "email": "their verified email or null",
      "source": "URL where you found this email"
    }}
  ],
  "contact_email": "best single email from contacts array, or null",
  "email_source_url": "URL where the best email was found, or null",
  "email_found_on_web": true/false,
  "reasoning": "Explain how you found (or couldn't find) the emails"
}}
```
Do not output anything else.
"""
    try:
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
        response = gemini_generate_content(
            client=client,
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
            max_attempts=5
        )
        if not response:
            return None
            
        text = response.text
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        json_str = match.group(1) if match else text
        data = json.loads(json_str.strip())
        
        # Validate all returned emails via MX check
        contacts = data.get("contacts", [])
        validated_contacts = []
        for c in contacts:
            email = c.get("email")
            if email and isinstance(email, str) and "@" in email:
                # Skip generic guessed emails
                if _is_generic_guessed_email(email):
                    print(f"    [!] Skipping generic/guessed email: {email}")
                    continue
                is_valid, reason = validate_email_domain(email)
                if is_valid:
                    validated_contacts.append(c)
                    print(f"    [+] Validated: {c.get('name', 'Unknown')} ({c.get('title', 'Unknown')}) - {email} ✓")
                else:
                    print(f"    [!] Email domain invalid: {email} ({reason})")
        
        data["contacts"] = validated_contacts
        # Update backward-compatible contact_email
        if validated_contacts:
            data["contact_email"] = validated_contacts[0]["email"]
        else:
            # Check if original contact_email is valid and not generic
            orig_email = data.get("contact_email")
            if orig_email and not _is_generic_guessed_email(orig_email):
                is_valid, reason = validate_email_domain(orig_email)
                if not is_valid:
                    data["contact_email"] = None
            else:
                data["contact_email"] = None
        
        # Save newly discovered contact details to local cache
        try:
            cache = {}
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
            cache[company_clean] = data
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache, f, indent=2)
            print(f"  [+] Saved '{company}' details to local contact cache.")
        except Exception as cache_err:
            print(f"  [!] Failed to save contact to cache: {cache_err}")
            
        return data
    except GeminiQuotaExhaustedError:
        print("  [!] Gemini daily quota exhausted. Switching permanently to OpenRouter fallback...")
        gemini_quota_exceeded = True
        data = _discover_contact_via_openrouter(company, company_url)
        if data:
            try:
                cache = {}
                if os.path.exists(cache_path):
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache = json.load(f)
                cache[company_clean] = data
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, indent=2)
                print(f"  [+] Saved '{company}' details to local contact cache (via OpenRouter).")
            except Exception as cache_err:
                print(f"  [!] Failed to save contact to cache: {cache_err}")
        return data
    except Exception as e:
        print(f"  [!] Gemini email discovery failed: {e}")
        if "quota" in str(e).lower() or "429" in str(e) or "resource_exhausted" in str(e).lower():
            print("  [!] Gemini quota exhausted or rate-limit retry limit hit. Switching to OpenRouter fallback...")
            # Immediately try OpenRouter fallback for THIS company
            data = _discover_contact_via_openrouter(company, company_url)
            if data:
                try:
                    cache = {}
                    if os.path.exists(cache_path):
                        with open(cache_path, 'r', encoding='utf-8') as f:
                            cache = json.load(f)
                    cache[company_clean] = data
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache, f, indent=2)
                    print(f"  [+] Saved '{company}' details to local contact cache (via OpenRouter).")
                except Exception as cache_err:
                    print(f"  [!] Failed to save contact to cache: {cache_err}")
            return data
        return None


def generate_tailored_text(client, resume_text, company, role, job_desc_text, company_domain):
    """Uses Gemini 2.5 Flash to rewrite resume sections emphasizing skills relevant to the target company/role."""
    print(f"  └─ Generating tailored resume content for {company} via Gemini...")

    prompt = f"""You are a professional resume optimization expert. Given the candidate's resume and a target company/role, 
rewrite specific resume sections to emphasize the most relevant skills and achievements for this specific opportunity.

RULES:
- Only reorder and re-emphasize existing skills and achievements. Do NOT invent new skills or experiences.
- Keep the same core facts and metrics, just shift the emphasis and word choice.
- The skills line should reorder languages/frameworks to put the most relevant ones first.
- Experience bullets should highlight the aspects most aligned with the company's tech stack.
- Keep each bullet concise (1-2 sentences max).

Candidate Resume:
{resume_text}

Target Company: {company}
Target Role: {role}
Company Domain: {company_domain}
Job Description Context:
{job_desc_text[:2000] if job_desc_text else 'No job description available. Infer from company and role.'}

Respond with a JSON object containing the tailored skills, experience bullets, and project descriptions.
"""
    try:
        result = gemini_structured_chat(client, prompt, response_schema=TailoredResume, temperature=0.2)
        if isinstance(result, dict) and "skills_line" in result:
            # Convert dict to a simple namespace object so .attribute access works
            class TailoredResult:
                pass
            obj = TailoredResult()
            obj.skills_line = result.get("skills_line", "")
            obj.experience_bullets = result.get("experience_bullets", [])
            obj.project1_description = result.get("project1_description", [])
            obj.project2_description = result.get("project2_description", [])
            print(f"  [+] Tailored resume content generated successfully.")
            return obj
        else:
            print(f"  [-] Failed to generate tailored content: unexpected response format.")
            return None
    except Exception as e:
        print(f"  [-] Resume tailoring failed: {e}")
        return None


def tailor_resume_docx(original_docx_path, tailored_data, company_name):
    """Clones the original .docx and replaces resume text nodes with tailored content.
    Preserves all formatting, fonts, margins by only modifying text within existing XML runs."""
    print(f"  └─ Building tailored .docx for {company_name}...")

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'tailored_resumes')
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize company name for filename
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', company_name.strip())
    output_path = os.path.join(output_dir, f'Resume_Hitesh_{safe_name}.docx')

    try:
        # Read original docx
        with zipfile.ZipFile(original_docx_path, 'r') as zin:
            # Read all file contents
            file_contents = {}
            for name in zin.namelist():
                file_contents[name] = zin.read(name)

        # Parse and modify document.xml
        ns = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        ET.register_namespace('w', ns)
        # Register all common OOXML namespaces to avoid ns0/ns1 prefix issues
        ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
        ET.register_namespace('wp', 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing')
        ET.register_namespace('a', 'http://schemas.openxmlformats.org/drawingml/2006/main')
        ET.register_namespace('mc', 'http://schemas.openxmlformats.org/markup-compatibility/2006')
        ET.register_namespace('wps', 'http://schemas.microsoft.com/office/word/2010/wordprocessingShape')
        ET.register_namespace('w14', 'http://schemas.microsoft.com/office/word/2010/wordml')
        ET.register_namespace('w15', 'http://schemas.microsoft.com/office/word/2012/wordml')
        ET.register_namespace('wpc', 'http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas')
        ET.register_namespace('v', 'urn:schemas-microsoft-com:vml')
        ET.register_namespace('o', 'urn:schemas-microsoft-com:office:office')

        root = ET.fromstring(file_contents['word/document.xml'])

        # Helper: get full text of a paragraph element
        def get_paragraph_text(p_elem):
            texts = [t.text for t in p_elem.iter(f'{{{ns}}}t') if t.text]
            return ''.join(texts)

        # Helper: replace text across runs in a paragraph while preserving formatting
        def replace_paragraph_text(p_elem, new_text):
            """Replace the text content of a paragraph while keeping run formatting.
            Strategy: put all new text in the first <w:t> and clear the rest."""
            t_elements = list(p_elem.iter(f'{{{ns}}}t'))
            if not t_elements:
                return
            # Set full new text on first <w:t>, clear others
            t_elements[0].text = new_text
            t_elements[0].set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            for t in t_elements[1:]:
                t.text = ''

        # Build replacement map: original_text_substring -> new_text
        # We'll match paragraphs by checking if they contain key phrases
        all_paragraphs = list(root.iter(f'{{{ns}}}p'))

        # 1. Replace Skills line (Languages line)
        for p in all_paragraphs:
            text = get_paragraph_text(p)
            if 'TypeScript, Python, JavaScript, Java, C, C++' in text:
                skills_val = tailored_data.skills_line or ""
                # Strip leading 'Languages:' prefix case-insensitively if present
                if skills_val.lower().startswith("languages:"):
                    skills_val = skills_val[len("languages:"):].strip()
                new_text = text.replace('TypeScript, Python, JavaScript, Java, C, C++', skills_val)
                replace_paragraph_text(p, new_text)
                break

        # 2. Replace experience bullets (Founder & Technical Lead section)
        original_experience_markers = [
            ('Architecting and developing a PropTech MVP', 0),
            ('Engineered a secure multi-tenant backend', 1),
            ('Integrated AI-powered geofencing', 2),
            ('Managing full-stack domain deployment', 3)
        ]
        for p in all_paragraphs:
            text = get_paragraph_text(p)
            for marker, idx in original_experience_markers:
                if marker in text and idx < len(tailored_data.experience_bullets):
                    replace_paragraph_text(p, tailored_data.experience_bullets[idx])
                    break

        # 3. Replace SentinelLog-AI project bullets
        sentinel_markers = [
            ('Engineered a headless data pipeline', 0),
            ('Programmed rule-based security filters', 1)
        ]
        for p in all_paragraphs:
            text = get_paragraph_text(p)
            for marker, idx in sentinel_markers:
                if marker in text and idx < len(tailored_data.project1_description):
                    replace_paragraph_text(p, tailored_data.project1_description[idx])
                    break

        # 4. Replace RAG Engine project bullets
        rag_markers = [
            ('Developed an asynchronous Retrieval-Augmented Generation', 0),
            ('Configured low-latency semantic similarity', 1)
        ]
        for p in all_paragraphs:
            text = get_paragraph_text(p)
            for marker, idx in rag_markers:
                if marker in text and idx < len(tailored_data.project2_description):
                    replace_paragraph_text(p, tailored_data.project2_description[idx])
                    break

        # Write modified XML back
        modified_xml = ET.tostring(root, encoding='unicode', xml_declaration=False)
        # Prepend XML declaration
        modified_xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + modified_xml
        file_contents['word/document.xml'] = modified_xml.encode('utf-8')

        # Write new docx
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for name, data in file_contents.items():
                zout.writestr(name, data)

        print(f"  [+] Tailored resume saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"  [-] Failed to create tailored .docx: {e}")
        print(f"  [*] Falling back to original resume.")
        return original_docx_path


def main():
    parser = argparse.ArgumentParser(description="Automated Internship Discovery and Cold Email Outreach Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run the full pipeline, matches, and drafts outputs locally WITHOUT staging to Gmail.")
    parser.add_argument("--send", action="store_true", help="Directly send outreach emails instead of staging drafts.")
    parser.add_argument("--resume", type=str, default="re.docx", help="Path to resume docx file.")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode (fails if Gmail authentication requires browser access).")
    args = parser.parse_args()

    # Resolve resume path relative to the script's directory if it is a relative path
    resume_path = args.resume
    if not os.path.isabs(resume_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        resume_path = os.path.abspath(os.path.join(script_dir, resume_path))

    # 1. Parse resume
    resume_text = parse_docx_resume(resume_path)

    # Clean up old tailored resumes from previous runs
    tailored_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tailored_resumes')
    if os.path.exists(tailored_dir):
        shutil.rmtree(tailored_dir)
        print("[*] Cleaned up old tailored resumes.")
    os.makedirs(tailored_dir, exist_ok=True)

    # Try to load GEMINI_API_KEY from .env if it exists and is not already in environment
    if not os.environ.get("GEMINI_API_KEY"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip() and not line.startswith('#') and '=' in line:
                        key, val = line.strip().split('=', 1)
                        if key.strip() == 'GEMINI_API_KEY':
                            os.environ['GEMINI_API_KEY'] = val.strip().strip('"').strip("'")

    # Check for Gemini API key
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("[-] Error: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    # Initialize Google GenAI client
    client = genai.Client()

    # 1b. Pre-flight: verify Gmail token BEFORE burning API calls
    if not args.dry_run:
        print("\n[*] Pre-flight check: verifying Gmail token...")
        check_gmail_token_valid(args.non_interactive)

    # 2. Fetch listings
    simplify_postings = fetch_speedyapply_postings()
    india_searched_postings = search_india_internships(client)
    print(f"[+] Found {len(india_searched_postings)} India listings via Gemini search grounding.")
    
    postings = simplify_postings + india_searched_postings

    # Define simple local filtering keywords
    keywords = ["ai", "ml", "software", "swe", "backend", "full stack", "fullstack", "frontend", "developer", "engineer", "data science", "machine learning", "infrastructure"]
    filtered_postings = []
    for p in postings:
        role_lower = p['role'].lower()
        if any(kw in role_lower for kw in keywords):
            filtered_postings.append(p)

    print(f"[*] Filtered down to {len(filtered_postings)} roles using local title keywords.")

    # Classify candidates as India vs Abroad
    india_keywords = ["india", "bangalore", "bengaluru", "mumbai", "delhi", "gurugram", "gurgaon", "noida", "pune", "hyderabad", "chennai", "kolkata", "ncr"]
    bangalore_cities = ["bangalore", "bengaluru"]
    other_priority_cities = ["mumbai", "delhi", "gurugram", "gurgaon", "noida", "ncr"]
    
    india_candidates = []
    abroad_candidates = []
    
    for p in filtered_postings:
        loc_lower = p['location'].lower()
        if any(kw in loc_lower for kw in india_keywords):
            if any(city in loc_lower for city in bangalore_cities):
                priority_score = 2
            elif any(city in loc_lower for city in other_priority_cities):
                priority_score = 1
            else:
                priority_score = 0
            india_candidates.append((priority_score, p))
        else:
            abroad_candidates.append(p)
            
    # Sort India candidates by priority score descending (Bangalore first, then other target cities, then others)
    india_candidates.sort(key=lambda x: x[0], reverse=True)
    india_postings = [x[1] for x in india_candidates]
    abroad_postings = abroad_candidates

    print(f"[+] India candidates found: {len(india_postings)} (prioritizing {len([x for x in india_candidates if x[0]])} in target cities)")
    print(f"[+] Abroad candidates found: {len(abroad_postings)}")

    matched_companies = {}  # Combined final matches: company_name -> match_info
    
    # Target: Send at least 50 emails total. We prioritize India postings.
    total_target = 50
    
    # helper evaluation run
    def evaluate_postings_list(postings_list, target_count, target_label, is_india):
        global gemini_quota_exceeded
        print(f"\n[*] Commencing Dynamic Evaluation for {target_label} Candidates (Target: {target_count})...")
        for idx, p in enumerate(postings_list):
            current_matched = len([c for c, m in matched_companies.items() if m['is_india'] == is_india])
            if current_matched >= target_count:
                break
                
            company = p['company']
            if company in matched_companies:
                continue
                
            if gemini_quota_exceeded and not os.environ.get("OPENROUTER_API_KEY"):
                print(f"  [-] Skipping evaluation for {company}: Both Gemini and OpenRouter are unavailable.")
                continue
                
            role = p['role']
            apply_link = p['apply_link']
            location = p['location']
            
            print(f"\n[{target_label} Match {current_matched+1}/{target_count}] Evaluating {company} - '{role}' ({location})...")
            
            # Step 1: Contact & Domain Discovery
            # Optimize: If we already have a verified founder email in the posting, use it directly to save API quota!
            contact_info = None
            if p.get('founder_email') and isinstance(p['founder_email'], str) and "@" in p['founder_email']:
                founder_email = p['founder_email'].strip()
                if founder_email.lower() not in ["null", "none", ""]:
                    is_valid, reason = validate_email_domain(founder_email)
                    if is_valid and not _is_generic_guessed_email(founder_email):
                        print(f"  [✓] Using pre-discovered contact email from search phase: {founder_email}")
                        contact_info = {
                            'company_domain': p.get('company_url') or founder_email.split('@')[1],
                            'contacts': [{
                                'name': p.get('founder_name') or 'Founder',
                                'title': p.get('founder_title') or 'CEO/Founder',
                                'email': founder_email,
                                'source': 'Search Phase'
                            }],
                            'contact_email': founder_email,
                            'email_found_on_web': True,
                            'reasoning': 'Discovered during search grounding phase'
                        }
            
            if not contact_info:
                contact_info = discover_company_contact(client, company, p.get('company_url'))
            
            # Collect all verified emails (multi-contact support)
            all_contacts = []
            
            # Check contacts array first (new format)
            if contact_info and contact_info.get("contacts"):
                for c in contact_info["contacts"]:
                    email = c.get("email")
                    if email and isinstance(email, str) and "@" in email:
                        email = email.strip()
                        if email.lower() not in ["null", "none", ""]:
                            all_contacts.append({
                                'name': c.get('name', 'Hiring Team'),
                                'title': c.get('title', ''),
                                'email': email
                            })
            
            # Also check backward-compatible contact_email
            if contact_info and contact_info.get("contact_email"):
                email_candidate = str(contact_info.get("contact_email")).strip()
                if email_candidate.lower() not in ["null", "none", ""] and "@" in email_candidate:
                    # Don't duplicate
                    existing_emails = {c['email'].lower() for c in all_contacts}
                    if email_candidate.lower() not in existing_emails:
                        all_contacts.append({
                            'name': 'Hiring Team',
                            'title': '',
                            'email': email_candidate
                        })
            
            # Check if we got founder info from the search phase
            if p.get('founder_email') and isinstance(p['founder_email'], str) and "@" in p['founder_email']:
                founder_email = p['founder_email'].strip()
                if founder_email.lower() not in ["null", "none", ""]:
                    existing_emails = {c['email'].lower() for c in all_contacts}
                    if founder_email.lower() not in existing_emails:
                        # Validate founder email from search
                        is_valid, reason = validate_email_domain(founder_email)
                        if is_valid and not _is_generic_guessed_email(founder_email):
                            all_contacts.insert(0, {  # Insert at front — founder has priority
                                'name': p.get('founder_name', 'Founder'),
                                'title': p.get('founder_title', 'CEO/Founder'),
                                'email': founder_email
                            })
            
            if not all_contacts:
                print(f"  [-] Skipping: No verified contact email found for '{company}' (Reasoning: {contact_info.get('reasoning') if contact_info else 'Search failed'})")
                continue
            
            # Show discovered contacts
            for c in all_contacts:
                print(f"  [✓] Contact: {c['name']} ({c['title']}) — {c['email']}")
                
            # Fetch job description text
            job_desc_text = ""
            if apply_link:
                print(f"  └─ Fetching job details from: {apply_link[:60]}...")
                job_desc_text = fetch_job_description(apply_link)
                if job_desc_text:
                    print(f"  └─ Successfully fetched {len(job_desc_text)} chars.")
                else:
                    print("  └─ Direct job page scraping blocked; using company metadata fallback.")
            
            # Use the primary (first) contact for email personalization
            primary_contact = all_contacts[0]
            verified_email = primary_contact['email']
            contact_name = primary_contact['name'] if primary_contact['name'] != 'Hiring Team' else None
                    
            # Build prompt for Step 2 (Stack Matching & Personalization)
            greeting_name = contact_name if contact_name else "Hiring Team"
            prompt = f"""
You are an Advanced Lead Generation & Outreach Engineer. Analyze the company, job role, and candidate details to evaluate matching and draft an outreach email.

Candidate Information:
- Name: Hitesh Kumar Singh
- Email: hiteshkumarsingh6395@gmail.com
- GitHub: https://github.com/Hitesh-singh67
- LinkedIn: https://www.linkedin.com/in/hitesh-singh67/
- Programming Languages: TypeScript, Python, JavaScript, Java, C, C++
- Resume Details:
{resume_text}

Discovered Job Posting:
- Company: {company}
- Role: {role}
- Location: {location}
- Application Link: {apply_link}
- Scraped Job Description text:
{job_desc_text}

Contact Details (Verified from Google Search):
- Official Domain: {contact_info.get('company_domain')}
- Verified Contact Email: {verified_email}
- Contact Person Name: {greeting_name}

Your Tasks:
1. Determine if this job role or the company matches the candidate's core programming languages (TypeScript, Python, JavaScript, Java, C, C++).
2. For the company_domain and contact_email fields in your response, use the values provided above: Domain = "{contact_info.get('company_domain')}", Email = "{verified_email}".
3. Draft a personalized cold email following this exact structure and style, but customize the details to align with the company ({company}) and the job description:

Structure of the Email:
- **Greeting**: "Greetings {greeting_name}," (use the contact person's name provided above).
- **Paragraph 1 (Intro)**: "I am Hitesh Kumar Singh, a final-year undergraduate pursuing a Bachelor of Computer Applications (BCA) from the University of Mysore. I am seeking an Internship opportunity with the esteemed {company} in any domain the team finds me a fit (with a strong preference for backend development, AI/ML, or data engineering roles)." (Modify the preference slightly to match the role if it's explicitly backend, AI/ML, frontend, full-stack, etc.).
- **Paragraph 2 (Primary Experience)**: Describe your experience as Founder & Technical Lead at PropelAI Technologies. Customize this paragraph to emphasize the technologies, databases, frameworks, or security/systems aspects that are most relevant to the target company's stack and business focus. Keep the details factual to your resume (PropTech SaaS MVP, Node.js, Python, PostgreSQL, RLS, geofencing, digital ledger).
- **Paragraph 3 (Projects & PORs)**: Describe your leadership role managing full-stack deployments and GTM strategy, and mention key projects (such as the SentinelLog-AI network security pipeline or the pgvector RAG engine). Highlight the project that is most relevant to the target company's job description.
- **Paragraph 4 (Eagerness & Education)**: "Also, I acknowledge that I am not from a Tier-1 college, but am highly eager to learn the engineering workings of {company} which would align with my career goals further. I know you'll be able to connect with me on this; I have been actively trying my best to push my boundaries by building complex, production-grade systems, participating in hands-on industry simulations (such as Tata's Data Analytics and AWS Solutions Architecture), and contributing to open-source tools. I am sure you would find my GitHub and LinkedIn worth a look!"
- **Paragraph 5 (Outro)**: "I would love to discuss more about how I can contribute to {company} and its engineering/product departments. I have shared my resume via the Google Drive link below for your kind perusal and consideration. I look forward to hearing from you and providing my time and skills to {company} soon!"
- **Links and Sign-off**:
  Resume - https://drive.google.com/file/d/1gYimVJcu0v2wsPVWNpFgGvPfLxUBDhE7/view?usp=drive_link
  GitHub - https://github.com/Hitesh-singh67
  LinkedIn - https://www.linkedin.com/in/hitesh-singh67/

  Thanking you in anticipation,

  Hitesh Kumar Singh
  +91 6398595165

Ensure the drafted email strictly adheres to this structure and matches the tone. Keep the body text under 350 words.
"""

            try:
                if gemini_quota_exceeded:
                    raise GeminiQuotaExhaustedError("Gemini quota previously exhausted")
                
                print("  └─ Sending prompt to Gemini for stack evaluation and drafting...")
                res = gemini_structured_chat(client, prompt, response_schema=MatchResult, temperature=0.2)
                            
                if isinstance(res, dict) and res.get("matches_stack"):
                    # Store all contacts for multi-send support
                    matched_companies[company] = {
                        'role': role,
                        'apply_link': apply_link,
                        'location': location,
                        'domain': contact_info.get('company_domain', ''),
                        'contact_email': verified_email,  # Primary contact — always use the verified one
                        'all_contacts': all_contacts,  # All contacts for multi-send
                        'subject': res.get('email_subject', ''),
                        'body': res.get('email_body', ''),
                        'is_india': is_india,
                        'tailored_resume': None
                    }
                    contact_names = ", ".join([f"{c['name']} ({c['email']})" for c in all_contacts])
                    print(f"  [+] Match Approved! Contacts: {contact_names}")
                else:
                    reason = res.get('reason', 'Unknown') if isinstance(res, dict) else 'API Error'
                    print(f"  [-] Match Rejected: {reason}")
            except GeminiQuotaExhaustedError:
                print("  [!] Gemini daily quota exhausted during evaluation. Switching permanently to OpenRouter...")
                gemini_quota_exceeded = True
                try:
                    print("  └─ Sending prompt to OpenRouter fallback for stack evaluation...")
                    schema_props = {
                        "matches_stack": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "company_domain": {"type": "string"},
                        "contact_email": {"type": "string"},
                        "email_subject": {"type": "string"},
                        "email_body": {"type": "string"}
                    }
                    res = openrouter_chat(prompt, json_schema_properties=schema_props, temperature=0.2)
                    if isinstance(res, dict) and res.get("matches_stack"):
                        matched_companies[company] = {
                            'role': role,
                            'apply_link': apply_link,
                            'location': location,
                            'domain': contact_info.get('company_domain', ''),
                            'contact_email': verified_email,
                            'all_contacts': all_contacts,
                            'subject': res.get('email_subject', ''),
                            'body': res.get('email_body', ''),
                            'is_india': is_india,
                            'tailored_resume': None
                        }
                        print(f"  [+] Match Approved via OpenRouter! Contacts: {verified_email}")
                    else:
                        reason = res.get('reason', 'Unknown') if isinstance(res, dict) else 'API Error'
                        print(f"  [-] Match Rejected via OpenRouter: {reason}")
                except Exception as or_err:
                    print(f"  [-] OpenRouter fallback also failed: {or_err}")
            except Exception as e:
                print(f"  [-] Gemini evaluation failed: {e}")

    # Evaluate India list (up to the total target limit)
    evaluate_postings_list(india_postings, total_target, "India", is_india=True)
    
    # Evaluate Abroad list (fill the remaining target space to hit at least 50 emails total)
    remaining_target = max(0, total_target - len(matched_companies))
    if remaining_target > 0:
        evaluate_postings_list(abroad_postings, remaining_target, "Abroad", is_india=False)

    print(f"\n[+] Staging Complete! Discovered {len(matched_companies)} matching companies.")

    # 3. Inbox Staging or Direct Sending
    if args.dry_run:
        if not matched_companies:
            print("[-] No companies matched the requirements. Exiting.")
            return
        print("\n" + "="*80)
        print("DRY RUN MODE: Outputting outreach drafts to terminal.")
        print("="*80)
        total_emails = 0
        for comp, info in matched_companies.items():
            contacts = info.get('all_contacts', [{'name': 'Primary', 'email': info['contact_email']}])
            print(f"\nCompany: {comp} (Domain: {info['domain']}) [India: {info['is_india']}]")
            print(f"Role: {info['role']}")
            print(f"Contacts ({len(contacts)}):")
            for c in contacts:
                print(f"  → {c.get('name', 'Unknown')} ({c.get('title', 'N/A')}) — {c['email']}")
            print(f"Subject: {info['subject']}")
            print(f"Body:\n{info['body']}")
            print("-" * 50)
            total_emails += len(contacts)
        print(f"\n[+] Dry run finished successfully. {len(matched_companies)} companies, {total_emails} total emails would be sent.")
    else:
        print("\n[*] Connecting securely to Gmail API...")
        gmail_service = get_gmail_service(args.non_interactive)
        print("[+] Gmail connection established!")
        
        if not matched_companies:
            print("[-] No companies matched the requirements.")
            send_summary_to_candidate(gmail_service, {}, args.send)
            return

        sent_count = 0
        skipped_count = 0
        
        if args.send:
            print("\n[*] Sending outreach emails directly via Gmail...")
            for comp, info in matched_companies.items():
                contacts = info.get('all_contacts', [{'name': 'Primary', 'email': info['contact_email']}])
                for contact in contacts:
                    email_addr = contact['email']
                    contact_name = contact.get('name', 'Unknown')
                    
                    # Final MX validation before sending
                    is_valid, reason = validate_email_domain(email_addr)
                    if not is_valid:
                        print(f"  [!] SKIPPING {email_addr} — MX validation failed: {reason}")
                        skipped_count += 1
                        continue
                    
                    print(f"  └─ Sending to {comp} → {contact_name} ({email_addr})...")
                    sent_msg = send_gmail_email(gmail_service, email_addr, info['subject'], info['body'], resume_path=None)
                    if sent_msg:
                        print(f"     [+] Successfully sent (ID: {sent_msg['id']})")
                        sent_count += 1
            print(f"\n[+] Email sending complete! Sent: {sent_count}, Skipped (bad domain): {skipped_count}")
        else:
            print("\n[*] Creating pending drafts in Gmail...")
            for comp, info in matched_companies.items():
                contacts = info.get('all_contacts', [{'name': 'Primary', 'email': info['contact_email']}])
                for contact in contacts:
                    email_addr = contact['email']
                    contact_name = contact.get('name', 'Unknown')
                    
                    # Final MX validation before staging
                    is_valid, reason = validate_email_domain(email_addr)
                    if not is_valid:
                        print(f"  [!] SKIPPING {email_addr} — MX validation failed: {reason}")
                        skipped_count += 1
                        continue
                    
                    print(f"  └─ Creating draft for {comp} → {contact_name} ({email_addr})...")
                    draft = stage_gmail_draft(gmail_service, email_addr, info['subject'], info['body'], resume_path=None)
                    if draft:
                        print(f"     [+] Successfully staged draft (ID: {draft['id']})")
                        sent_count += 1
            print(f"\n[+] Draft staging complete! Staged: {sent_count}, Skipped (bad domain): {skipped_count}")
        
        send_summary_to_candidate(gmail_service, matched_companies, args.send)

if __name__ == '__main__':
    main()
