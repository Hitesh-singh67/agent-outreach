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
    Uses the 'openrouter/auto' router which auto-selects the best free model."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")

    messages = [{"role": "user", "content": prompt}]
    body = {
        "model": "openrouter/auto",
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
    email_body: str = Field(description="A personalized, fluff-free cold email (<130 words) from the candidate to the career contact aligning candidate's background with the company's product stack.")

# Custom HTML parser to extract tables from SimplifyJobs README.md
class SimplifyParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_tbody = False
        self.in_tr = False
        self.in_td = False
        self.current_row = []
        self.current_cell_data = []
        self.current_hrefs = []
        self.rows = []
        self.current_section = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'table':
            self.in_table = True
        elif tag == 'tbody' and self.in_table:
            self.in_tbody = True
        elif tag == 'tr' and self.in_tbody:
            self.in_tr = True
            self.current_row = []
        elif tag == 'td' and self.in_tr:
            self.in_td = True
            self.current_cell_data = []
            self.current_hrefs = []
        elif tag == 'a' and self.in_td:
            if 'href' in attrs_dict:
                self.current_hrefs.append(attrs_dict['href'])
        elif tag == 'br' and self.in_td:
            self.current_cell_data.append('\n')

    def handle_endtag(self, tag):
        if tag == 'table':
            self.in_table = False
            self.in_tbody = False
        elif tag == 'tbody':
            self.in_tbody = False
        elif tag == 'tr' and self.in_tbody:
            self.in_tr = False
            if self.current_row:
                self.rows.append((self.current_section, self.current_row))
        elif tag == 'td' and self.in_tr:
            self.in_td = False
            cell_text = "".join(self.current_cell_data).strip()
            self.current_row.append({
                'text': cell_text,
                'hrefs': self.current_hrefs
            })

    def handle_data(self, data):
        if self.in_td:
            self.current_cell_data.append(data)

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

def fetch_simplify_postings():
    """Scrapes SimplifyJobs Summer 2026 tech internships and parses active listings."""
    print("[*] Fetching active tech internships from SimplifyJobs...")
    url = "https://raw.githubusercontent.com/SimplifyJobs/Summer2026-Internships/dev/README.md"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
    except Exception as e:
        print(f"[-] Failed to download SimplifyJobs README.md: {e}")
        return []

    sections = re.split(r'##\s+', content)
    parser = SimplifyParser()
    all_postings = []

    for sec in sections:
        lines = sec.split('\n')
        if not lines:
            continue
        header = lines[0].strip()

        category = None
        if "Software Engineering Internship Roles" in header:
            category = "SWE"
        elif "Data Science, AI & Machine Learning Internship Roles" in header:
            category = "AI/ML"

        if not category:
            continue

        table_match = re.search(r'<table>.*?</table>', sec, re.DOTALL | re.IGNORECASE)
        if not table_match:
            continue

        table_html = table_match.group(0)
        parser.current_section = category
        parser.rows = []
        parser.feed(table_html)

        current_company = None
        current_company_url = None

        for cat, row in parser.rows:
            if len(row) < 4:
                continue

            company_cell = row[0]
            role_cell = row[1]
            location_cell = row[2]
            app_cell = row[3]

            company_text = company_cell['text'].replace('🔥', '').strip()
            if company_text == '↳' or not company_text:
                pass
            else:
                current_company = company_text
                current_company_url = company_cell['hrefs'][0] if company_cell['hrefs'] else None

            role_text = role_cell['text'].strip()
            location_text = location_cell['text'].strip()
            apply_link = app_cell['hrefs'][0] if app_cell['hrefs'] else None

            # Skip closed postings
            if "🚫" in company_text or "🚫" in role_text or "closed" in role_text.lower():
                continue

            all_postings.append({
                'category': cat,
                'company': current_company,
                'company_url': current_company_url,
                'role': role_text,
                'location': location_text,
                'apply_link': apply_link
            })

    print(f"[+] SimplifyJobs active postings scraped: {len(all_postings)}")
    return all_postings

def search_india_internships(client):
    """Searches Google for active Software Engineering and AI/ML internship postings in India using Gemini Search Grounding."""
    print("[*] Searching Google for active India internship postings (Bangalore/Mumbai/Delhi NCR) using Gemini Search Grounding...")
    
    prompt = """
Find 12 active Software Engineer Intern, AI Intern, or Machine Learning Intern postings in India (specifically target locations like Bangalore/Bengaluru, Mumbai, Gurugram, Gurgaon, Noida, Pune, Hyderabad).
The postings should be on Greenhouse (boards.greenhouse.io), Lever (jobs.lever.co), Ashby (jobs.ashbyhq.com), or official company career portals.
Only return active internship postings from the last 2-3 months. Do NOT include closed roles.

Format your response exactly as a JSON array of objects inside a single markdown code block, like this:
```json
[
  {
    "company": "Company Name",
    "company_url": "https://company.com",
    "role": "Role Title",
    "location": "Location (e.g. Bangalore, India)",
    "apply_link": "https://boards.greenhouse.io/... or direct apply link"
  },
  ...
]
```
Do not output anything else.
"""
    try:
        import time
        backoff = 3
        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    )
                )
                break
            except Exception as e:
                if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                    print(f"  [!] API busy or rate-limited. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise e
                    
        if not response:
            raise Exception("No response received from GenAI API.")
            
        text = response.text
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        json_str = match.group(1) if match else text
        postings_data = json.loads(json_str.strip())
        
    except Exception as e:
        print(f"[-] Gemini search grounding for India jobs failed: {e}")
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
        postings.append({
            'category': 'AI/ML' if any(x in item.get('role', '').lower() for x in ['ai', 'ml', 'machine', 'learning']) else 'SWE',
            'company': item.get('company', 'Unknown'),
            'company_url': item.get('company_url', ''),
            'role': item.get('role', 'Intern'),
            'location': item.get('location', 'India'),
            'apply_link': item.get('apply_link', '')
        })
    print(f"[+] Found {len(postings)} India listings.")
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
                body_lines.append(f"  Contact Email: {info['contact_email']}")
                body_lines.append(f"  Resume Type: {'Tailored' if info.get('tailored_resume') and 'tailored' in str(info['tailored_resume']) else 'Original'}")
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
    NOTE: OpenRouter models don't have live web search, so this relies on the model's training data knowledge."""
    url_hint = f" (possible URL/domain hint: {company_url})" if company_url else ""
    prompt = f"""Find the official website domain and the actual careers, recruiting, or talent contact email for the company "{company}"{url_hint}.

Look for:
1. Careers page or contact page on their website.
2. Known contact emails like jobs@company.com, careers@company.com, recruiting@company.com, hr@company.com, talent@company.com.
3. Do NOT guess or fabricate an email. If the company only uses application portals (Greenhouse, Lever, Workday) and you don't know a public careers email, set "contact_email" to null.
4. Do NOT return legal, privacy, security, abuse, billing, developer support, or customer support emails. If only these types of emails are known, set "contact_email" to null.

Respond with a JSON object with these keys:
- "company_domain": string or null
- "contact_email": string or null
- "email_source_url": string or null
- "email_found_on_web": boolean
- "reasoning": string
"""
    schema_props = {
        "company_domain": {"type": ["string", "null"]},
        "contact_email": {"type": ["string", "null"]},
        "email_source_url": {"type": ["string", "null"]},
        "email_found_on_web": {"type": "boolean"},
        "reasoning": {"type": "string"}
    }
    try:
        result = openrouter_chat(prompt, json_schema_properties=schema_props, temperature=0.1)
        if isinstance(result, dict):
            return result
        return None
    except Exception as e:
        print(f"  [!] OpenRouter contact discovery also failed: {e}")
        return None


def discover_company_contact(client, company, company_url=None):
    """Uses a local cache file first, then gemini-2.5-flash with Google Search grounding to discover verified contact emails.
    Falls back to OpenRouter free tier if Gemini quota is exhausted."""
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
                # Smart skip: if cached entry has no email, return immediately without burning an API call
                cached_email = cached.get("contact_email") if cached else None
                if cached_email and str(cached_email).strip().lower() not in ["null", "none", ""]:
                    print(f"  └─ Found '{company}' details in local contact cache.")
                    return cached
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
Find the official website domain and the actual careers, recruiting, or talent contact email for the company "{company}"{url_hint}.
Use Google Search to find this information. Look for:
1. Careers page or contact page on their website.
2. Job postings or press releases that list a contact email (e.g. jobs@company.com, careers@company.com, recruiting@company.com, hr@company.com, talent@company.com).
3. Do NOT guess or guess-construct an email (like "careers@company.com") if it is not explicitly mentioned on the web. If they only use application portals (like Greenhouse, Lever, Workday) and do not have a public careers email, answer that no email was found.
4. Do NOT return legal, privacy, security, abuse, fraud-reporting, billing, developer support, or customer support/service emails (e.g., privacy@company.com, dpo@company.com, security@company.com, billing@company.com, support@company.com, abuse@company.com, reporthiringfraud@company.com). If only these types of emails are found on the web, treat it as if no verified careers email was found and set "contact_email" to null.

Format your response exactly as a JSON object inside a single markdown code block, like this:
```json
{{
  "company_domain": "string or null",
  "contact_email": "string or null",
  "email_source_url": "string or null",
  "email_found_on_web": boolean,
  "reasoning": "string"
}}
```
Do not output anything else.
"""
    try:
        import time
        # Respect API rate limits
        time.sleep(2)
        
        # Add retry/backoff logic to handle 503 and rate limit errors
        backoff = 3
        response = None
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    )
                )
                break
            except Exception as e:
                if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                    print(f"  [!] API busy or rate-limited. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise e

        if not response:
            return None
            
        text = response.text
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        json_str = match.group(1) if match else text
        data = json.loads(json_str.strip())
        
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
    except Exception as e:
        print(f"  [!] Gemini email discovery failed: {e}")
        if "quota" in str(e).lower() or "429" in str(e) or "resource_exhausted" in str(e).lower():
            print("  [!] Gemini quota exhausted. Switching to OpenRouter fallback for contact discovery...")
            gemini_quota_exceeded = True
            # Immediately try OpenRouter fallback for THIS company instead of returning None
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
    """Uses OpenRouter free tier to rewrite resume sections emphasizing skills relevant to the target company/role."""
    print(f"  └─ Generating tailored resume content for {company} via OpenRouter...")

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

Respond with a JSON object with these exact keys:
- "skills_line": The candidate's skills line reordered to put the most relevant languages/frameworks first for this company. Only include skills the candidate actually has.
- "experience_bullets": An array of exactly 4 strings. The 4 experience bullets from the Founder & Technical Lead section, rewritten to emphasize aspects most relevant to this company's stack. Keep the same achievements, just shift emphasis.
- "project1_description": An array of exactly 2 strings. The 2 bullet points for SentinelLog-AI project, rewritten to emphasize relevance to this company. Keep same achievements.
- "project2_description": An array of exactly 2 strings. The 2 bullet points for RAG Engine project, rewritten to emphasize relevance to this company. Keep same achievements.
"""

    schema_props = {
        "skills_line": {"type": "string"},
        "experience_bullets": {"type": "array", "items": {"type": "string"}},
        "project1_description": {"type": "array", "items": {"type": "string"}},
        "project2_description": {"type": "array", "items": {"type": "string"}}
    }

    try:
        result = openrouter_chat(prompt, json_schema_properties=schema_props, temperature=0.2)
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
    simplify_postings = fetch_simplify_postings()
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
    
    # Target: 2 India matches, 1 Abroad matches (optimized to stay within Gemini Free Tier limits of 20 requests/day)
    target_india = 2
    target_abroad = 1
    
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
            
            # Step 1: Contact & Domain Discovery using Search Grounding
            contact_info = discover_company_contact(client, company, p.get('company_url'))
            
            verified_email = None
            if contact_info and contact_info.get("contact_email"):
                email_candidate = str(contact_info.get("contact_email")).strip()
                if email_candidate.lower() not in ["null", "none", ""] and "@" in email_candidate:
                    verified_email = email_candidate
            
            if not verified_email:
                print(f"  [-] Skipping: No verified contact email found on the web for '{company}' (Reasoning: {contact_info.get('reasoning') if contact_info else 'Search failed'})")
                continue
                
            # Fetch job description text
            job_desc_text = ""
            if apply_link:
                print(f"  └─ Fetching job details from: {apply_link[:60]}...")
                job_desc_text = fetch_job_description(apply_link)
                if job_desc_text:
                    print(f"  └─ Successfully fetched {len(job_desc_text)} chars.")
                else:
                    print("  └─ Direct job page scraping blocked; using company metadata fallback.")
                    
            # Build prompt for Step 2 (Stack Matching & Personalization)
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

Your Tasks:
1. Determine if this job role or the company matches the candidate's core programming languages (TypeScript, Python, JavaScript, Java, C, C++).
2. For the company_domain and contact_email fields in your response, use the values provided above: Domain = "{contact_info.get('company_domain')}", Email = "{verified_email}".
3. Draft a personalized, fluff-free cold email (<130 words) from the candidate to the career contact.
   - Align the candidate's key project achievements (e.g. SentinelLog-AI logs analyzer, the PostgreSQL RAG/pgvector engine, or the PropTech MVP) with the company's stack and business focus.
   - Keep the tone professional, low-friction, and conversational.
   - Explicitly mention in the body of the email that the resume is attached (e.g., "I've attached my resume for your consideration." or "Please find my resume attached.").
   - Mention the GitHub and LinkedIn profile links cleanly.
   - Do NOT ask if the candidate is free to chat this week or available to talk this week. Instead, the final sentence of the email body (just before the sign-off/signature) must be exactly: 'Waiting for your response.'

Ensure that the drafted email is strictly under 130 words.
"""

            try:
                print("  └─ Sending prompt to OpenRouter for stack evaluation and drafting...")
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
                    # Tailor resume for this company
                    tailored_resume_path = resume_path  # fallback to original
                    tailored = generate_tailored_text(client, resume_text, company, role, job_desc_text, res.get("company_domain", ""))
                    if tailored:
                        tailored_resume_path = tailor_resume_docx(resume_path, tailored, company)

                    matched_companies[company] = {
                        'role': role,
                        'apply_link': apply_link,
                        'location': location,
                        'domain': res.get('company_domain', ''),
                        'contact_email': res.get('contact_email', verified_email),
                        'subject': res.get('email_subject', ''),
                        'body': res.get('email_body', ''),
                        'is_india': is_india,
                        'tailored_resume': tailored_resume_path
                    }
                    print(f"  [+] Match Approved! Verified Contact: {res.get('contact_email', verified_email)}")
                else:
                    reason = res.get('reason', 'Unknown') if isinstance(res, dict) else 'API Error'
                    print(f"  [-] Match Rejected: {reason}")
            except Exception as e:
                print(f"  [-] OpenRouter evaluation failed: {e}")

    # Evaluate India list
    evaluate_postings_list(india_postings, target_india, "India", is_india=True)
    
    # Evaluate Abroad list
    evaluate_postings_list(abroad_postings, target_abroad, "Abroad", is_india=False)

    print(f"\n[+] Staging Complete! Discovered {len(matched_companies)} matching companies.")

    # 3. Inbox Staging or Direct Sending
    if args.dry_run:
        if not matched_companies:
            print("[-] No companies matched the requirements. Exiting.")
            return
        print("\n" + "="*80)
        print("DRY RUN MODE: Outputting outreach drafts to terminal.")
        print("="*80)
        for comp, info in matched_companies.items():
            print(f"\nCompany: {comp} (Domain: {info['domain']}) [India: {info['is_india']}]")
            print(f"Role: {info['role']}")
            print(f"Contact Email: {info['contact_email']}")
            print(f"Tailored Resume: {info.get('tailored_resume', 'N/A')}")
            print(f"Subject: {info['subject']}")
            print(f"Body:\n{info['body']}")
            print("-" * 50)
        print("\n[+] Dry run finished successfully.")
    else:
        print("\n[*] Connecting securely to Gmail API...")
        gmail_service = get_gmail_service(args.non_interactive)
        print("[+] Gmail connection established!")
        
        if not matched_companies:
            print("[-] No companies matched the requirements.")
            send_summary_to_candidate(gmail_service, {}, args.send)
            return

        if args.send:
            print("\n[*] Sending outreach emails directly via Gmail...")
            for comp, info in matched_companies.items():
                attach_path = info.get('tailored_resume') or resume_path
                print(f"  └─ Sending email to {comp} ({info['contact_email']}) with {'tailored' if attach_path != resume_path else 'original'} resume...")
                sent_msg = send_gmail_email(gmail_service, info['contact_email'], info['subject'], info['body'], attach_path)
                if sent_msg:
                    print(f"     [+] Successfully sent email (ID: {sent_msg['id']})")
            print("\n[+] All outreach emails have been successfully sent!")
        else:
            print("\n[*] Creating pending drafts in Gmail...")
            for comp, info in matched_companies.items():
                attach_path = info.get('tailored_resume') or resume_path
                print(f"  └─ Creating draft for {comp} ({info['contact_email']}) with {'tailored' if attach_path != resume_path else 'original'} resume...")
                draft = stage_gmail_draft(gmail_service, info['contact_email'], info['subject'], info['body'], attach_path)
                if draft:
                    print(f"     [+] Successfully staged draft (ID: {draft['id']})")
            print("\n[+] All outreach drafts have been successfully staged in your Gmail Inbox!")
        
        send_summary_to_candidate(gmail_service, matched_companies, args.send)

if __name__ == '__main__':
    main()
