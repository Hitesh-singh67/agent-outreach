from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os

SCOPES = ['https://mail.google.com/']

def test_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Token expired. Refreshing...")
            try:
                creds.refresh(Request())
                print("Refresh successful!")
                with open('token.json', 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                print(f"Refresh failed: {e}")
                return
        else:
            print("No valid credentials or refresh token. Interaction required.")
            return

    try:
        service = build('gmail', 'v1', credentials=creds)
        profile = service.users().getProfile(userId='me').execute()
        print("Success! Gmail connected. Profile:")
        print(profile)
    except Exception as e:
        print(f"Error fetching profile: {e}")

if __name__ == '__main__':
    test_gmail()
