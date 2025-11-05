"""
Gmail Email Fetcher
==================
Automatically fetches the last 30 emails from Gmail using IMAP, extracts content,
and saves them to local files with attachments support.

Requirements:
    pip install secure-smtplib

Author: Email Automation Script
Version: 1.1
"""

import imaplib
import email
from email.header import decode_header
from email.message import EmailMessage
import os
import re
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gmail_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmailData:
    """Data class to hold email information"""
    def __init__(self):
        self.sender: str = ""
        self.subject: str = ""
        self.body: str = ""
        self.date: str = ""
        self.message_id: str = ""
        self.attachments: List[Dict[str, Any]] = []
        self.html_body: str = ""
        self.plain_body: str = ""

class GmailFetcher:
    """
    Gmail email fetcher using IMAP protocol.
    
    Handles:
    - Connecting to Gmail via IMAP
    - Fetching last 30 emails
    - Extracting content and attachments
    - Saving emails to local storage
    - Marking emails as read
    """
    
    def __init__(self, email_address: str, app_password: str, output_dir: str = "emails"):
        self.email_address = email_address
        self.app_password = app_password
        self.output_dir = Path(output_dir)
        self.attachments_dir = self.output_dir / "attachments"
        self.imap_server = None
        self.IMAP_HOST = "imap.gmail.com"
        self.IMAP_PORT = 993
        self._create_directories()
        self.stats = {
            'emails_processed': 0,
            'attachments_downloaded': 0,
            'errors': 0
        }
    
    def _create_directories(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.attachments_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directories: {self.output_dir}, {self.attachments_dir}")
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            raise
    
    def connect(self) -> bool:
        try:
            self.imap_server = imaplib.IMAP4_SSL(self.IMAP_HOST, self.IMAP_PORT)
            result = self.imap_server.login(self.email_address, self.app_password)
            if result[0] == 'OK':
                logger.info(f"Successfully connected to Gmail for {self.email_address}")
                return True
            else:
                logger.error(f"Login failed: {result}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Gmail: {e}")
            return False
    
    def disconnect(self) -> None:
        try:
            if self.imap_server:
                self.imap_server.close()
                self.imap_server.logout()
                logger.info("Disconnected from Gmail")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    def get_last_emails(self, folder: str = "INBOX", max_emails: int = 30) -> List[str]:
        """
        Get list of the last `max_emails` email IDs from specified folder.
        """
        try:
            result = self.imap_server.select(folder)
            if result[0] != 'OK':
                logger.error(f"Failed to select folder {folder}")
                return []
            
            result, message_ids = self.imap_server.search(None, 'ALL')
            if result == 'OK':
                all_email_ids = message_ids[0].split()
                logger.info(f"Total emails in folder: {len(all_email_ids)}")
                last_email_ids = all_email_ids[-max_emails:]
                logger.info(f"Fetching last {len(last_email_ids)} emails")
                return [email_id.decode() for email_id in last_email_ids]
            else:
                logger.error("Failed to search for emails")
                return []
        except Exception as e:
            logger.error(f"Error getting emails: {e}")
            return []
    
    def _decode_header(self, header: str) -> str:
        try:
            decoded_parts = decode_header(header)
            decoded_header = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    decoded_header += part.decode(encoding or 'utf-8', errors='ignore')
                else:
                    decoded_header += str(part)
            return decoded_header
        except Exception as e:
            logger.error(f"Error decoding header: {e}")
            return str(header)
    
    def _extract_body(self, msg: EmailMessage) -> Tuple[str, str]:
        plain_body = ""
        html_body = ""
        try:
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))
                    if "attachment" in content_disposition:
                        continue
                    if content_type == "text/plain":
                        body = part.get_payload(decode=True)
                        if body:
                            plain_body += body.decode('utf-8', errors='ignore')
                    elif content_type == "text/html":
                        body = part.get_payload(decode=True)
                        if body:
                            html_body += body.decode('utf-8', errors='ignore')
            else:
                content_type = msg.get_content_type()
                body = msg.get_payload(decode=True)
                if body:
                    decoded_body = body.decode('utf-8', errors='ignore')
                    if content_type == "text/html":
                        html_body = decoded_body
                    else:
                        plain_body = decoded_body
        except Exception as e:
            logger.error(f"Error extracting email body: {e}")
        return plain_body, html_body
    
    def _extract_attachments(self, msg: EmailMessage, email_id: str) -> List[Dict[str, Any]]:
        attachments = []
        try:
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition", ""))
                if "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        filename = self._decode_header(filename)
                        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_filename = f"{email_id}_{timestamp}_{filename}"
                        file_path = self.attachments_dir / safe_filename
                        try:
                            payload = part.get_payload(decode=True)
                            if payload:
                                with open(file_path, 'wb') as f:
                                    f.write(payload)
                                attachment_info = {
                                    'original_filename': filename,
                                    'saved_filename': safe_filename,
                                    'file_path': str(file_path),
                                    'size': len(payload),
                                    'content_type': part.get_content_type()
                                }
                                attachments.append(attachment_info)
                                self.stats['attachments_downloaded'] += 1
                                logger.info(f"Saved attachment: {safe_filename}")
                        except Exception as e:
                            logger.error(f"Error saving attachment {filename}: {e}")
                            self.stats['errors'] += 1
        except Exception as e:
            logger.error(f"Error extracting attachments: {e}")
        return attachments
    
    def _parse_email(self, email_id: str, raw_email: bytes) -> Optional[EmailData]:
        try:
            msg = email.message_from_bytes(raw_email)
            email_data = EmailData()
            email_data.sender = self._decode_header(msg.get("From", ""))
            email_data.subject = self._decode_header(msg.get("Subject", ""))
            email_data.date = msg.get("Date", "")
            email_data.message_id = msg.get("Message-ID", email_id)
            email_data.plain_body, email_data.html_body = self._extract_body(msg)
            email_data.body = email_data.plain_body or email_data.html_body
            email_data.attachments = self._extract_attachments(msg, email_id)
            return email_data
        except Exception as e:
            logger.error(f"Error parsing email {email_id}: {e}")
            self.stats['errors'] += 1
            return None
    
    def _save_email_to_file(self, email_data: EmailData, email_id: str) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_subject = re.sub(r'[<>:"/\\|?*]', '_', email_data.subject[:50])
            filename = f"email_{email_id}_{timestamp}_{safe_subject}.txt"
            file_path = self.output_dir / filename
            content = [
                "="*80,
                "EMAIL DETAILS",
                "="*80,
                f"Email ID: {email_id}",
                f"Message ID: {email_data.message_id}",
                f"Date: {email_data.date}",
                f"From: {email_data.sender}",
                f"Subject: {email_data.subject}",
                f"Downloaded: {datetime.now().isoformat()}",
            ]
            if email_data.attachments:
                content.append(f"Attachments: {len(email_data.attachments)}")
                for i, attachment in enumerate(email_data.attachments, 1):
                    content.append(f"  {i}. {attachment['original_filename']} "
                                   f"({attachment['size']} bytes) -> {attachment['saved_filename']}")
            else:
                content.append("Attachments: None")
            content.extend([
                "\n" + "="*80,
                "EMAIL BODY",
                "="*80,
                email_data.body
            ])
            if email_data.html_body and email_data.html_body != email_data.plain_body:
                content.extend([
                    "\n" + "="*80,
                    "HTML BODY",
                    "="*80,
                    email_data.html_body
                ])
            with open(file_path, 'w', encoding='utf-8', errors='ignore') as f:
                f.write('\n'.join(content))
            logger.info(f"Saved email to: {filename}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Error saving email to file: {e}")
            self.stats['errors'] += 1
            return ""
    
    def _save_email_to_database(self, email_data: EmailData, email_id: str) -> bool:
        logger.info("Database saving not implemented yet")
        return False
    
    def mark_as_read(self, email_id: str) -> bool:
        try:
            result = self.imap_server.store(email_id, '+FLAGS', '\\Seen')
            if result[0] == 'OK':
                logger.info(f"Marked email {email_id} as read")
                return True
            else:
                logger.error(f"Failed to mark email {email_id} as read: {result}")
                return False
        except Exception as e:
            logger.error(f"Error marking email as read: {e}")
            return False
    
    def fetch_and_save_emails(self, mark_read: bool = True, save_method: str = "file") -> Dict[str, Any]:
        logger.info("Starting email fetch and save process...")
        self.stats = {'emails_processed': 0, 'attachments_downloaded': 0, 'errors': 0}
        try:
            # Get last 30 emails
            email_ids = self.get_last_emails(max_emails=30)
            if not email_ids:
                logger.info("No emails found")
                return self.stats
            
            for email_id in email_ids:
                try:
                    logger.info(f"Processing email {email_id}")
                    result, email_data_raw = self.imap_server.fetch(email_id, "(RFC822)")
                    if result != 'OK':
                        logger.error(f"Failed to fetch email {email_id}")
                        self.stats['errors'] += 1
                        continue
                    parsed_email = self._parse_email(email_id, email_data_raw[0][1])
                    if not parsed_email:
                        logger.error(f"Failed to parse email {email_id}")
                        continue
                    saved = False
                    if save_method in ["file", "both"]:
                        file_path = self._save_email_to_file(parsed_email, email_id)
                        saved = bool(file_path)
                    if save_method in ["database", "both"]:
                        db_saved = self._save_email_to_database(parsed_email, email_id)
                        saved = saved or db_saved
                    if saved:
                        self.stats['emails_processed'] += 1
                        if mark_read:
                            self.mark_as_read(email_id)
                except Exception as e:
                    logger.error(f"Error processing email {email_id}: {e}")
                    self.stats['errors'] += 1
            logger.info(f"Email processing complete. Stats: {self.stats}")
        except Exception as e:
            logger.error(f"Error in fetch and save process: {e}")
            self.stats['errors'] += 1
        return self.stats

def create_config_file(config_path: str = "gmail_config.json") -> None:
    config = {
        "email_address": "nandabiju2020@gmail.com",
        "app_password": "ucmw msyi grvc qqah",
        "output_directory": "emails",
        "mark_as_read": True,
        "save_method": "file"
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Sample configuration file created: {config_path}")
    print("Please edit this file with your Gmail credentials and settings.")

def main():
    config_file = "gmail_config.json"
    if not os.path.exists(config_file):
        create_config_file(config_file)
        return
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        EMAIL_ADDRESS = config.get("email_address")
        APP_PASSWORD = config.get("app_password")
        OUTPUT_DIR = config.get("output_directory", "emails")
        MARK_AS_READ = config.get("mark_as_read", True)
        SAVE_METHOD = config.get("save_method", "file")
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        print("Using default configuration...")
    
    if EMAIL_ADDRESS == "your_email@gmail.com" or APP_PASSWORD == "your_app_password":
        print("Please configure your Gmail credentials in gmail_config.json")
        return
    
    fetcher = GmailFetcher(EMAIL_ADDRESS, APP_PASSWORD, OUTPUT_DIR)
    
    try:
        if not fetcher.connect():
            logger.error("Failed to connect to Gmail")
            return
        stats = fetcher.fetch_and_save_emails(mark_read=MARK_AS_READ, save_method=SAVE_METHOD)
        print("\n" + "="*60)
        print("EMAIL FETCH RESULTS")
        print("="*60)
        print(f"Emails processed: {stats['emails_processed']}")
        print(f"Attachments downloaded: {stats['attachments_downloaded']}")
        print(f"Errors encountered: {stats['errors']}")
        print(f"Output directory: {OUTPUT_DIR}")
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        fetcher.disconnect()

if __name__ == "__main__":
    main()
