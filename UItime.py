"""
SECUROX Email Analysis System - UI with Real Gmail Integration
Integrated with DistilBERT model and GmailFetcher for real email processing.
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from collections import deque

# --- PySide6 imports ---
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QProgressBar, QGroupBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QStatusBar, QDialog, QDialogButtonBox, QFormLayout, QLineEdit,
    QSpinBox, QCheckBox, QFrame, QMessageBox, QFileDialog,
    QSystemTrayIcon, QMenu, QAbstractItemView, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont, QPalette, QColor, QPainter

# --- ML imports ---
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# --- LIME for Explainable AI ---
try:
    import lime
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not available. Explainable AI features disabled.")

# --- Gmail Fetcher import ---
try:
    from gmail_fetcher import GmailFetcher, EmailData
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    logging.warning("gmail_fetcher.py not found. Gmail integration disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "../models/securox_distilbert"
LABEL_MAP = {
    0: ("Legit", "#4CAF50"),
    1: ("Spam", "#FF9800"),
    2: ("Phishing", "#F44336")
}

# Load model & tokenizer
_model = None
_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_loaded = False
_lime_explainer = None

# Model Wrapper for LIME
class ModelWrapper:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def predict_proba(self, texts):
        # Handle both single string and list of strings
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probs

def apply_heuristics(sender: str, email_text: str, model_label: str) -> str:
    sender = sender.lower() if sender else ""
    text = email_text.lower() if email_text else ""
    label = model_label.lower()

    SAFE_SENDERS = ["google", "google security", "no-reply@google.com", "gmail", "googlealerts","nptel"]
    if any(safe in sender for safe in SAFE_SENDERS):
        return "Legit"  # Changed to capitalized

    PROMO_KEYWORDS = ["exciting offer", "job alert", "don't miss out", "dont miss out",
                      "promotion", "special deal", "limited time", "exclusive offer","Job opportunities","job offer"]
    if any(word in text for word in PROMO_KEYWORDS):
        return "Spam"  # Changed to capitalized

    PHISH_KEYWORDS = ["account suspended", "bank account", "card number",
                      "verify identity", "reset password", "unauthorized access"]
    if label == "spam" and any(word in text for word in PHISH_KEYWORDS):
        return "Phishing"  # Changed to capitalized

    return model_label  # Return the original model_label (already capitalized)

try:
    logger.info(f"Loading model from {MODEL_PATH} on device {_device}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.to(_device)
    _model.eval()
    _model_loaded = True
    
    # Initialize LIME explainer if available
    if LIME_AVAILABLE and _model_loaded:
        model_wrapper = ModelWrapper(_model, _tokenizer, _device)
        _lime_explainer = lime.lime_text.LimeTextExplainer(
            class_names=['Legit', 'Spam', 'Phishing'],
            bow=False
        )
        logger.info("LIME explainer initialized successfully.")
    
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Could not load model from {MODEL_PATH}: {e}")
    logger.info("Falling back to heuristic analyzer.")

def check_lime_availability():
    """Check if LIME is properly available and working"""
    if not LIME_AVAILABLE:
        logger.error("LIME is not available. Please install: pip install lime")
        return False
    
    if _lime_explainer is None:
        logger.error("LIME explainer is not initialized")
        return False
    
    if not _model_loaded:
        logger.error("Model is not loaded")
        return False
    
    # Test with a simple example
    try:
        test_text = "This is a test email with urgent account verification required"
        test_label, test_confidence, _ = classify_email(test_text)
        explanation = explain_prediction(test_text, test_label)
        
        if explanation:
            logger.info("LIME is working correctly")
            return True
        else:
            logger.warning("LIME explanation returned None")
            return False
            
    except Exception as e:
        logger.error(f"LIME test failed: {e}")
        return False

def test_lime_explanation():
    """Test LIME explanation with a known phishing email"""
    test_phishing_text = """
    URGENT: Your account has been suspended!
    Click here to verify your identity: http://fake-bank.com/verify
    We need your login credentials to restore access.
    This is required immediately to avoid permanent account closure.
    """
    
    logger.info("Testing LIME with phishing example...")
    label, confidence, color = classify_email(test_phishing_text)
    logger.info(f"Classification: {label} ({confidence:.1f}%)")
    
    explanation = explain_prediction(test_phishing_text, label)
    if explanation:
        features = explanation.get('explanation', [])
        logger.info(f"Generated {len(features)} explanation features")
        for feature, weight in features[:5]:  # Show top 5
            logger.info(f"  {feature}: {weight:+.3f}")
    else:
        logger.warning("LIME explanation not available for test case")

def classify_email(text: str, sender: str = ""):
    """Classify email text using DistilBERT or heuristic fallback"""
    if not text:
        return ("Legit", 100.0, LABEL_MAP[0][1])

    if _model_loaded and _model is not None and _tokenizer is not None:
        try:
            inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = _model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().squeeze()
            pred = int(torch.argmax(probs, dim=0).item())
            confidence = float(probs[pred].item() * 100.0)
            model_label, color = LABEL_MAP.get(pred, ("Unknown", "#CCCCCC"))
            
            # Apply heuristic override
            final_label = apply_heuristics(sender, text, model_label)
            
            return (final_label, confidence, color)
        except Exception as e:
            logger.error(f"Error during model inference: {e}")

    # Heuristic fallback
    text_lower = text.lower()
    phishing_indicators = ['verify', 'urgent', 'suspended', 'click here', 'account', 'login', 'password', 'update']
    spam_indicators = ['winner', 'prize', 'congratulations', 'lottery', 'free', 'offer', 'discount']

    phishing_score = sum(1 for k in phishing_indicators if k in text_lower)
    spam_score = sum(1 for k in spam_indicators if k in text_lower)

    if phishing_score >= 2:
        model_label = "Phishing"
    elif spam_score >= 2:
        model_label = "Spam"
    else:
        model_label = "Legit"
    
    # Apply heuristic override for fallback classification too
    final_label = apply_heuristics(sender, text, model_label)
    
    if final_label == "Phishing":
        return ("Phishing", 85.0, LABEL_MAP[2][1])
    elif final_label == "Spam":
        return ("Spam", 70.0, LABEL_MAP[1][1])
    else:
        return ("Legit", 90.0, LABEL_MAP[0][1])

def explain_prediction(text: str, label: str):
    """Generate LIME explanation for email classification"""
    if not LIME_AVAILABLE:
        logger.warning("LIME not available")
        return None
        
    if _lime_explainer is None:
        logger.warning("LIME explainer not initialized")
        return None
        
    if not text or len(text.strip()) < 10:
        logger.warning("Text too short for explanation")
        return None
    
    try:
        # Simple model wrapper
        def predict_proba(texts):
            if isinstance(texts, str):
                texts = [texts]
                
            inputs = _tokenizer(
                texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = _model(**inputs)
            
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            return probs
        
        # Get class index - Use the correct mapping from LABEL_MAP
        class_idx = None
        for idx, (class_name, _) in LABEL_MAP.items():
            if class_name == label:
                class_idx = idx
                break
        
        if class_idx is None:
            logger.warning(f"Could not find index for label: {label}")
            class_idx = 0  # Default to legit
        
        logger.info(f"Generating LIME explanation for class {class_idx} ({label})...")
        
        # Generate explanation
        exp = _lime_explainer.explain_instance(
            text, 
            predict_proba,
            num_features=8,
            top_labels=3,  # Get all 3 labels
            num_samples=50  # Reduced for speed
        )
        
        # Get explanation for our predicted class
        explanation = exp.as_list(label=class_idx)
        
        logger.info(f"Successfully generated LIME explanation with {len(explanation)} features")
        
        return {
            'explanation': explanation,
            'score': exp.score if hasattr(exp, 'score') else 1.0,
        }
        
    except Exception as e:
        logger.error(f"LIME error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

class EmailItem:
    """Data class for email items"""
    def __init__(self):
        self.id: str = ""
        self.sender: str = ""
        self.subject: str = ""
        self.body: str = ""
        self.date: datetime = datetime.now()
        self.classification: str = "Unknown"
        self.confidence: float = 0.0
        self.risk_score: float = 0.0
        self.status: str = "Pending"
        self.attachments: List[str] = []
        self.file_path: str = ""
        self.explanation: Optional[Dict] = None

class ExplanationPanel(QWidget):
    """Side panel to show LIME explanations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.setFixedWidth(400)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = QLabel("🔍 AI Explanation")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #03DAC6; padding: 10px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Close button
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(30, 30)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                border-radius: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #E57373;
            }
        """)
        close_btn.clicked.connect(self.hide)
        
        header_layout = QHBoxLayout()
        header_layout.addWidget(header)
        header_layout.addWidget(close_btn)
        layout.addLayout(header_layout)
        
        # Email info
        self.email_info = QLabel()
        self.email_info.setWordWrap(True)
        self.email_info.setStyleSheet("background-color: #2D2D2D; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.email_info)
        
        # Explanation section
        explanation_group = QGroupBox("🤖 Why this classification?")
        explanation_layout = QVBoxLayout()
        
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMaximumHeight(200)
        explanation_layout.addWidget(self.explanation_text)
        
        explanation_group.setLayout(explanation_layout)
        layout.addWidget(explanation_group)
        
        # Feature importance
        features_group = QGroupBox("📊 Key Features")
        features_layout = QVBoxLayout()
        
        self.features_list = QTextEdit()
        self.features_list.setReadOnly(True)
        features_layout.addWidget(self.features_list)
        
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        # Risk factors
        risk_group = QGroupBox("⚠️ Risk Factors")
        risk_layout = QVBoxLayout()
        
        self.risk_factors = QTextEdit()
        self.risk_factors.setReadOnly(True)
        risk_layout.addWidget(self.risk_factors)
        
        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.setStyleSheet("""
            ExplanationPanel {
                background-color: #1E1E1E;
                border-left: 2px solid #444444;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #444444;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                color: #03DAC6;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTextEdit {
                background-color: #2D2D2D;
                border: 1px solid #444444;
                border-radius: 4px;
                color: #E0E0E0;
                padding: 5px;
            }
        """)
    
    def show_explanation(self, email: EmailItem):
        """Show explanation for the given email"""
        # Generate explanation if not already available
        if not email.explanation:
            text = f"{email.subject} {email.body}"
            logger.info(f"Generating explanation for email: {email.subject}")
            email.explanation = explain_prediction(text, email.classification)
        
        # Update email info
        self.email_info.setText(
            f"<b>From:</b> {email.sender}<br>"
            f"<b>Subject:</b> {email.subject}<br>"
            f"<b>Classification:</b> <span style='color:{LABEL_MAP[2][1] if email.classification == 'Phishing' else LABEL_MAP[1][1] if email.classification == 'Spam' else LABEL_MAP[0][1]}'>{email.classification}</span><br>"
            f"<b>Confidence:</b> {email.confidence:.1f}%"
        )
        
        if email.explanation:
            explanation_data = email.explanation
            features = explanation_data.get('explanation', [])
            
            # Update explanation text
            if explanation_data.get('score'):
                score_text = f"Explanation quality: {explanation_data['score']:.3f}"
            else:
                score_text = "Explanation generated successfully"
            
            self.explanation_text.setText(
                f"The AI classified this email as {email.classification} because of the following factors:\n\n"
                f"{score_text}"
            )
            
            # Update features list
            if features:
                features_html = "<table width='100%'>"
                for feature, weight in features:
                    color = "#F44336" if weight > 0 else "#4CAF50"
                    features_html += f"<tr><td>{feature}</td><td style='color: {color}; text-align: right;'>{weight:+.3f}</td></tr>"
                features_html += "</table>"
                self.features_list.setHtml(features_html)
            else:
                self.features_list.setText("No significant features identified.")
            
            # Update risk factors
            risk_factors = []
            for feature, weight in features:
                if weight > 0:  # Features that contributed to phishing classification
                    risk_factors.append(f"• {feature} (weight: {weight:+.3f})")
            
            if risk_factors:
                self.risk_factors.setText("\n".join(risk_factors))
            else:
                self.risk_factors.setText("No strong risk factors identified.")
                
        else:
            # Provide more detailed error information
            error_details = [
                "No detailed explanation available.",
                "Possible reasons:",
                "• LIME package not installed",
                "• Text too short for analysis",
                "• Model not properly loaded",
                "• Error during explanation generation"
            ]
            self.explanation_text.setText("\n".join(error_details))
            self.features_list.setText("Feature importance data not available.")
            self.risk_factors.setText("Risk factor analysis not available.")
        
        self.setVisible(True)

class GmailFetchWorker(QThread):
    """Worker thread for fetching emails from Gmail - Now with continuous monitoring"""
    email_fetched = Signal(EmailItem)
    status_updated = Signal(str)
    progress_updated = Signal(int)
    error_occurred = Signal(str)
    batch_completed = Signal(dict)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.is_running = False
        self.fetched_emails = []
        self.processed_email_ids = set()  # Track already processed emails
        self.monitoring_interval = self.config.get('monitoring_interval', 60)  # Default 60 seconds
        
    def run(self):
        """Continuously monitor Gmail for new emails"""
        self.is_running = True
        
        while self.is_running:
            try:
                self.status_updated.emit("Monitoring Gmail for new emails...")
                
                if self._fetch_and_analyze_new_emails():
                    self.status_updated.emit(f"Monitoring active - checking every {self.monitoring_interval} seconds")
                else:
                    self.status_updated.emit("Monitoring paused due to error")
                
                # Wait before next check
                for i in range(self.monitoring_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
                        
            except Exception as e:
                self.error_occurred.emit(f"Monitoring error: {str(e)}")
                # Wait before retrying
                for i in range(30):  # 30 second retry delay
                    if not self.is_running:
                        break
                    time.sleep(1)
    
    def _fetch_and_analyze_new_emails(self) -> bool:
        """Fetch and analyze only new emails from Gmail"""
        if not GMAIL_AVAILABLE:
            self.error_occurred.emit("Gmail fetcher not available. Check gmail_fetcher.py")
            return False
            
        try:
            email_address = self.config.get('email')
            app_password = self.config.get('app_password')
            folder = self.config.get('folder', 'INBOX')
            mark_read = self.config.get('mark_read', True)
            
            if not email_address or not app_password:
                self.error_occurred.emit("Email credentials not configured")
                return False
            
            # Create temporary directory for email storage
            temp_dir = Path("temp_emails")
            fetcher = GmailFetcher(email_address, app_password, str(temp_dir))
            
            # Connect to Gmail
            if not fetcher.connect():
                self.error_occurred.emit("Failed to connect to Gmail. Check credentials.")
                return False
            
            # Get recent emails (limit to 50 to check for new ones)
            email_ids = fetcher.get_last_emails(folder=folder, max_emails=50)
            
            if not email_ids:
                fetcher.disconnect()
                return True  # No emails found, but monitoring should continue
            
            new_emails_found = 0
            total = len(email_ids)
            
            for idx, email_id in enumerate(email_ids):
                if not self.is_running:
                    break
                
                # Skip already processed emails
                if email_id in self.processed_email_ids:
                    continue
                
                try:
                    # Fetch email
                    result, email_data_raw = fetcher.imap_server.fetch(email_id, "(RFC822)")
                    if result != 'OK':
                        continue
                    
                    # Parse email
                    parsed_email = fetcher._parse_email(email_id, email_data_raw[0][1])
                    if not parsed_email:
                        continue
                    
                    # Save to file
                    file_path = fetcher._save_email_to_file(parsed_email, email_id)
                    
                    # Mark as read if configured
                    if mark_read:
                        fetcher.mark_as_read(email_id)
                    
                    # Convert to EmailItem
                    email_item = EmailItem()
                    email_item.id = email_id
                    email_item.sender = parsed_email.sender
                    email_item.subject = parsed_email.subject
                    email_item.body = parsed_email.body
                    email_item.file_path = file_path
                    
                    # Parse date
                    try:
                        from email.utils import parsedate_to_datetime
                        email_item.date = parsedate_to_datetime(parsed_email.date)
                    except:
                        email_item.date = datetime.now()
                    
                    # Store attachments info
                    email_item.attachments = [att['original_filename'] for att in parsed_email.attachments]
                    
                    # Analyze the email
                    self._analyze_single_email(email_item)
                    
                    # Add to processed set and emit signal
                    self.processed_email_ids.add(email_id)
                    self.email_fetched.emit(email_item)
                    
                    new_emails_found += 1
                    self.progress_updated.emit(int((idx + 1) / total * 100))
                    
                    # Small delay to avoid overwhelming the system
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing email {email_id}: {e}")
                    continue
            
            fetcher.disconnect()
            
            if new_emails_found > 0:
                self.batch_completed.emit({
                    'count': new_emails_found,
                    'timestamp': datetime.now().isoformat(),
                    'new_emails': True
                })
            
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"Error fetching emails: {str(e)}")
            return False
    
    def _analyze_single_email(self, email: EmailItem):
        """Analyze a single email using the model"""
        try:
            text = f"{email.subject} {email.body}"
            label, confidence, color = classify_email(text, email.sender)
            email.classification = label
            email.confidence = confidence
            
            if label == "Phishing":
                email.status = "Quarantined"
                email.risk_score = min(100.0, confidence)
            elif label == "Spam":
                email.status = "Analyzed"
                email.risk_score = min(100.0, confidence * 0.8)
            else:
                email.status = "Safe"
                email.risk_score = max(0.0, 100.0 - confidence * 0.2)
        except Exception as e:
            logger.error(f"Error analyzing email {email.id}: {e}")
            email.classification = "Legit"
            email.confidence = 60.0
            email.status = "Safe"
            email.risk_score = 10.0
    
    def stop(self):
        """Stop the worker"""
        self.is_running = False

class ConfigurationDialog(QDialog):
    """Configuration dialog with monitoring settings"""
    
    def __init__(self, parent=None, current_config=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration Settings")
        self.setFixedSize(500, 550)  # Increased height for new setting
        self.current_config = current_config or {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Gmail Settings
        gmail_group = QGroupBox("📧 Gmail Settings")
        gmail_layout = QFormLayout()
        
        self.email_input = QLineEdit(self.current_config.get('email', ''))
        self.email_input.setPlaceholderText('your-email@gmail.com')
        gmail_layout.addRow('Email Address:', self.email_input)
        
        self.password_input = QLineEdit(self.current_config.get('app_password', ''))
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText('16-character app password')
        gmail_layout.addRow('App Password:', self.password_input)
        
        self.folder_input = QLineEdit(self.current_config.get('folder', 'INBOX'))
        gmail_layout.addRow('Email Folder:', self.folder_input)
        
        self.mark_read = QCheckBox()
        self.mark_read.setChecked(self.current_config.get('mark_read', True))
        gmail_layout.addRow('Mark as Read:', self.mark_read)
        
        gmail_group.setLayout(gmail_layout)
        layout.addWidget(gmail_group)
        
        # Monitoring Settings
        monitoring_group = QGroupBox("🔄 Monitoring Settings")
        monitoring_layout = QFormLayout()
        
        self.monitoring_interval = QSpinBox()
        self.monitoring_interval.setRange(10, 300)  # 10 seconds to 5 minutes
        self.monitoring_interval.setValue(self.current_config.get('monitoring_interval', 60))
        self.monitoring_interval.setSuffix(' seconds')
        self.monitoring_interval.setToolTip("How often to check for new emails")
        monitoring_layout.addRow('Check Interval:', self.monitoring_interval)
        
        self.auto_start = QCheckBox()
        self.auto_start.setChecked(self.current_config.get('auto_start', False))
        self.auto_start.setToolTip("Start monitoring automatically when application launches")
        monitoring_layout.addRow('Auto-start Monitoring:', self.auto_start)
        
        monitoring_group.setLayout(monitoring_layout)
        layout.addWidget(monitoring_group)
        
        # Analysis Settings
        analysis_group = QGroupBox("🔍 Analysis Settings")
        analysis_layout = QFormLayout()
        
        self.auto_quarantine = QCheckBox()
        self.auto_quarantine.setChecked(self.current_config.get('auto_quarantine', True))
        analysis_layout.addRow('Auto Quarantine:', self.auto_quarantine)
        
        self.confidence_threshold = QSpinBox()
        self.confidence_threshold.setRange(50, 95)
        self.confidence_threshold.setValue(self.current_config.get('confidence_threshold', 75))
        self.confidence_threshold.setSuffix('%')
        analysis_layout.addRow('Confidence Threshold:', self.confidence_threshold)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Notification Settings
        notif_group = QGroupBox("🔔 Notifications")
        notif_layout = QFormLayout()
        
        self.system_notifications = QCheckBox()
        self.system_notifications.setChecked(self.current_config.get('notifications', True))
        notif_layout.addRow('System Notifications:', self.system_notifications)
        
        self.email_alerts = QCheckBox()
        self.email_alerts.setChecked(self.current_config.get('email_alerts', False))
        notif_layout.addRow('Email Alerts:', self.email_alerts)
        
        notif_group.setLayout(notif_layout)
        layout.addWidget(notif_group)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E1E;
                color: #E0E0E0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444444;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                color: #03DAC6;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLineEdit, QSpinBox {
                background-color: #2D2D2D;
                border: 2px solid #444444;
                border-radius: 4px;
                padding: 5px;
                color: #E0E0E0;
            }
            QLineEdit:focus, QSpinBox:focus {
                border-color: #03DAC6;
            }
            QCheckBox {
                color: #E0E0E0;
            }
            QCheckBox::indicator:checked {
                background-color: #03DAC6;
                border: 2px solid #03DAC6;
            }
            QPushButton {
                background-color: #6200EE;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7C43BD;
            }
        """)

    def get_config(self):
        """Return configuration with monitoring settings"""
        return {
            "email": self.email_input.text().strip(),
            "app_password": self.password_input.text().strip(),
            "folder": self.folder_input.text().strip() or "INBOX",
            "mark_read": self.mark_read.isChecked(),
            "monitoring_interval": self.monitoring_interval.value(),
            "auto_start": self.auto_start.isChecked(),
            "auto_quarantine": self.auto_quarantine.isChecked(),
            "confidence_threshold": self.confidence_threshold.value(),
            "notifications": self.system_notifications.isChecked(),
            "email_alerts": self.email_alerts.isChecked()
        }

class EmailDashboard(QWidget):
    """Dashboard showing email statistics with explanation panel"""
    
    def __init__(self):
        super().__init__()
        self.stats = {
            'total_processed': 0,
            'phishing_detected': 0,
            'spam_detected': 0,
            'legitimate': 0,
            'quarantined': 0
        }
        self.explanation_panel = ExplanationPanel()
        self.init_ui()
        
    def init_ui(self):
        # Create splitter for main layout
        splitter = QSplitter(Qt.Horizontal)
        
        # Main content widget
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        
        # Statistics cards
        stats_layout = QHBoxLayout()
        
        self.stat_cards = {}
        stats_data = [
            ('Total Processed', 'total_processed', '#03DAC6'),
            ('Phishing Detected', 'phishing_detected', '#F44336'),
            ('Spam Detected', 'spam_detected', '#FF9800'),
            ('Legitimate', 'legitimate', '#4CAF50'),
            ('Quarantined', 'quarantined', '#9C27B0')
        ]
        
        for title, key, color in stats_data:
            card = self._create_stat_card(title, 0, color)
            self.stat_cards[key] = card
            stats_layout.addWidget(card)
        
        layout.addLayout(stats_layout)
        
        # Recent activity
        activity_group = QGroupBox("📊 Recent Activity ")
        activity_layout = QVBoxLayout()
        
        self.activity_list = QTableWidget()
        self.activity_list.setColumnCount(5)
        self.activity_list.setHorizontalHeaderLabels(['Time', 'Sender', 'Subject', 'Classification', 'Status'])
        self.activity_list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.activity_list.doubleClicked.connect(self.on_activity_double_click)
        
        header = self.activity_list.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        
        self.activity_list.setColumnWidth(0, 100)
        self.activity_list.setColumnWidth(3, 140)
        self.activity_list.setColumnWidth(4, 100)
        
        activity_layout.addWidget(self.activity_list)
        activity_group.setLayout(activity_layout)
        layout.addWidget(activity_group)
        
        main_widget.setLayout(layout)
        
        # Add widgets to splitter
        splitter.addWidget(main_widget)
        splitter.addWidget(self.explanation_panel)
        
        # Set splitter properties
        splitter.setSizes([800, 400])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)
        
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def on_activity_double_click(self, index):
        """Handle double click on activity list - show explanation for phishing emails"""
        row = index.row()
        sender = self.activity_list.item(row, 1).text()
        subject = self.activity_list.item(row, 2).text()
        classification_item = self.activity_list.item(row, 3)
        classification = classification_item.text().replace('● ', '')
        
        # Only show explanation for phishing emails
        if classification == "Phishing":
            # Find the corresponding email object
            main_window = self.get_main_window()
            if main_window:
                for email in main_window.emails:
                    if email.sender == sender and email.subject.startswith(subject.replace('...', '')):
                        self.explanation_panel.show_explanation(email)
                        break
    
    def get_main_window(self):
        """Get reference to main window"""
        parent = self.parent()
        while parent and not isinstance(parent, SecuroxMainWindow):
            parent = parent.parent()
        return parent
    
    def _create_stat_card(self, title: str, value: int, color: str) -> QFrame:
        """Create a statistics card"""
        card = QFrame()
        card.setFrameStyle(QFrame.StyledPanel)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #2D2D2D;
                border: 2px solid {color};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        
        layout = QVBoxLayout()
        
        value_label = QLabel(str(value))
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setFont(QFont("Arial", 24, QFont.Bold))
        value_label.setStyleSheet(f"color: {color};")
        
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        title_label.setStyleSheet("color: #E0E0E0;")
        
        layout.addWidget(value_label)
        layout.addWidget(title_label)
        card.setLayout(layout)
        
        card.value_label = value_label
        
        return card
    
    def update_stats(self, new_stats: Dict[str, int]):
        """Update dashboard statistics"""
        self.stats.update(new_stats)
        
        for key, value in self.stats.items():
            if key in self.stat_cards:
                self.stat_cards[key].value_label.setText(str(value))
    
    def add_activity(self, email: EmailItem):
        """Add new activity to the table"""
        row = self.activity_list.rowCount()
        self.activity_list.insertRow(row)
        
        time_item = QTableWidgetItem(email.date.strftime('%H:%M'))
        self.activity_list.setItem(row, 0, time_item)
        
        sender_item = QTableWidgetItem(email.sender)
        self.activity_list.setItem(row, 1, sender_item)
        
        subject_item = QTableWidgetItem(email.subject[:50] + ('...' if len(email.subject) > 50 else ''))
        self.activity_list.setItem(row, 2, subject_item)
        
        colors = {'Phishing': '#F44336', 'Spam': '#FF9800', 'Legit': '#4CAF50'}
        dot = '●'
        class_text = f"{dot} {email.classification}"
        class_item = QTableWidgetItem(class_text)
        if email.classification in colors:
            class_item.setForeground(QColor(colors[email.classification]))
        self.activity_list.setItem(row, 3, class_item)
        
        status_item = QTableWidgetItem(email.status)
        self.activity_list.setItem(row, 4, status_item)
        
        # Update statistics when adding new activity
        self.stats['total_processed'] += 1
        if email.classification == "Phishing":
            self.stats['phishing_detected'] += 1
            if email.status == "Quarantined":
                self.stats['quarantined'] += 1
        elif email.classification == "Spam":
            self.stats['spam_detected'] += 1
        else:
            self.stats['legitimate'] += 1
        
        # Update the stat cards
        self.update_stats(self.stats)
        
        if self.activity_list.rowCount() > 50:
            self.activity_list.removeRow(0)
        
        self.activity_list.scrollToBottom()

class SecuroxMainWindow(QMainWindow):
    """Main window with real Gmail integration and LIME explanations"""
    
    def __init__(self):
        super().__init__()
        self.fetch_worker = None
        self.config = {}
        self.emails = deque(maxlen=30)
        self.tray_icon = None
        self.current_manual_analysis = None
        
        self.init_ui()
        self.load_config()
        self.setup_system_tray()
        
        # Test LIME functionality on startup
        check_lime_availability()
        test_lime_explanation()
        
        # Auto-start monitoring if configured
        if self.config.get('auto_start', False):
            QTimer.singleShot(2000, self.start_monitoring)  # Start after 2 seconds
        
    def init_ui(self):
        self.setWindowTitle("🔒 SECUROX - Email Analysis System")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)
        
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
                color: #E0E0E0;
            }
            QTabWidget::pane {
                border: 2px solid #444444;
                border-radius: 8px;
                background-color: #1E1E1E;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #E0E0E0;
                padding: 10px 20px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #6200EE;
                color: white;
            }
            QTabBar::tab:hover {
                background-color: #444444;
            }
            QPushButton {
                background-color: #6200EE;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7C43BD;
            }
            QPushButton:pressed {
                background-color: #3700B3;
            }
            QPushButton:disabled {
                background-color: #444444;
                color: #888888;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #444444;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                color: #03DAC6;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTableWidget {
                background-color: #2D2D2D;
                border: 1px solid #444444;
                border-radius: 4px;
                gridline-color: #444444;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #444444;
            }
            QTableWidget::item:selected {
                background-color: #6200EE;
            }
            QHeaderView::section {
                background-color: #1E1E1E;
                color: #03DAC6;
                padding: 8px;
                border: 1px solid #444444;
                font-weight: bold;
            }
            QProgressBar {
                border: 2px solid #444444;
                border-radius: 5px;
                text-align: center;
                background-color: #2D2D2D;
            }
            QProgressBar::chunk {
                background-color: #03DAC6;
                border-radius: 3px;
            }
        """)
        
        self.create_menu_bar()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        self.tab_widget = QTabWidget()
        
        self.dashboard = EmailDashboard()
        self.tab_widget.addTab(self.dashboard, "📊 Dashboard")
        
        manual_tab = self.create_manual_analysis_tab()
        self.tab_widget.addTab(manual_tab, "🔍 Manual Analysis")
        
        management_tab = self.create_email_management_tab()
        self.tab_widget.addTab(management_tab, "📧 Email Management")
        
        quarantine_tab = self.create_quarantine_tab()
        self.tab_widget.addTab(quarantine_tab, "🔒 Quarantine")
        
        layout.addWidget(self.tab_widget)
        
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready - Configure Gmail settings to start monitoring")
        self.setStatusBar(self.status_bar)
        
        # Progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        self.center_on_screen()
    
    def create_control_panel(self) -> QWidget:
        """Create control panel with monitoring controls"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        panel.setStyleSheet("""
            QFrame {
                background-color: #1E1E1E;
                border: 2px solid #444444;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        """)
        
        layout = QHBoxLayout()
        
        title = QLabel("🤖 Gmail Email Monitor")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #BB86FC;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setFont(QFont("Arial", 16))
        self.status_indicator.setStyleSheet("color: #888888;")
        layout.addWidget(self.status_indicator)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #E0E0E0; margin-right: 20px;")
        layout.addWidget(self.status_label)
        
        # Add Start/Stop monitoring buttons
        self.start_monitor_btn = QPushButton("▶️ Start Monitoring")
        self.start_monitor_btn.clicked.connect(self.start_monitoring)
        self.start_monitor_btn.setStyleSheet("QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #66BB6A; }")
        layout.addWidget(self.start_monitor_btn)
        
        self.stop_monitor_btn = QPushButton("⏹️ Stop Monitoring")
        self.stop_monitor_btn.clicked.connect(self.stop_monitoring)
        self.stop_monitor_btn.setStyleSheet("QPushButton { background-color: #FF9800; } QPushButton:hover { background-color: #FFB74D; }")
        self.stop_monitor_btn.setEnabled(False)
        layout.addWidget(self.stop_monitor_btn)
        
        self.fetch_btn = QPushButton("📥 Fetch Now")
        self.fetch_btn.clicked.connect(self.fetch_emails_once)
        self.fetch_btn.setStyleSheet("QPushButton { background-color: #2196F3; } QPushButton:hover { background-color: #64B5F6; }")
        layout.addWidget(self.fetch_btn)
        
        self.config_btn = QPushButton("⚙️ Configure")
        self.config_btn.clicked.connect(self.show_configuration)
        layout.addWidget(self.config_btn)
        
        panel.setLayout(layout)
        return panel

    def create_manual_analysis_tab(self) -> QWidget:
        """Create manual analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        input_group = QGroupBox("📧 Email Content Analysis")
        input_layout = QVBoxLayout()
        
        self.manual_text_edit = QTextEdit()
        self.manual_text_edit.setPlaceholderText("Paste email content here for manual analysis...")
        self.manual_text_edit.setMinimumHeight(200)
        input_layout.addWidget(self.manual_text_edit)
        
        button_layout = QHBoxLayout()
        
        analyze_btn = QPushButton("🔍 Analyze Email")
        analyze_btn.clicked.connect(self.analyze_manual_email)
        button_layout.addWidget(analyze_btn)
        
        upload_btn = QPushButton("📂 Upload File")
        upload_btn.clicked.connect(self.upload_email_file)
        button_layout.addWidget(upload_btn)
        
        clear_btn = QPushButton("🧹 Clear")
        clear_btn.clicked.connect(self.manual_text_edit.clear)
        button_layout.addWidget(clear_btn)
        
        button_layout.addStretch()
        input_layout.addLayout(button_layout)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Results
        results_group = QGroupBox("📊 Analysis Results")
        results_layout = QVBoxLayout()
        
        self.manual_result_label = QLabel("Ready for analysis...")
        self.manual_result_label.setAlignment(Qt.AlignCenter)
        self.manual_result_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.manual_result_label.setStyleSheet("padding: 20px; background-color: #2D2D2D; border-radius: 8px;")
        results_layout.addWidget(self.manual_result_label)
        
        # Explanation button (initially hidden)
        self.explain_btn = QPushButton("🔍 Show AI Explanation")
        self.explain_btn.clicked.connect(self.show_manual_explanation)
        self.explain_btn.setVisible(False)
        results_layout.addWidget(self.explain_btn)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        widget.setLayout(layout)
        return widget

    def create_email_management_tab(self) -> QWidget:
        """Create email management tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.refresh_email_list)
        controls_layout.addWidget(refresh_btn)
        
        controls_layout.addStretch()
        
        search_input = QLineEdit()
        search_input.setPlaceholderText("Search emails...")
        controls_layout.addWidget(search_input)
        
        layout.addLayout(controls_layout)
        
        # Email table
        self.email_table = QTableWidget()
        self.email_table.setColumnCount(8)
        self.email_table.setHorizontalHeaderLabels([
            'Date', 'Sender', 'Subject', 'Label', 'Classification', 'Confidence', 'Status', 'Actions'
        ])
        
        header = self.email_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        
        layout.addWidget(self.email_table)
        
        widget.setLayout(layout)
        return widget

    def create_quarantine_tab(self) -> QWidget:
        """Create quarantine management tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Warning header
        warning_frame = QFrame()
        warning_frame.setStyleSheet("""
            QFrame {
                background-color: #FF5252;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        """)
        warning_layout = QHBoxLayout()
        warning_icon = QLabel("⚠️")
        warning_icon.setFont(QFont("Arial", 16))
        warning_text = QLabel("QUARANTINE ZONE - Potentially malicious emails are isolated here")
        warning_text.setFont(QFont("Arial", 12, QFont.Bold))
        warning_text.setStyleSheet("color: white;")
        warning_layout.addWidget(warning_icon)
        warning_layout.addWidget(warning_text)
        warning_layout.addStretch()
        warning_frame.setLayout(warning_layout)
        layout.addWidget(warning_frame)
        
        # Controls
        quarantine_controls = QHBoxLayout()
        
        release_btn = QPushButton("✅ Release Selected")
        release_btn.clicked.connect(self.release_from_quarantine)
        release_btn.setStyleSheet("QPushButton { background-color: #4CAF50; } QPushButton:hover { background-color: #66BB6A; }")
        quarantine_controls.addWidget(release_btn)
        
        delete_btn = QPushButton("🗑️ Permanently Delete")
        delete_btn.clicked.connect(self.delete_quarantined)
        delete_btn.setStyleSheet("QPushButton { background-color: #F44336; } QPushButton:hover { background-color: #E57373; }")
        quarantine_controls.addWidget(delete_btn)
        
        quarantine_controls.addStretch()
        
        layout.addLayout(quarantine_controls)
        
        # Quarantine table
        self.quarantine_table = QTableWidget()
        self.quarantine_table.setColumnCount(6)
        self.quarantine_table.setHorizontalHeaderLabels([
            'Quarantined Date', 'Sender', 'Subject', 'Risk Score', 'Reason', 'Actions'
        ])
        
        self.quarantine_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        layout.addWidget(self.quarantine_table)
        
        widget.setLayout(layout)
        return widget

    def create_menu_bar(self):
        """Create application menu bar with monitoring options"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        export_action = file_menu.addAction('📤 Export Reports')
        export_action.triggered.connect(self.export_reports)
        export_action.setShortcut('Ctrl+E')
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('❌ Exit')
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut('Ctrl+Q')
        
        # Gmail menu
        gmail_menu = menubar.addMenu('Gmail')
        
        start_monitor_action = gmail_menu.addAction('▶️ Start Monitoring')
        start_monitor_action.triggered.connect(self.start_monitoring)
        start_monitor_action.setShortcut('Ctrl+M')
        
        stop_monitor_action = gmail_menu.addAction('⏹️ Stop Monitoring')
        stop_monitor_action.triggered.connect(self.stop_monitoring)
        stop_monitor_action.setShortcut('Ctrl+Shift+M')
        
        fetch_action = gmail_menu.addAction('📥 Fetch Once')
        fetch_action.triggered.connect(self.fetch_emails_once)
        fetch_action.setShortcut('Ctrl+F')
        
        gmail_menu.addSeparator()
        
        config_action = gmail_menu.addAction('⚙️ Configure')
        config_action.triggered.connect(self.show_configuration)
        config_action.setShortcut('Ctrl+,')
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = help_menu.addAction('ℹ️ About')
        about_action.triggered.connect(self.show_about)
        
        help_action = help_menu.addAction('❓ Help')
        help_action.triggered.connect(self.show_help)

    def setup_system_tray(self):
        """Setup system tray icon"""
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = QSystemTrayIcon(self)
            
            tray_menu = QMenu()
            
            show_action = tray_menu.addAction('Show SECUROX')
            show_action.triggered.connect(self.show)
            
            tray_menu.addSeparator()
            
            start_monitor_action = tray_menu.addAction('Start Monitoring')
            start_monitor_action.triggered.connect(self.start_monitoring)
            
            stop_monitor_action = tray_menu.addAction('Stop Monitoring')
            stop_monitor_action.triggered.connect(self.stop_monitoring)
            
            fetch_action = tray_menu.addAction('Fetch Once')
            fetch_action.triggered.connect(self.fetch_emails_once)
            
            tray_menu.addSeparator()
            
            quit_action = tray_menu.addAction('Quit')
            quit_action.triggered.connect(self.close)
            
            self.tray_icon.setContextMenu(tray_menu)
            self.tray_icon.setToolTip('SECUROX Email Security')
            
            self.tray_icon.show()

    def center_on_screen(self):
        """Center window on screen"""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(x, y)

    def load_config(self):
        """Load configuration from file"""
        config_file = Path("securox_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self.config = {}
        else:
            self.config = {}

    def save_config(self):
        """Save configuration to file"""
        try:
            with open("securox_config.json", 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def show_configuration(self):
        """Show configuration dialog"""
        dialog = ConfigurationDialog(self, self.config)
        if dialog.exec() == QDialog.Accepted:
            self.config = dialog.get_config()
            self.save_config()
            self.status_bar.showMessage("Configuration updated")

    def start_monitoring(self):
        """Start continuous monitoring"""
        if not self.config.get('email') or not self.config.get('app_password'):
            QMessageBox.warning(
                self, 
                "Configuration Required", 
                "Please configure your Gmail credentials first."
            )
            self.show_configuration()
            return
        
        if self.fetch_worker and self.fetch_worker.isRunning():
            QMessageBox.information(self, "Already Running", "Monitoring is already active.")
            return
        
        # Create and start worker thread
        self.fetch_worker = GmailFetchWorker(self.config)
        self.fetch_worker.email_fetched.connect(self.handle_new_email)
        self.fetch_worker.status_updated.connect(self.update_status)
        self.fetch_worker.progress_updated.connect(self.update_progress)
        self.fetch_worker.error_occurred.connect(self.handle_error)
        self.fetch_worker.batch_completed.connect(self.handle_batch_completion)
        
        self.fetch_worker.start()
        
        # Update UI
        self.start_monitor_btn.setEnabled(False)
        self.stop_monitor_btn.setEnabled(True)
        self.fetch_btn.setEnabled(False)
        self.status_indicator.setStyleSheet("color: #4CAF50;")
        self.status_label.setText("Monitoring...")
        self.status_bar.showMessage("Started monitoring Gmail for new emails")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if self.fetch_worker and self.fetch_worker.isRunning():
            self.fetch_worker.stop()
            self.fetch_worker.wait(5000)
        
        # Update UI
        self.start_monitor_btn.setEnabled(True)
        self.stop_monitor_btn.setEnabled(False)
        self.fetch_btn.setEnabled(True)
        self.status_indicator.setStyleSheet("color: #888888;")
        self.status_label.setText("Ready")
        self.status_bar.showMessage("Monitoring stopped")

    def fetch_emails_once(self):
        """Fetch emails once (original functionality)"""
        if not self.config.get('email') or not self.config.get('app_password'):
            QMessageBox.warning(
                self, 
                "Configuration Required", 
                "Please configure your Gmail credentials first."
            )
            self.show_configuration()
            return
        
        if self.fetch_worker and self.fetch_worker.isRunning():
            QMessageBox.information(self, "Operation in Progress", "Please stop monitoring first.")
            return
        
        # Use a temporary worker for one-time fetch
        temp_worker = GmailFetchWorker(self.config)
        temp_worker.email_fetched.connect(self.handle_new_email)
        temp_worker.status_updated.connect(self.update_status)
        temp_worker.progress_updated.connect(self.update_progress)
        temp_worker.error_occurred.connect(self.handle_error)
        temp_worker.batch_completed.connect(self.handle_batch_completion)
        
        # Run one fetch cycle
        temp_worker._fetch_and_analyze_new_emails()

    def fetch_emails(self):
        """Legacy method - now uses fetch_emails_once"""
        self.fetch_emails_once()

    def handle_new_email(self, email: EmailItem):
        """Handle new email from fetch worker"""
        self.emails.append(email)
        
        # Update dashboard statistics
        emails_list = list(self.emails)
        stats = {
            'total_processed': len(emails_list),
            'phishing_detected': sum(1 for e in emails_list if e.classification == 'Phishing'),
            'spam_detected': sum(1 for e in emails_list if e.classification == 'Spam'),
            'legitimate': sum(1 for e in emails_list if e.classification == 'Legit'),
            'quarantined': sum(1 for e in emails_list if e.status == 'Quarantined')
        }
        
        # Update dashboard with new stats
        self.dashboard.update_stats(stats)
        
        # Update dashboard activity list
        self.dashboard.add_activity(email)
        
        # Add to email management table
        self.add_email_to_table(email)
        while self.email_table.rowCount() > 30:
            self.email_table.removeRow(0)
        
        # If quarantined, add to quarantine table
        if email.status == 'Quarantined':
            self.add_to_quarantine(email)
            if self.config.get('notifications', True) and self.tray_icon:
                self.tray_icon.showMessage(
                    'Phishing Email Detected!',
                    f'High-risk email from {email.sender} has been quarantined',
                    QSystemTrayIcon.Warning,
                    5000
                )

    def update_status(self, status: str):
        """Update status message"""
        self.status_bar.showMessage(status)
        self.status_label.setText(status)

    def update_progress(self, value: int):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def handle_error(self, error: str):
        """Handle errors"""
        logger.error(f"Error: {error}")
        self.status_bar.showMessage(f"Error: {error}")
        self.start_monitor_btn.setEnabled(True)
        self.stop_monitor_btn.setEnabled(False)
        self.fetch_btn.setEnabled(True)
        self.status_indicator.setStyleSheet("color: #F44336;")
        self.status_label.setText("Error")
        self.progress_bar.setVisible(False)
        
        if self.config.get('notifications', True) and self.tray_icon:
            self.tray_icon.showMessage(
                'Fetch Error',
                error,
                QSystemTrayIcon.Critical,
                5000
            )

    def handle_batch_completion(self, batch_info: Dict[str, Any]):
        """Handle completion of email batch"""
        count = batch_info.get('count', 0)
        if batch_info.get('new_emails'):
            self.status_bar.showMessage(f"Found and analyzed {count} new emails")
        else:
            self.status_bar.showMessage("No new emails found")
        
        if self.config.get('notifications', True) and self.tray_icon and batch_info.get('new_emails'):
            self.tray_icon.showMessage(
                'New Emails Processed',
                f'Analyzed {count} new emails',
                QSystemTrayIcon.Information,
                3000
            )

    def add_email_to_table(self, email: EmailItem):
        """Add email to management table"""
        row = self.email_table.rowCount()
        self.email_table.insertRow(row)
        
        date_item = QTableWidgetItem(email.date.strftime('%Y-%m-%d %H:%M'))
        self.email_table.setItem(row, 0, date_item)
        
        sender_item = QTableWidgetItem(email.sender)
        self.email_table.setItem(row, 1, sender_item)
        
        subject_item = QTableWidgetItem(email.subject)
        self.email_table.setItem(row, 2, subject_item)
        
        # Label (colored dot)
        label_item = QTableWidgetItem('●')
        colors = {'Phishing': '#F44336', 'Spam': '#FF9800', 'Legit': '#4CAF50'}
        label_color = QColor(colors.get(email.classification, '#CCCCCC'))
        label_item.setForeground(label_color)
        label_item.setTextAlignment(Qt.AlignCenter)
        self.email_table.setItem(row, 3, label_item)
        
        # Classification
        class_item = QTableWidgetItem(email.classification)
        class_item.setForeground(label_color)
        self.email_table.setItem(row, 4, class_item)
        
        # Confidence
        confidence_item = QTableWidgetItem(f"{email.confidence:.1f}%")
        self.email_table.setItem(row, 5, confidence_item)
        
        # Status
        status_item = QTableWidgetItem(email.status)
        self.email_table.setItem(row, 6, status_item)
        
        # Actions - View button
        view_btn = QPushButton("View")
        view_btn.email_item = email
        view_btn.clicked.connect(self.view_email_details)
        self.email_table.setCellWidget(row, 7, view_btn)

    def view_email_details(self):
        """Show email details dialog"""
        btn = self.sender()
        if not hasattr(btn, 'email_item'):
            return
        email = btn.email_item
        
        dlg = QDialog(self)
        dlg.setWindowTitle("Email Details")
        dlg.setMinimumSize(700, 500)
        layout = QVBoxLayout()
        
        header = QLabel(f"<b>From:</b> {email.sender}<br>"
                        f"<b>Subject:</b> {email.subject}<br>"
                        f"<b>Date:</b> {email.date.strftime('%Y-%m-%d %H:%M')}<br>"
                        f"<b>Classification:</b> <span style='color:{LABEL_MAP[0][1] if email.classification == 'Legit' else LABEL_MAP[1][1] if email.classification == 'Spam' else LABEL_MAP[2][1]}'>{email.classification}</span> ({email.confidence:.1f}%)")
        header.setTextFormat(Qt.RichText)
        layout.addWidget(header)
        
        body_group = QGroupBox("Message Body")
        body_layout = QVBoxLayout()
        body_text = QTextEdit()
        body_text.setReadOnly(True)
        body_text.setPlainText(email.body or "(no body)")
        body_layout.addWidget(body_text)
        body_group.setLayout(body_layout)
        layout.addWidget(body_group)
        
        if email.attachments:
            att_group = QGroupBox("Attachments")
            att_layout = QVBoxLayout()
            for att in email.attachments:
                att_layout.addWidget(QLabel(str(att)))
            att_group.setLayout(att_layout)
            layout.addWidget(att_group)
        
        if email.file_path:
            path_label = QLabel(f"<b>Saved file:</b> {email.file_path}")
            path_label.setTextFormat(Qt.RichText)
            layout.addWidget(path_label)
        
        # Add explanation button for phishing emails
        if email.classification == "Phishing":
            explain_btn = QPushButton("🔍 Show AI Explanation")
            explain_btn.clicked.connect(lambda: self.show_explanation_dialog(email))
            layout.addWidget(explain_btn)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        layout.addWidget(btns)
        
        dlg.setLayout(layout)
        dlg.exec()

    def show_explanation_dialog(self, email: EmailItem):
        """Show explanation dialog for an email"""
        if not email.explanation:
            # Generate explanation if not already available
            text = f"{email.subject} {email.body}"
            email.explanation = explain_prediction(text, email.classification)
        
        dialog = QDialog(self)
        dialog.setWindowTitle("AI Explanation")
        dialog.setMinimumSize(500, 600)
        layout = QVBoxLayout()
        
        # Explanation content similar to ExplanationPanel
        header = QLabel("🔍 AI Explanation")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #03DAC6; padding: 10px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        if email.explanation:
            explanation_data = email.explanation
            features = explanation_data.get('explanation', [])
            
            # Features list
            features_group = QGroupBox("📊 Key Features Contributing to Classification")
            features_layout = QVBoxLayout()
            
            features_text = QTextEdit()
            features_text.setReadOnly(True)
            
            features_html = "<table width='100%'>"
            for feature, weight in features:
                color = "#F44336" if weight > 0 else "#4CAF50"
                features_html += f"<tr><td>{feature}</td><td style='color: {color}; text-align: right;'>{weight:+.3f}</td></tr>"
            features_html += "</table>"
            
            features_text.setHtml(features_html)
            features_layout.addWidget(features_text)
            features_group.setLayout(features_layout)
            layout.addWidget(features_group)
            
            # Risk factors for phishing
            if email.classification == "Phishing":
                risk_group = QGroupBox("⚠️ Risk Factors Identified")
                risk_layout = QVBoxLayout()
                
                risk_text = QTextEdit()
                risk_text.setReadOnly(True)
                
                risk_factors = []
                for feature, weight in features:
                    if weight > 0:
                        risk_factors.append(f"• {feature} (contribution: {weight:+.3f})")
                
                risk_text.setText("\n".join(risk_factors) if risk_factors else "No strong risk factors identified.")
                risk_layout.addWidget(risk_text)
                risk_group.setLayout(risk_layout)
                layout.addWidget(risk_group)
        
        else:
            no_exp_label = QLabel("No detailed explanation available.")
            no_exp_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(no_exp_label)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()

    def add_to_quarantine(self, email: EmailItem):
        """Add email to quarantine table"""
        row = self.quarantine_table.rowCount()
        self.quarantine_table.insertRow(row)
        
        date_item = QTableWidgetItem(datetime.now().strftime('%Y-%m-%d %H:%M'))
        self.quarantine_table.setItem(row, 0, date_item)
        
        sender_item = QTableWidgetItem(email.sender)
        self.quarantine_table.setItem(row, 1, sender_item)
        
        subject_item = QTableWidgetItem(email.subject)
        self.quarantine_table.setItem(row, 2, subject_item)
        
        risk_item = QTableWidgetItem(f"{email.risk_score:.1f}")
        risk_item.setForeground(QColor('#F44336'))
        self.quarantine_table.setItem(row, 3, risk_item)
        
        reason_item = QTableWidgetItem(f"Classified as {email.classification}")
        self.quarantine_table.setItem(row, 4, reason_item)
        
        actions_item = QTableWidgetItem("Review")
        self.quarantine_table.setItem(row, 5, actions_item)

    def analyze_manual_email(self):
        """Analyze manually input email with explanation"""
        text = self.manual_text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "No Content", "Please enter email content to analyze.")
            return
        
        try:
            label, confidence, color = classify_email(text)
            
            # Store the current analysis for explanation
            self.current_manual_analysis = {
                'text': text,
                'label': label,
                'confidence': confidence,
                'explanation': explain_prediction(text, label) if LIME_AVAILABLE else None
            }
            
            result_text = f"Classification: {label} ({confidence:.1f}% confidence)"
            
            # Show explanation button for phishing emails
            if label == "Phishing" and LIME_AVAILABLE:
                self.explain_btn.setVisible(True)
                result_text += "\n\n🔍 Explanation available - click 'Show AI Explanation'"
            else:
                self.explain_btn.setVisible(False)
            
            self.manual_result_label.setText(result_text)
            self.manual_result_label.setStyleSheet(f"""
                padding: 20px; 
                background-color: #2D2D2D; 
                border-radius: 8px;
                color: {color};
                border: 2px solid {color};
            """)
            
        except Exception as e:
            logger.error(f"Manual classification failed: {e}")
            label, confidence, color = ("Legit", 60.0, LABEL_MAP[0][1])
            self.manual_result_label.setText(f"Classification: {label} ({confidence:.1f}% confidence)")
            self.manual_result_label.setStyleSheet(f"""
                padding: 20px; 
                background-color: #2D2D2D; 
                border-radius: 8px;
                color: {color};
                border: 2px solid {color};
            """)
            self.explain_btn.setVisible(False)

    def show_manual_explanation(self):
        """Show explanation for manual analysis"""
        if not hasattr(self, 'current_manual_analysis'):
            QMessageBox.information(self, "No Analysis", "Please analyze an email first.")
            return
        
        analysis = self.current_manual_analysis
        dialog = QDialog(self)
        dialog.setWindowTitle("AI Explanation")
        dialog.setMinimumSize(500, 600)
        layout = QVBoxLayout()
        
        # Explanation content
        header = QLabel("🔍 AI Explanation")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #03DAC6; padding: 10px;")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        if analysis['explanation']:
            explanation_data = analysis['explanation']
            features = explanation_data.get('explanation', [])
            
            # Features list
            features_group = QGroupBox("📊 Key Features Contributing to Classification")
            features_layout = QVBoxLayout()
            
            features_text = QTextEdit()
            features_text.setReadOnly(True)
            
            features_html = "<table width='100%'>"
            for feature, weight in features:
                color = "#F44336" if weight > 0 else "#4CAF50"
                features_html += f"<tr><td>{feature}</td><td style='color: {color}; text-align: right;'>{weight:+.3f}</td></tr>"
            features_html += "</table>"
            
            features_text.setHtml(features_html)
            features_layout.addWidget(features_text)
            features_group.setLayout(features_layout)
            layout.addWidget(features_group)
            
            # Risk factors for phishing
            if analysis['label'] == "Phishing":
                risk_group = QGroupBox("⚠️ Risk Factors Identified")
                risk_layout = QVBoxLayout()
                
                risk_text = QTextEdit()
                risk_text.setReadOnly(True)
                
                risk_factors = []
                for feature, weight in features:
                    if weight > 0:
                        risk_factors.append(f"• {feature} (contribution: {weight:+.3f})")
                
                risk_text.setText("\n".join(risk_factors) if risk_factors else "No strong risk factors identified.")
                risk_layout.addWidget(risk_text)
                risk_group.setLayout(risk_layout)
                layout.addWidget(risk_group)
        
        else:
            no_exp_label = QLabel("No detailed explanation available.\nLIME may not be installed or there was an error generating the explanation.")
            no_exp_label.setAlignment(Qt.AlignCenter)
            no_exp_label.setWordWrap(True)
            layout.addWidget(no_exp_label)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()

    def upload_email_file(self):
        """Upload email file for manual analysis"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Upload Email File", "", "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    self.manual_text_edit.setPlainText(content)
                
                QTimer.singleShot(500, self.analyze_manual_email)
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not read file:\n{str(e)}")

    def refresh_email_list(self):
        """Refresh email list"""
        self.status_bar.showMessage("Email list refreshed")

    def release_from_quarantine(self):
        """Release selected emails from quarantine"""
        selected_rows = set(item.row() for item in self.quarantine_table.selectedItems())
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select emails to release.")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Release", 
            f"Release {len(selected_rows)} email(s) from quarantine?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for row in sorted(selected_rows, reverse=True):
                self.quarantine_table.removeRow(row)
            
            self.status_bar.showMessage(f"Released {len(selected_rows)} email(s)")

    def delete_quarantined(self):
        """Permanently delete quarantined emails"""
        selected_rows = set(item.row() for item in self.quarantine_table.selectedItems())
        if not selected_rows:
            QMessageBox.information(self, "No Selection", "Please select emails to delete.")
            return
        
        reply = QMessageBox.warning(
            self, "Confirm Deletion", 
            f"Permanently delete {len(selected_rows)} email(s)?\nThis action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for row in sorted(selected_rows, reverse=True):
                self.quarantine_table.removeRow(row)
            
            self.status_bar.showMessage(f"Deleted {len(selected_rows)} email(s)")

    def export_reports(self):
        """Export analysis reports"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", f"securox_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;All Files (*.*)"
        )
        
        if file_path:
            try:
                report_data = {
                    'generated': datetime.now().isoformat(),
                    'total_emails': len(self.emails),
                    'statistics': {
                        'phishing': sum(1 for e in self.emails if e.classification == 'Phishing'),
                        'spam': sum(1 for e in self.emails if e.classification == 'Spam'),
                        'legitimate': sum(1 for e in self.emails if e.classification == 'Legit')
                    },
                    'emails': [
                        {
                            'date': e.date.isoformat(),
                            'sender': e.sender,
                            'subject': e.subject,
                            'classification': e.classification,
                            'confidence': e.confidence,
                            'status': e.status
                        }
                        for e in self.emails
                    ]
                }
                
                with open(file_path, 'w') as f:
                    json.dump(report_data, f, indent=4)
                
                self.status_bar.showMessage(f"Report exported to {file_path}")
                
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Could not export report:\n{str(e)}")

    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>🔒 SECUROX Email Security System</h2>
        <p><b>Version 3.0 - Real Gmail Integration</b></p>
        
        <p>Advanced AI-powered email security system with:</p>
        <ul>
            <li>Real Gmail email fetching via IMAP</li>
            <li>DistilBERT-powered phishing detection</li>
            <li>LIME Explainable AI for transparency</li>
            <li>Smart quarantine system</li>
            <li>Comprehensive analytics dashboard</li>
            <li>Manual email analysis tools</li>
            <li>Continuous email monitoring</li>
        </ul>
        
        <p><b>Technologies:</b> PySide6, DistilBERT, PyTorch, LIME, IMAP</p>
        <p><b>Author:</b> SECUROX Development Team</p>
        """
        
        QMessageBox.about(self, "About SECUROX", about_text)

    def show_help(self):
        """Show help dialog"""
        help_text = """
        <h3>🆘 SECUROX Help</h3>
        
        <h4>Getting Started:</h4>
        <ol>
            <li>Click "Configure" to set up Gmail credentials</li>
            <li>Use a Gmail App Password (not your regular password)</li>
            <li>Click "Start Monitoring" for continuous protection</li>
            <li>Or use "Fetch Now" for one-time analysis</li>
        </ol>
        
        <h4>Features:</h4>
        <ul>
            <li><b>Continuous Monitoring:</b> Automatically analyzes new emails as they arrive</li>
            <li><b>Dashboard:</b> View real-time statistics with AI explanations</li>
            <li><b>Manual Analysis:</b> Analyze individual emails with explanations</li>
            <li><b>Email Management:</b> Review all fetched emails</li>
            <li><b>Quarantine:</b> Manage suspicious emails</li>
        </ul>
        
        <h4>AI Explanations:</h4>
        <ul>
            <li>Double-click phishing emails in Dashboard to see explanations</li>
            <li>View feature importance and risk factors</li>
            <li>Understand why emails are classified as phishing</li>
        </ul>
        
        <h4>Keyboard Shortcuts:</h4>
        <ul>
            <li>Ctrl+M: Start Monitoring</li>
            <li>Ctrl+Shift+M: Stop Monitoring</li>
            <li>Ctrl+F: Fetch Emails Once</li>
            <li>Ctrl+,: Configuration</li>
            <li>Ctrl+E: Export Reports</li>
            <li>Ctrl+Q: Quit</li>
        </ul>
        
        <h4>Important Notes:</h4>
        <ul>
            <li>Continuously monitors your Gmail inbox for new emails</li>
            <li>Optionally marks fetched emails as read</li>
            <li>Uses DistilBERT model for accurate classification</li>
            <li>LIME provides transparent AI explanations</li>
            <li>All emails are saved locally for analysis</li>
            <li>Configure monitoring interval in settings (default: 60 seconds)</li>
        </ul>
        """
        
        msg = QMessageBox()
        msg.setWindowTitle("Help")
        msg.setText(help_text)
        msg.setTextFormat(Qt.RichText)
        msg.exec()

    def closeEvent(self, event):
        """Handle application close event"""
        if self.fetch_worker and self.fetch_worker.isRunning():
            reply = QMessageBox.question(
                self, "Monitoring in Progress",
                "Email monitoring is active. Stop and exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.fetch_worker.stop()
                self.fetch_worker.wait(5000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("SECUROX")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("SecurOx Security")
    
    # Set dark theme palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(18, 18, 18))
    palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(68, 68, 68))
    palette.setColor(QPalette.ToolTipBase, QColor(224, 224, 224))
    palette.setColor(QPalette.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.Button, QColor(68, 68, 68))
    palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    # Create and show main window
    window = SecuroxMainWindow()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())