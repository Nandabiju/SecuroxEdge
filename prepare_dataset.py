import os
import re
import tarfile
import zipfile
import requests
import subprocess
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# -------------------------------
# 1. Cleaning function
# -------------------------------
def clean_html(raw_html: str) -> str:
    """Remove HTML tags, URLs, emails, and extra whitespace."""
    if not isinstance(raw_html, str):
        return ""
    # Remove HTML
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")
    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Keep letters, numbers, punctuation
    text = re.sub(r"[^0-9A-Za-z\s\.,;:!?'\-]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------------
# 2. Download helper
# -------------------------------
def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"[INFO] {output_path} already exists, skipping download.")
        return
    print(f"[INFO] Downloading {url}...")
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

# -------------------------------
# 3. Extract helper
# -------------------------------
def extract_file(filepath, extract_to):
    print(f"[INFO] Extracting {filepath}...")
    if filepath.endswith(".tar.gz") or filepath.endswith(".tgz"):
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=extract_to)
    elif filepath.endswith(".tar.bz2"):
        with tarfile.open(filepath, "r:bz2") as tar:
            tar.extractall(path=extract_to)
    elif filepath.endswith(".zip"):
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

# -------------------------------
# 4. Read text files into list
# -------------------------------
def read_emails_from_dir(directory, label):
    emails = []
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                with open(os.path.join(root, file), 'r', encoding="latin1") as f:
                    raw = f.read()
                    emails.append({"text": clean_html(raw), "label": label})
            except:
                pass
    return emails

# -------------------------------
# MAIN SCRIPT
# -------------------------------
if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)

    # --- Download Enron ---
    enron_url = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
    enron_path = "../data/enron.tar.gz"
    download_file(enron_url, enron_path)
    extract_file(enron_path, "../data/enron")

    # --- Download SpamAssassin (ham & spam) ---
    spamassassin_ham_url = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"
    spamassassin_spam_url = "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2"
    spamassassin_ham_path = "../data/easy_ham.tar.bz2"
    spamassassin_spam_path = "../data/spam.tar.bz2"

    download_file(spamassassin_ham_url, spamassassin_ham_path)
    download_file(spamassassin_spam_url, spamassassin_spam_path)
    extract_file(spamassassin_ham_path, "../data/spamassassin_ham")
    extract_file(spamassassin_spam_path, "../data/spamassassin_spam")

    # --- Download phishing dataset from Kaggle ---
    print("[INFO] Downloading phishing dataset from Kaggle...")
    os.makedirs("../data/phishing", exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download", "-d", "subhajournal/phishingemails", "-p", "../data/phishing"
    ])
    # Extract phishing dataset
    for file in os.listdir("../data/phishing"):
        if file.endswith(".zip"):
            extract_file(os.path.join("../data/phishing", file), "../data/phishing")

    # Read phishing CSV
    phishing_df = pd.DataFrame()
    phishing_csv = os.path.join("../data/phishing", "phishing_emails.csv")
    if os.path.exists(phishing_csv):
        phishing_df = pd.read_csv(phishing_csv)
        if 'Email Text' in phishing_df.columns:
            phishing_df = phishing_df.rename(columns={'Email Text': 'text'})
            phishing_df['text'] = phishing_df['text'].apply(clean_html)
            phishing_df['label'] = 'phishing'
            print(f"[INFO] Loaded {len(phishing_df)} phishing emails.")

    # --- Read & label legit + spam from text files ---
    print("[INFO] Reading Enron & SpamAssassin emails...")
    all_data = []
    all_data.extend(read_emails_from_dir("../data/enron", "legit"))
    all_data.extend(read_emails_from_dir("../data/spamassassin_ham", "legit"))
    all_data.extend(read_emails_from_dir("../data/spamassassin_spam", "spam"))

    # --- Add phishing if available ---
    if not phishing_df.empty:
        all_data.extend(phishing_df.to_dict(orient="records"))

    # --- Save final CSV ---
    df = pd.DataFrame(all_data)
    print(f"[INFO] Final dataset size: {len(df)} emails.")
    df.to_csv("../data/emails.csv", index=False)
    print("[INFO] Saved to ../data/emails.csv")
