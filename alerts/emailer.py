from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText


def send_email(subject: str, body: str) -> bool:
    host = (os.getenv("SMTP_HOST") or "").strip()
    port = int((os.getenv("SMTP_PORT") or "587").strip())
    user = (os.getenv("SMTP_USER") or "").strip()
    pwd = (os.getenv("SMTP_PASS") or "").strip()
    to_email = (os.getenv("ALERT_TO_EMAIL") or "").strip()

    if not (host and user and pwd and to_email):
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_email

    with smtplib.SMTP(host, port, timeout=20) as s:
        s.starttls()
        s.login(user, pwd)
        s.sendmail(user, [to_email], msg.as_string())

    return True
