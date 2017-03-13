import smtplib

from os.path import basename

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# AWS Config
EMAIL_HOST = ''
EMAIL_HOST_USER = ''
EMAIL_HOST_PASSWORD = ''
EMAIL_PORT = 587

def send_email(from_addr, to_addr, subject, message, attachment=None):
	msg = MIMEMultipart()
	msg['Subject'] = subject

	msg['From'] = from_addr
	msg['To'] = to_addr

	mime_text = MIMEText(message)
	msg.attach(mime_text)

	if attachment:
		with open(attachment, "rb") as fil:
		    part = MIMEApplication(
		        fil.read(),
		        Name=basename(attachment)
		    )
		    part['Content-Disposition'] = 'attachment; filename="%s"' % basename(attachment)
		    msg.attach(part)

	s = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
	s.starttls()
	s.login(EMAIL_HOST_USER, EMAIL_HOST_PASSWORD)
	s.sendmail(from_addr, to_addr, msg.as_string())
	s.quit()
