import base64
import streamlit as st
import joblib

from bs4 import BeautifulSoup
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from streamlit_oauth import OAuth2Component


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

MODEL_PATH = "spam_detection_model.pkl"

CLIENT_ID = st.secrets["auth"]["client_id"]
CLIENT_SECRET = st.secrets["auth"]["client_secret"]

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"

REDIRECT_URI = (
    "https://smartemailspamdetection.streamlit.app/"
    "component/streamlit_oauth.authorize_button"
)

SCOPE = (
    "openid email profile "
    "https://www.googleapis.com/auth/gmail.readonly"
)


# --------------------------------------------------
# OAUTH COMPONENT
# --------------------------------------------------

oauth2 = OAuth2Component(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    authorize_endpoint=AUTHORIZE_URL,
    token_endpoint=TOKEN_URL
)


# --------------------------------------------------
# EMAIL BODY EXTRACTION
# --------------------------------------------------

def get_email_body(payload):

    body = ""

    # Email contains multiple parts
    if "parts" in payload:

        for part in payload["parts"]:

            body += get_email_body(part)

    else:

        mime_type = payload.get("mimeType", "")
        data = payload.get("body", {}).get("data")

        if data:

            try:

                decoded = base64.urlsafe_b64decode(
                    data
                ).decode(
                    "utf-8",
                    errors="ignore"
                )

                # Plain text email
                if mime_type == "text/plain":

                    body += decoded

                # HTML email
                elif mime_type == "text/html":

                    soup = BeautifulSoup(
                        decoded,
                        "html.parser"
                    )

                    body += soup.get_text(
                        separator=" ",
                        strip=True
                    )

            except Exception:
                pass

    return body


# --------------------------------------------------
# GMAIL SERVICE
# --------------------------------------------------

def get_gmail_service(token):

    access_token = token.get("access_token")

    credentials = Credentials(
        token=access_token,
        refresh_token=token.get("refresh_token"),
        token_uri=TOKEN_URL,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    return build(
        "gmail",
        "v1",
        credentials=credentials,
        cache_discovery=False
    )


# --------------------------------------------------
# GET ALL GMAIL MESSAGES
# --------------------------------------------------

def get_all_messages(service):

    all_messages = []
    page_token = None

    while True:

        response = service.users().messages().list(
            userId="me",
            q="-in:trash",
            maxResults=500,
            pageToken=page_token,
            includeSpamTrash=True
        ).execute()

        all_messages.extend(
            response.get("messages", [])
        )

        page_token = response.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return all_messages


# --------------------------------------------------
# GET HEADER VALUE
# --------------------------------------------------

def get_header(headers, header_name):

    for header in headers:

        if header.get("name", "").lower() == header_name.lower():

            return header.get(
                "value",
                ""
            )

    return ""


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

st.title("📧 Email Spam Detection")

st.write(
    "Connect your Gmail account and classify your emails as "
    "SPAM or NOT SPAM."
)


# --------------------------------------------------
# LOGIN
# --------------------------------------------------

if "token" not in st.session_state:

    st.subheader("Connect Gmail")

    st.write(
        "Sign in with Google and give permission to read your Gmail "
        "messages for spam classification."
    )

    result = oauth2.authorize_button(
        name="🔗 Connect Gmail",
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        key="google_login",
        extras_params={
            "response_type": "code",
            "access_type": "offline",
            "prompt": "consent"
        }
    )

    if result and "token" in result:

        st.session_state["token"] = result["token"]

        st.rerun()

    st.stop()


# --------------------------------------------------
# CONNECTED USER
# --------------------------------------------------

st.success(
    "✅ Gmail connected successfully!"
)

st.write(
    "You can now classify your Gmail messages."
)


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

if st.button("Logout"):

    del st.session_state["token"]

    st.rerun()


# --------------------------------------------------
# DETECT SPAM
# --------------------------------------------------

if st.button("🔍 Detect Spam in All Gmail Emails"):

    try:

        # Load trained machine-learning model
        model = joblib.load(
            MODEL_PATH
        )

        # Connect to Gmail
        service = get_gmail_service(
            st.session_state["token"]
        )

        st.info(
            "Getting your Gmail messages..."
        )

        # Get Gmail messages
        messages = get_all_messages(
            service
        )

        total = len(messages)

        if total == 0:

            st.warning(
                "No Gmail messages were found."
            )

            st.stop()

        st.success(
            f"Found {total} Gmail messages."
        )

        spam_count = 0
        not_spam_count = 0

        progress = st.progress(0)

        # --------------------------------------------------
        # PROCESS EACH EMAIL
        # --------------------------------------------------

        for index, message in enumerate(messages):

            email_data = service.users().messages().get(
                userId="me",
                id=message["id"],
                format="full"
            ).execute()

            payload = email_data.get(
                "payload",
                {}
            )

            headers = payload.get(
                "headers",
                []
            )

            # Get sender
            sender = get_header(
                headers,
                "from"
            )

            # Get subject
            subject = get_header(
                headers,
                "subject"
            )

            if not sender:

                sender = "Unknown Sender"

            if not subject:

                subject = "No Subject"

            # Get complete email body
            body = get_email_body(
                payload
            )

            # Combine subject + body
            email_text = (
                subject + " " + body
            ).strip()

            # If body could not be extracted,
            # use Gmail snippet as fallback
            if not body.strip():

                email_text = (
                    subject + " " +
                    email_data.get(
                        "snippet",
                        ""
                    )
                ).strip()

            # --------------------------------------------------
            # MODEL PREDICTION
            # --------------------------------------------------

            prediction = model.predict(
                [email_text]
            )[0]

            # --------------------------------------------------
            # DISPLAY EMAIL
            # --------------------------------------------------

            st.divider()

            st.write(
                "**From:**",
                sender
            )

            st.write(
                "**Subject:**",
                subject
            )

            if str(prediction).lower() == "spam":

                st.error(
                    "🚨 SPAM"
                )

                spam_count += 1

            else:

                st.success(
                    "✅ NOT SPAM"
                )

                not_spam_count += 1

            progress.progress(
                (index + 1) / total
            )

        # --------------------------------------------------
        # CLASSIFICATION SUMMARY
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Classification Summary"
        )

        st.write(
            "🚨 Spam:",
            spam_count
        )

        st.write(
            "✅ Not Spam:",
            not_spam_count
        )

        st.write(
            "📧 Total:",
            total
        )

        st.success(
            "✅ All Gmail messages have been classified!"
        )


    # --------------------------------------------------
    # GMAIL API ERROR
    # --------------------------------------------------

    except HttpError as e:

        st.error(
            f"Gmail API error: {e}"
        )


    # --------------------------------------------------
    # OTHER ERROR
    # --------------------------------------------------

    except Exception as e:

        st.error(
            f"Error: {e}"
        )