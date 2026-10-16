import base64
import html
import os
import pickle
import re

from flask import Flask, flash, jsonify, redirect, render_template_string, request, session, url_for

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    Request = None
    Credentials = None
    Flow = None
    build = None
    HttpError = Exception

    


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/gmail.readonly",
]
TOKEN_URI = "https://oauth2.googleapis.com/token"
AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
REDIRECT_PATH = "/oauth2callback"
CLIENT_SECRET_FILES = (
    os.environ.get("GOOGLE_CLIENT_SECRET_FILE", "").strip(),
    os.path.join(BASE_DIR, "client_secret.json"),
    os.path.join(BASE_DIR, "credentials.json"),
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret-key")

if os.environ.get("OAUTHLIB_INSECURE_TRANSPORT") is None:
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

MODEL = None
VECTORIZER = None
MODEL_STATUS = "Using keyword fallback because vectorizer.pkl is not available."


def load_model_assets():
    global MODEL, VECTORIZER, MODEL_STATUS

    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    model_path = os.path.join(BASE_DIR, "MNB.pkl")

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        return

    try:
        with open(vectorizer_path, "rb") as vectorizer_file:
            VECTORIZER = pickle.load(vectorizer_file)
        with open(model_path, "rb") as model_file:
            MODEL = pickle.load(model_file)
        MODEL_STATUS = "Using trained Multinomial Naive Bayes model."
    except Exception as exc:
        MODEL = None
        VECTORIZER = None
        MODEL_STATUS = f"Using keyword fallback because model assets could not be loaded: {exc}"


def fallback_predict(text):
    spam_words = {
        "win", "winner", "free", "cash", "prize", "urgent", "claim", "offer",
        "limited", "click", "congratulations", "guaranteed", "lottery", "reward",
        "credit", "loan", "bonus", "deal", "buy", "now", "call", "subscribe"
    }
    normalized = re.sub(r"[^a-zA-Z]", " ", text).lower()
    words = normalized.split()
    hits = sum(1 for word in words if word in spam_words)
    has_link = "http://" in text.lower() or "https://" in text.lower() or "www." in text.lower()
    has_money = bool(re.search(r"[$]\s*\d+|\d+\s*(rs|inr|usd|dollars?)", text.lower()))
    score = hits + int(has_link) + int(has_money)
    return {
        "label": "Spam" if score >= 2 else "Not Spam",
        "confidence": min(95, 55 + score * 10),
        "engine": "keyword fallback"
    }


def classify_text(text):
    if MODEL is not None and VECTORIZER is not None:
        transformed = VECTORIZER.transform([text])
        prediction = MODEL.predict(transformed)[0]
        label = "Spam" if int(prediction) == 1 else "Not Spam"
        confidence = 90
        if hasattr(MODEL, "predict_proba"):
            confidence = round(float(max(MODEL.predict_proba(transformed)[0])) * 100, 2)
        return {"label": label, "confidence": confidence, "engine": "trained model"}

    return fallback_predict(text)


def google_config():
    return {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", "").strip(),
    }


def client_secret_file():
    for path in CLIENT_SECRET_FILES:
        if path and os.path.exists(path):
            return path
    return None


def gmail_ready():
    config = google_config()
    has_env_credentials = config["client_id"] and config["client_secret"]
    return bool(Flow and build and (has_env_credentials or client_secret_file()))


def make_flow(state=None, code_verifier=None):
    config = google_config()
    redirect_uri = url_for("oauth2callback", _external=True)

    if client_secret_file():
        return Flow.from_client_secrets_file(
            client_secret_file(),
            scopes=SCOPES,
            redirect_uri=redirect_uri,
            state=state,
            code_verifier=code_verifier,
        )

    client_config = {
        "web": {
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "auth_uri": AUTH_URI,
            "token_uri": TOKEN_URI,
            "redirect_uris": [redirect_uri],
        }
    }
    return Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=redirect_uri,
        state=state,
        code_verifier=code_verifier,
    )


def credentials_to_dict(credentials):
    return {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }


def credentials_from_session():
    token_data = session.get("credentials")
    if not token_data or Credentials is None:
        return None

    credentials = Credentials(**token_data)
    if credentials.expired and credentials.refresh_token and Request:
        credentials.refresh(Request())
        session["credentials"] = credentials_to_dict(credentials)
    return credentials


def decode_gmail_body(payload):
    body_parts = []

    def walk(part):
        for child in part.get("parts", []):
            walk(child)

        data = part.get("body", {}).get("data")
        if not data:
            return

        try:
            decoded = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode(
                "utf-8",
                errors="ignore",
            )
        except Exception:
            return

        mime_type = part.get("mimeType", "")
        if mime_type == "text/html":
            decoded = html.unescape(re.sub(r"<[^>]+>", " ", decoded))
        if mime_type in {"text/plain", "text/html"}:
            body_parts.append(decoded)

    walk(payload)
    return " ".join(body_parts)


def header_value(headers, name):
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def get_gmail_messages(max_results=10):
    credentials = credentials_from_session()
    if not credentials:
        return None, "Please connect Gmail first."

    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
    response = service.users().messages().list(
        userId="me",
        q="-in:trash",
        maxResults=max_results,
        includeSpamTrash=True,
    ).execute()

    rows = []
    for message in response.get("messages", []):
        email_data = service.users().messages().get(
            userId="me",
            id=message["id"],
            format="full",
        ).execute()
        payload = email_data.get("payload", {})
        headers = payload.get("headers", [])
        subject = header_value(headers, "subject") or "No subject"
        sender = header_value(headers, "from") or "Unknown sender"
        body = decode_gmail_body(payload)
        email_text = f"{subject} {body or email_data.get('snippet', '')}".strip()
        prediction = classify_text(email_text)
        rows.append({
            "id": message["id"],
            "from": sender,
            "subject": subject,
            "snippet": email_data.get("snippet", ""),
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "engine": prediction["engine"],
        })

    return rows, None


load_model_assets()


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Email Spam Detection</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #eef3f8;
      color: #172033;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(135deg, rgba(23, 105, 170, 0.10), rgba(6, 118, 71, 0.08)),
        #eef3f8;
    }
    main {
      width: min(1080px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 5vw, 46px);
      line-height: 1.05;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 20px;
    }
    p {
      margin: 0;
      color: #526071;
      line-height: 1.5;
    }
    a,
    button {
      border: 0;
      border-radius: 8px;
      padding: 12px 16px;
      background: #1769aa;
      color: white;
      font: inherit;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      box-sizing: border-box;
    }
    button:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    .secondary {
      background: #e8eef6;
      color: #172033;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(280px, 380px) 1fr;
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: #ffffff;
      border: 1px solid #d8e1eb;
      border-radius: 8px;
      box-shadow: 0 18px 45px rgba(23, 32, 51, 0.10);
      padding: 20px;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }
    textarea {
      box-sizing: border-box;
      width: 100%;
      min-height: 180px;
      resize: vertical;
      border: 1px solid #c8d2e0;
      border-radius: 8px;
      padding: 14px;
      font: inherit;
      line-height: 1.45;
    }
    .result,
    .status {
      margin-top: 14px;
      padding: 14px;
      border-radius: 8px;
      border: 1px solid #dce3ee;
      background: #f8fafc;
      min-height: 48px;
    }
    .status {
      min-height: auto;
      color: #526071;
      font-size: 14px;
    }
    .label {
      font-size: 20px;
      font-weight: 800;
      margin-bottom: 4px;
    }
    .spam {
      color: #b42318;
    }
    .ham {
      color: #067647;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }
    .metric {
      border: 1px solid #dce3ee;
      border-radius: 8px;
      padding: 12px;
      background: #f8fafc;
    }
    .metric strong {
      display: block;
      font-size: 24px;
    }
    .email-list {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }
    .email-row {
      border: 1px solid #dce3ee;
      border-radius: 8px;
      padding: 14px;
      background: #ffffff;
    }
    .email-head {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }
    .subject {
      font-weight: 800;
    }
    .sender,
    .snippet,
    small {
      color: #5c6878;
    }
    .pill {
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
      background: #eef3f8;
    }
    @media (max-width: 820px) {
      header,
      .layout {
        display: block;
      }
      header .actions {
        margin-top: 14px;
      }
      .panel {
        margin-bottom: 16px;
      }
      .summary {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    {% with messages = get_flashed_messages() %}
      {% for message in messages %}
        <div class="status">{{ message }}</div>
      {% endfor %}
    {% endwith %}
    <header>
      <div>
        <h1>Email Spam Detection</h1>
        <p>Connect Gmail to automatically classify recent emails, or paste a message manually.</p>
      </div>
      <div class="actions">
        {% if connected %}
          <a class="secondary" href="/logout">Disconnect Gmail</a>
        {% elif gmail_ready %}
          <a href="/gmail/login">Connect Gmail</a>
        {% endif %}
      </div>
    </header>

    <section class="layout">
      <div class="panel">
        <h2>Manual Check</h2>
        <form id="spam-form">
          <textarea id="text" name="text" placeholder="Type or paste your email message here..." required></textarea>
          <button id="check-button" type="submit">Check Message</button>
        </form>
        <section id="result" class="result">Result will appear here.</section>
        <div class="status">{{ model_status }}</div>
      </div>

      <div class="panel">
        <h2>Gmail Auto Scan</h2>
        {% if not gmail_ready %}
          <p>Gmail is not configured yet. Add Google OAuth credentials, then restart this app.</p>
          <div class="status">
            Use GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET, or put client_secret.json / credentials.json in this folder.
          </div>
        {% elif connected %}
          <p>Gmail is connected. Recent messages will scan automatically.</p>
          <div class="actions">
            <button id="scan-button" type="button">Scan Latest Emails</button>
          </div>
          <section class="summary">
            <div class="metric"><small>Total</small><strong id="total-count">0</strong></div>
            <div class="metric"><small>Spam</small><strong id="spam-count">0</strong></div>
            <div class="metric"><small>Not Spam</small><strong id="ham-count">0</strong></div>
          </section>
          <div id="gmail-status" class="status">Starting Gmail scan...</div>
          <div id="email-list" class="email-list"></div>
        {% else %}
          <p>Sign in with Google to read your recent Gmail messages and classify them automatically.</p>
          <div class="actions">
            <a href="/gmail/login">Connect Gmail</a>
          </div>
          <div class="status">The app requests Gmail read-only access.</div>
        {% endif %}
      </div>
    </section>
  </main>

  <script>
    const form = document.getElementById("spam-form");
    const button = document.getElementById("check-button");
    const result = document.getElementById("result");
    const scanButton = document.getElementById("scan-button");
    const gmailStatus = document.getElementById("gmail-status");
    const emailList = document.getElementById("email-list");

    function escapeHtml(value) {
      return String(value || "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }

    function resultHtml(data) {
      const isSpam = data.label.toLowerCase().includes("spam") && !data.label.toLowerCase().includes("not");
      return `
        <div class="label ${isSpam ? "spam" : "ham"}">${escapeHtml(data.label)}</div>
        <div>Confidence: ${escapeHtml(data.confidence)}%</div>
        <div>Engine: ${escapeHtml(data.engine)}</div>
      `;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      result.textContent = "Checking...";

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: document.getElementById("text").value })
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Prediction failed");
        }
        result.innerHTML = resultHtml(data);
      } catch (error) {
        result.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });

    async function scanGmail() {
      if (!scanButton || !gmailStatus || !emailList) {
        return;
      }

      scanButton.disabled = true;
      gmailStatus.textContent = "Scanning Gmail messages...";
      emailList.innerHTML = "";

      try {
        const response = await fetch("/gmail/scan?max_results=10");
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Gmail scan failed");
        }

        const spamCount = data.messages.filter((message) => message.label === "Spam").length;
        const hamCount = data.messages.length - spamCount;
        document.getElementById("total-count").textContent = data.messages.length;
        document.getElementById("spam-count").textContent = spamCount;
        document.getElementById("ham-count").textContent = hamCount;

        gmailStatus.textContent = data.messages.length
          ? "Scan complete."
          : "No Gmail messages were found.";

        emailList.innerHTML = data.messages.map((message) => {
          const isSpam = message.label === "Spam";
          return `
            <article class="email-row">
              <div class="email-head">
                <div>
                  <div class="subject">${escapeHtml(message.subject)}</div>
                  <div class="sender">${escapeHtml(message.from)}</div>
                </div>
                <span class="pill ${isSpam ? "spam" : "ham"}">${escapeHtml(message.label)}</span>
              </div>
              <div class="snippet">${escapeHtml(message.snippet)}</div>
              <small>${escapeHtml(message.confidence)}% confidence - ${escapeHtml(message.engine)}</small>
            </article>
          `;
        }).join("");
      } catch (error) {
        gmailStatus.textContent = error.message;
      } finally {
        scanButton.disabled = false;
      }
    }

    if (scanButton) {
      scanButton.addEventListener("click", scanGmail);
      scanGmail();
    }
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    # Start the Google OAuth flow as soon as the app opens when credentials
    # are configured. Google still requires the user to sign in and consent.
    if gmail_ready() and not session.get("credentials") and not session.get("oauth_error"):
        return redirect(url_for("gmail_login"))

    session.pop("oauth_error", None)

    return render_template_string(
        PAGE,
        connected=bool(session.get("credentials")),
        gmail_ready=gmail_ready(),
        model_status=MODEL_STATUS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Please enter a message to check."}), 400

    return jsonify(classify_text(text))


@app.route("/gmail/login")
def gmail_login():
    if not gmail_ready():
        return redirect(url_for("index"))

    flow = make_flow()
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    session["state"] = state
    # PKCE uses a verifier created by this Flow instance. Keep it across the
    # browser redirect so the callback can use the same verifier to exchange
    # Google's authorization code.
    session["code_verifier"] = flow.code_verifier
    return redirect(authorization_url)


@app.route(REDIRECT_PATH)
def oauth2callback():
    if not gmail_ready():
        return redirect(url_for("index"))

    state = session.pop("state", None)
    code_verifier = session.pop("code_verifier", None)
    if not state or not code_verifier:
        session["oauth_error"] = True
        flash("Google sign-in session expired. Please try connecting again.")
        return redirect(url_for("index"))

    try:
        flow = make_flow(state=state, code_verifier=code_verifier)
        flow.fetch_token(authorization_response=request.url)
        session["credentials"] = credentials_to_dict(flow.credentials)
    except Exception as exc:
        app.logger.exception("Google OAuth callback failed")
        session["oauth_error"] = True
        if "Scope has changed" in str(exc):
            flash(
                "Google did not grant Gmail read access. Add "
                "https://www.googleapis.com/auth/gmail.readonly to your OAuth "
                "Data Access scopes, then reconnect and approve Gmail access."
            )
        else:
            flash(f"Gmail sign-in failed: {exc}")
    return redirect(url_for("index"))


@app.route("/gmail/scan")
def gmail_scan():
    try:
        max_results = min(max(int(request.args.get("max_results", 10)), 1), 25)
        messages, error = get_gmail_messages(max_results=max_results)
        if error:
            return jsonify({"error": error}), 401
        return jsonify({"messages": messages})
    except HttpError as exc:
        return jsonify({"error": f"Gmail API error: {exc}"}), 502
    except Exception as exc:
        return jsonify({"error": f"Unable to scan Gmail: {exc}"}), 500


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
