# Logfire is configured before other app imports so spans are captured from the start.
# send_to_logfire="if-token-present": runs fine with no LOGFIRE_TOKEN (traces are simply not sent).
import logfire
from dotenv import load_dotenv

load_dotenv()
logfire.configure(send_to_logfire="if-token-present")

from fastapi import FastAPI

app = FastAPI(title="AdvancedRAG API")


@app.get("/")
def home():
    return {"message": "AdvancedRAG API is live.", "step": 0}
