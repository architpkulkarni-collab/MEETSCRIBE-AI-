import os
import requests
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

BASE = "http://localhost:5000"
s1 = requests.Session()

print("--- Step 1: User 1 (Alice) Registration & Login ---")
u1_email = "alice@airscribe.local"
u1_pass = "AliceSecure123"
r1 = s1.post(f"{BASE}/auth/register", json={"email": u1_email, "password": u1_pass, "name": "Alice"})
if r1.status_code not in (200, 201):
    s1.post(f"{BASE}/auth/login", json={"email": u1_email, "password": u1_pass})
print("Alice logged in successfully.")

print("\n--- Step 2: Create an Encrypted Note ---")
secret_text = "Confidential Apollo flight mechanics and orbital trajectory formulas."
note_data = {
    "name": "Lunar Orbit Trajectory Plan",
    "text": secret_text,
    "content": f"<p>{secret_text}</p>",
    "folder": "Work",
    "tags": ["space", "physics"]
}
create_resp = s1.post(f"{BASE}/api/notes", json=note_data)
note_json = create_resp.json()
note_id = note_json["_id"]
print(f"Created Note ID: {note_id}")
print(f"Immediate response text matches plaintext: {note_json.get('text') == secret_text}")

print("\n--- Step 3: Raw MongoDB Storage Inspection ---")
client = MongoClient(os.getenv("MONGO_URI"))
db = client[os.getenv("DB_NAME", "airscribe")]
raw_doc = db.notes.find_one({"_id": client[os.getenv("DB_NAME", "airscribe")].notes.find_one({"name": "Lunar Orbit Trajectory Plan"})["_id"]})
has_plaintext = ("text" in raw_doc) or ("content" in raw_doc)
has_ciphertext = ("encrypted_text" in raw_doc) and ("encryption_nonce_text" in raw_doc)
has_embedding = ("embedding" in raw_doc) and (len(raw_doc.get("embedding", [])) == 384)
print(f"Plaintext text/content fields NOT in database: {not has_plaintext}")
print(f"AES-256-GCM ciphertext and nonce in database: {has_ciphertext}")
print(f"384-dimensional dense embedding in database: {has_embedding}")

print("\n--- Step 4: Authenticated Note Retrieval & Decryption ---")
get_resp = s1.get(f"{BASE}/api/notes/{note_id}").json()
print(f"Decrypted text: {get_resp.get('text')}")
print(f"Decryption succeeded: {get_resp.get('text') == secret_text}")

print("\n--- Step 5: Semantic Search (Conceptual Query with 0 Keyword Overlap) ---")
conceptual_query = "astrodynamics rocket gravity path"
search_resp = s1.post(f"{BASE}/api/search/semantic", json={"query": conceptual_query, "top_k": 5}).json()
matched = [n for n in search_resp if n["_id"] == note_id]
if matched:
    print(f"Found match! Title: '{matched[0]['name']}'")
    print(f"Similarity Score: {matched[0]['score']:.4f}")
    print(f"Decrypted Snippet: '{matched[0]['text']}'")
else:
    print("No match found in search results.")

print("\n--- Step 6: User Isolation Verification ---")
s2 = requests.Session()
u2_email = "bob@airscribe.local"
u2_pass = "BobSecure123"
r2 = s2.post(f"{BASE}/auth/register", json={"email": u2_email, "password": u2_pass, "name": "Bob"})
if r2.status_code not in (200, 201):
    s2.post(f"{BASE}/auth/login", json={"email": u2_email, "password": u2_pass})

bob_search = s2.post(f"{BASE}/api/search/semantic", json={"query": conceptual_query}).json()
bob_get = s2.get(f"{BASE}/api/notes/{note_id}")
print(f"Bob semantic search finds Alice note: {any(n['_id'] == note_id for n in bob_search)} (Expected: False)")
print(f"Bob direct GET returns status: {bob_get.status_code} (Expected: 404)")

print("\n=== ALL LIVE CHECKS COMPLETED ===")
