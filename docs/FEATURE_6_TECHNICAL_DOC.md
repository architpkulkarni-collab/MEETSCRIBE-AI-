# AirScribe Feature #6 — Searchable Encrypted Note Store

## Technical Architecture

### Overview

Feature #6 adds at-rest encryption for all note content and semantic search
capability to AirScribe. The system encrypts note text and HTML content before
persistence and decrypts only when serving to authenticated users. Semantic
search is enabled by pre-computed embedding vectors stored alongside encrypted
content.

### Data Flow

```
CREATE/UPDATE:
  plaintext → generate embedding → encrypt → store (encrypted + embedding)

READ:
  retrieve → decrypt → return plaintext to authenticated user

SEARCH:
  query → generate query embedding → cosine similarity rank → decrypt top-K
```

### Module Architecture

```
app.py (Flask API layer)
  ├── secure_notes.py (orchestration — encrypt/decrypt/embed)
  │     ├── encryption.py (AES-256-GCM crypto primitives)
  │     └── note_embeddings.py (sentence-transformers embedding)
  └── ai_server/config.py (centralized configuration)
```

---

## Encryption

### Algorithm: AES-256-GCM

- **Standard technology**: AES-256-GCM is a widely adopted NIST-approved
  authenticated encryption algorithm. It is NOT novel.
- **Key size**: 256-bit (32 bytes)
- **Nonce**: 96-bit (12 bytes), cryptographically random, unique per encryption
- **Authentication**: GCM provides built-in authentication (AEAD) — tampered
  ciphertext is detected and rejected
- **AAD**: The note's MongoDB `_id` is used as Associated Authenticated Data,
  binding each ciphertext to a specific note

### Key Management

- **Key Derivation**: HKDF-SHA256 derives the AES key from a master secret
  - HKDF is used here because the master key is a high-entropy random value
    (not a password). This is the correct use case for HKDF.
  - HKDF is NOT used as a password hash. It would be inappropriate for that.
- **Master Key**: 256-bit hex string stored in `NOTE_ENCRYPTION_MASTER_KEY`
  environment variable
- **Salt**: 128-bit hex string stored in `NOTE_ENCRYPTION_SALT` environment
  variable
- **Storage**: Keys are in `.env` (gitignored). Never in source code.
- **Rotation**: Not currently supported. Key rotation would require
  re-encrypting all notes with the new key.

### Encrypted Fields

| Field | Stored As | Notes |
|-------|-----------|-------|
| `text` (plaintext) | `encrypted_text` + `encryption_nonce_text` | Base64-encoded AES-GCM ciphertext |
| `content` (HTML) | `encrypted_content` + `encryption_nonce_content` | Same |
| `name` | Plaintext | Left unencrypted for metadata search |
| `folder` | Plaintext | Left unencrypted for filtering |
| `tags` | Plaintext | Left unencrypted for metadata search |
| `embedding` | Float array | Stored unencrypted for search |

---

## Semantic Search

### Embedding Model: all-MiniLM-L6-v2

- **Standard technology**: `sentence-transformers/all-MiniLM-L6-v2` is a
  well-known open-source model. It is NOT novel.
- **Dimensions**: 384
- **Size**: ~90MB (auto-downloaded on first use)
- **Runtime**: Local CPU inference (no API keys, no external calls)
- **License**: Apache 2.0

### Search Process

1. User submits a search query via `POST /api/search/semantic`
2. The query text is embedded using the same model
3. All note embeddings for the authenticated user are fetched from MongoDB
4. Cosine similarity is computed between the query embedding and each note
5. Results above the threshold (0.25) are ranked by score
6. Only the top-K results are decrypted and returned

### Security Considerations for Embeddings

**Embeddings are NOT equivalent to encryption.** They are a lossy semantic
representation that encodes topical information. An attacker with access to
embeddings could determine what topics notes are about.

Embeddings are stored unencrypted because:
- Encrypting embeddings would require decrypting the entire corpus for every
  search query, defeating the purpose of the encrypted store
- The embedding model runs locally — no note content is sent to external APIs
- This is a deliberate tradeoff: partial semantic leakage vs. searchability

---

## Database Changes

### Modified Collection: `notes`

New fields added per document:

| Field | Type | Description |
|-------|------|-------------|
| `encrypted_text` | String (base64) | AES-GCM ciphertext of `text` |
| `encryption_nonce_text` | String (base64) | Unique nonce for text encryption |
| `encrypted_content` | String (base64) | AES-GCM ciphertext of `content` |
| `encryption_nonce_content` | String (base64) | Unique nonce for content encryption |
| `encryption_version` | Integer | Currently `1` |
| `embedding` | Array of Float | 384-dimensional semantic vector |

Removed fields (after migration):

| Field | Replacement |
|-------|-------------|
| `text` | `encrypted_text` + `encryption_nonce_text` |
| `content` | `encrypted_content` + `encryption_nonce_content` |

### Modified Collection: `trash`

Same schema changes as `notes`.

---

## API Changes

### Modified Endpoints

All existing note CRUD endpoints now:
- Encrypt content before storage
- Decrypt content before returning to frontend
- Enforce `userId` scoping server-side

### New Endpoint

```
POST /api/search/semantic
Content-Type: application/json

Request:  { "query": "search text", "top_k": 10 }
Response: [
  {
    "_id": "...",
    "name": "Note Title",
    "text": "Decrypted snippet (max 200 chars)...",
    "score": 0.87,
    "folder": "General",
    "tags": ["tag1"],
    "updatedAt": "2024-01-01T00:00:00Z"
  }
]
```

### Modified Search Endpoint

```
GET /api/search?q=query
```

Now searches only unencrypted metadata fields (name, tags, folder) since
note text/content is encrypted. Falls back gracefully.

---

## Migration

### Script: `migration_encrypt.py`

```bash
# Preview changes without modifying data
python migration_encrypt.py --dry-run

# Run the migration
python migration_encrypt.py

# Verify migration (decrypts all notes to confirm correctness)
python migration_encrypt.py --verify
```

### What it does:
1. Iterates all documents in `notes` and `trash` collections
2. Skips already-encrypted documents (idempotent)
3. For each plaintext document:
   - Encrypts `text` and `content` fields
   - Generates a 384-dim embedding
   - Writes encrypted fields
   - Removes plaintext fields
4. Runs a verification pass to confirm all encrypted notes decrypt correctly
5. Reports summary with counts

### Safety:
- **Idempotent**: Already-encrypted notes are skipped
- **No silent deletion**: Plaintext is only removed after successful encryption
- **Verification**: Post-migration verification decrypts all notes
- **Dry run**: Preview mode available
- **Error reporting**: Failed notes are reported with IDs

---

## Security Assumptions & Limitations

### Assumptions
1. The `.env` file is protected by filesystem permissions
2. MongoDB Atlas uses TLS for transport encryption
3. The server process is trusted (decryption happens server-side)
4. Session-based authentication correctly identifies users

### Known Limitations
1. **Single encryption key**: All notes share one key. Compromise means all
   notes are exposed.
2. **No key rotation**: Changing keys requires re-encrypting all notes.
3. **Server-side decryption**: The server sees plaintext during request
   processing. True end-to-end encryption would require client-side crypto.
4. **Embedding leakage**: Embeddings reveal topic information even without
   decrypting note content.
5. **Metadata unencrypted**: Note names, tags, and folders are not encrypted.
6. **No per-user keys**: All users share the same encryption key. A compromised
   key exposes all users' notes.

---

## Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Generate Encryption Keys
```bash
python -c "import secrets; print('Master key:', secrets.token_hex(32)); print('Salt:', secrets.token_hex(16))"
```

### Run Migration
```bash
python migration_encrypt.py
```

### Verify Migration
```bash
python migration_encrypt.py --verify
```

### Run Tests
```bash
pytest tests/ -v
```

### Start Application
```bash
python app.py
```

---

## File Inventory

### Created (Feature #6)
| File | Purpose |
|------|---------|
| `encryption.py` | AES-256-GCM encrypt/decrypt primitives |
| `note_embeddings.py` | Sentence-transformers embedding + semantic search |
| `secure_notes.py` | Orchestration layer (encrypt→store, retrieve→decrypt) |
| `migration_encrypt.py` | One-time migration for existing plaintext notes |
| `tests/test_encryption.py` | Encryption unit tests |
| `tests/test_note_embeddings.py` | Embedding + search tests |
| `tests/test_secure_notes.py` | Orchestration + round-trip tests |
| `tests/test_integration_security.py` | Integration + security tests |

### Modified (Feature #6)
| File | Changes |
|------|---------|
| `app.py` | Encryption integration, semantic search endpoint, user scoping |
| `requirements.txt` | Added `cryptography`, `sentence-transformers` |
| `.env.example` | Added encryption key placeholders |
| `.env` | Added encryption key values |
| `ai_server/config.py` | Added encryption config variables |
| `AirScribe_v3.html` | Connected search UI to semantic search endpoint |
