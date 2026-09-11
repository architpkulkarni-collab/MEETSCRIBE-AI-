# AirScribe Feature #6 — R&D / Patent Technical Record

## Document Purpose

This document provides a technical record of the system-level architecture
of Feature #6 (Searchable Encrypted Note Store) for R&D reference.

It explicitly distinguishes standard/well-known technologies from any
potentially differentiated system-level mechanisms.

---

## Standard Technologies Used (NOT Novel)

These are industry-standard, widely available technologies. Their use in
this project does not constitute novelty.

| Technology | Usage | Standard |
|-----------|-------|----------|
| **AES-256-GCM** | Authenticated encryption of note content | NIST SP 800-38D |
| **HKDF-SHA256** | Key derivation from master secret | RFC 5869 |
| **sentence-transformers (all-MiniLM-L6-v2)** | Semantic text embedding | Open-source (Apache 2.0) |
| **Cosine similarity** | Embedding comparison for search ranking | Standard linear algebra |
| **MongoDB** | Document storage | Open-source database |
| **Flask** | HTTP API framework | Open-source (BSD) |
| **Base64 encoding** | Ciphertext serialization for JSON/BSON | RFC 4648 |

---

## System-Level Architecture (Potentially Differentiable)

The following describes the **system-level orchestration** — not the individual
components, but how they are combined and sequenced.

### Encrypt-Then-Embed-Then-Store Pattern

The core data flow follows a specific sequence:

```
1. Receive plaintext note content from authenticated user
2. Generate semantic embedding from plaintext (BEFORE encryption)
3. Encrypt plaintext with AES-256-GCM (note ID as AAD)
4. Store: encrypted content + unencrypted embedding + unencrypted metadata
5. Discard plaintext from memory (not persisted)
```

**Key design decision**: Embedding is generated from plaintext BEFORE
encryption, because the embedding model requires readable text. The embedding
is stored unencrypted alongside the encrypted content to enable search without
full corpus decryption.

### Selective Decryption Search Pattern

```
1. Receive search query from authenticated user
2. Generate embedding for query text
3. Fetch ONLY embeddings (lightweight) for user's notes — no decryption
4. Rank by cosine similarity
5. Decrypt ONLY the top-K results (minimizes decryption operations)
6. Return decrypted snippets to user
```

**Key design decision**: The search pipeline avoids decrypting the entire
corpus. Only embeddings (unencrypted) are used for ranking. Decryption is
deferred to the minimum necessary result set.

### AAD-Bound Note Identity

Each note's ciphertext is bound to its MongoDB `_id` via AES-GCM's
Associated Authenticated Data (AAD). This prevents:
- Ciphertext from being moved between notes
- A note's encrypted content from being substituted with another note's
- Decryption with the wrong note ID

### User-Scoped Server-Side Authorization

All database queries include `userId` filtering when a user is authenticated.
This is enforced server-side in every CRUD operation, preventing:
- User A from reading User B's notes by guessing IDs
- User A from modifying or deleting User B's notes
- Cross-user semantic search results

---

## Security Analysis

### What This System Protects Against
- **Database breach**: Encrypted content is unreadable without the key
- **Ciphertext swapping**: AAD binding prevents note content substitution
- **Tampered data**: GCM authentication rejects modified ciphertext
- **Cross-user access**: Server-side userId scoping on all operations

### What This System Does NOT Protect Against
- **Server compromise**: Decryption happens server-side; a compromised server
  can read all notes
- **Key compromise**: Single key for all users; compromise exposes everything
- **Embedding analysis**: Unencrypted embeddings leak topical information
- **Metadata analysis**: Note names, tags, folders are unencrypted
- **Side-channel attacks**: Not addressed

### Honest Assessment
This system provides **at-rest encryption with searchability** — a practical
defense against database-level data exposure. It does NOT provide end-to-end
encryption, per-user key isolation, or protection against a compromised
application server.

---

## Date of Implementation

September 2026

## Contributors

AirScribe Development Team

---

*This document is a factual technical record. It does not make claims of
novelty for any individual standard technology (AES-256-GCM, HKDF,
sentence-transformers, cosine similarity, MongoDB). Any potential
differentiation exists only at the system-level orchestration of these
standard components.*
