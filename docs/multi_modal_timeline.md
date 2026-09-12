# Multi-Modal Timeline Fusion

## Purpose

MeetScribe now maintains one meeting timeline for three modalities:

- **Audio/transcript** — Whisper segment timestamps are shifted by the originating 8-second audio chunk offset instead of restarting at `0s` for every request.
- **Typed notes** — Quill user input is captured with the current meeting-relative timestamp after a short debounce.
- **Screen/OCR** — an optional screen-share capture samples the shared screen every 4 seconds and runs local browser OCR with Tesseract.js. The timestamp is taken when the frame is captured, while the OCR completion time is retained as `observedAt`.

## Event model

Each event contains:

- `timestamp`: normalized meeting-relative timestamp used for ordering
- `rawTimestamp`: original producer timestamp
- `observedAt`: arrival/completion timestamp when available
- `correction`: explicit timing correction applied/recorded
- `correctionReason`: why the correction was recorded
- `type`: `transcript`, `note`, or `ocr`
- `source`: `audio`, `typed-note`, or `screen`
- `content`: event text
- `metadata`: speaker, OCR dimensions, chunk origin, etc.
- `confidence`: producer confidence where available

## Timing-drift handling

The engine does not replace a capture timestamp with a later processing timestamp. When both are available and the difference is positive but within the configured 12-second bound, the event remains anchored to its capture time and records the negative drift explicitly. This prevents slow OCR/transcription processing from moving an event to the wrong point on the meeting timeline.

## Transcript chunk alignment

The existing AI meeting recorder sends approximately 8-second chunks. Whisper returns timestamps relative to each chunk. The integration now calculates:

`meetingTimestamp = chunkIndex * 8 + whisperSegment.start`

This prevents the timeline from containing repeated `0s–8s` ranges for successive chunks.

## Privacy

OCR is performed in the browser with Tesseract.js after the user explicitly starts screen capture. Screen capture is stopped when the browser track ends or the user presses the stop control. The timeline is kept in memory and can be exported as JSON for testing.

## Manual verification

1. Start an AI meeting.
2. Speak for more than 16 seconds.
3. Confirm transcript events appear around `0–8s`, `8–16s`, etc., rather than restarting at zero.
4. Type a note while speaking; confirm a `note` event appears near the typing time.
5. Open **Unified Timeline → Start Screen/OCR**, share a presentation/window, and wait for OCR samples.
6. Confirm OCR events use the frame capture time even when OCR takes longer to finish.
7. Export the timeline and inspect `rawTimestamp`, `observedAt`, `correction`, and `correctionReason`.
