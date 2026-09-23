# EDITH voice core

This is the first local voice pipeline for EDITH: local wake-word detection, bounded voice-activity capture, Groq Whisper transcription, OpenRouter conversation state, and local `pyttsx3` speaker output.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Put the Groq and OpenRouter keys in `.env`. Never commit that file. The
`OPENROUTER_MODEL` setting defaults to `nvidia/nemotron-3.5-lightning:free`; set it to the exact GLM
5.3 model ID shown in your OpenRouter account if that is the model you intend to use.

The `WAKEWORD_MODEL_PATH` should point to a local openWakeWord `.onnx` model trained/configured for `EDITH`. The included controller does not upload the always-on microphone stream. Cloud calls begin only after the local detector fires and voice activity produces an utterance.

If `WAKEWORD_ENGINE=openwakeword` but the model file is missing, EDITH automatically falls back to spoken wake commands:
- `HEY ...`

In fallback mode, only utterances that start with `HEY` are forwarded to chat.
EDITH prints the raw transcript and explicit wake status in fallback mode so you can verify detection.

For wake-command testing without chat responses, set `WAKEWORD_TEST_MODE=1`. In this mode:
- only `GROQ_API_KEY` is required,
- each attempt prints transcript + detection status,
- successful detection prints `WAKE TEST SUCCESS: ...`.

Run with:

```powershell
python -m edith.app
```

The 30-second follow-up window keeps the conversation active after an answer. Speaker playback blocks the microphone state and adds a cooldown, preventing assistant audio from being transcribed as user input.

## Persistence roadmap

The assistant is being built in vertical slices:

1. **Drive-backed memory (current slice):** OAuth creates an `EDITH` folder in
   Google Drive. Conversations, metadata, and memory facts are JSON files in
   subfolders; no conversation data is written locally. Each model request
   includes recent messages plus lexical retrieval from stored conversations.
2. **Structured daily planning:** store goals, time slots, task status, and
   missed-task reasons as dated plan records. You can say commands such as
   `plan today goals: finish project, exercise; 2pm to 4pm: leetcode`. Plans
   are stored under `EDITH/plans/YYYY-MM-DD.json`. Natural language is also
   supported, for example: `Today I need to show JJ sir his project progress
   before 5:30, do 3 hours of LeetCode, and work on internship applications
   between 10pm and 11pm`. Later additions/removals such as `add at 5pm go
   collect my laundry` and `remove LeetCode` update the same plan.
3. **Reminders and check-ins:** a background scheduler checks Drive-backed plans
   every `REMINDER_INTERVAL_SECONDS` (default 15 seconds), reminds you
   `REMINDER_ADVANCE_MINUTES` before a slot, announces its start, and asks for
   completion afterward. Say “yes/done” to complete it or “no” and then give a
   reason; both the result and reason are saved in the plan JSON. Because
   reminder events are persisted, EDITH can recover reminders after downtime.
4. **Reports:** say “give me my weekly report” or “give me my monthly report”.
   EDITH aggregates plan completion, missed tasks, unfinished work, missed-task
   reasons, completion rates, and recommendations. Reports are saved under
   `EDITH/reports/`.
5. **Dashboard:** a later UI will read the same Drive-backed records.

EDITH must be running for real-time microphone reminders and speech. At the
configured daily prompt time (default **7:00 AM**), it asks for that day's
goals and time slots. If you miss it, the prompt is persisted in Drive and
repeated every `DAILY_PROMPT_RETRY_MINUTES` (default 15 minutes). The next
time you speak to EDITH, it checks for missed reminders immediately; giving a
plan for today marks the daily prompt answered. For reminders while EDITH is
off, keep the app running at startup (for example with Windows Task Scheduler)
or add a future OS notification service.

To enable the current slice, create a Google Cloud OAuth **Desktop app**
credential, save its downloaded JSON as `credentials.json` (or configure
`GOOGLE_DRIVE_CREDENTIALS_FILE`), and run EDITH. The first run opens a browser
for consent; only the OAuth token cache is local, while user data remains in
Drive. The Drive API dependencies are listed in `requirements.txt`.
