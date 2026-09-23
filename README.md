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

## Phase 1 camera presence (optional)

Install `opencv-python-headless` from `requirements.txt` and set
`CAMERA_PRESENCE_ENABLED=1` in `.env`. EDITH samples the configured
`CAMERA_INDEX` at `CAMERA_SAMPLE_INTERVAL_SECONDS` and uses OpenCV's built-in
HOG person detector. `CAMERA_MIN_CONSECUTIVE_DETECTIONS` and
`CAMERA_MIN_CONSECUTIVE_ABSENCE` prevent one noisy frame from changing the
state. Optional `CAMERA_QUIET_START` and `CAMERA_QUIET_END` values use local
24-hour `HH:MM` time. During that interval EDITH releases the webcam
completely (rather than merely skipping frames), stops any active visitor
recording safely, clears the study observer's frame history, and marks
camera-derived presence as unavailable with a `camera quiet hours` reason.
When the active window starts, the worker opens the webcam again and resumes
sampling. This makes quiet hours an active camera power schedule; leaving both
values empty preserves the always-active behavior.

For an external webcam, set `CAMERA_INDEX` to that camera's OpenCV device
index. Start with `0`, then try `1`, `2`, and so on if the default device is
not the external webcam. The camera must be visible to the operating system
and available to only one application at a time.

While a person is not detected, reminder speech is held and flushed when
occupancy returns. During quiet hours presence is explicitly unavailable:
owner-gated commands remain fail-closed, while ordinary reminders retain
their normal behavior. If OpenCV or the camera cannot be used, EDITH reports
the degraded state and retries opening it on the next active sample. Frames are
processed locally and are never recorded, uploaded, or used for face
recognition. This phase supports presence only; it does not identify people.

### Optional local owner verification

Set `OWNER_VERIFICATION_ENABLED=1` only after installing the optional
`face-recognition` package and enrolling the owner. Enrollment is intentionally
local. For example, with EDITH stopped, run a one-off local script that opens
the camera, checks the returned frame, and calls
`OwnerVerifier.enroll(frame)` while one face is visible:

```python
import cv2
from edith.owner_verification import OwnerVerifier

camera = cv2.VideoCapture(0)
ok, frame = camera.read()
camera.release()
if not ok:
    raise RuntimeError("camera frame unavailable")
OwnerVerifier(".edith-owner.json").enroll(frame)
```

This writes only the numeric face encoding to
`OWNER_ENCODING_FILE` (default `.edith-owner.json`). Do not copy that file to
cloud storage. The verifier never records or uploads frames. A missing backend,
missing enrollment, or an unrecognized face is explicitly reported as
**unknown**, never as the owner. Verification is disabled by default and Phase
1 occupancy behavior is unchanged when it is disabled.

When verification is enabled, `PresenceSnapshot.identity` is `owner` only for
a successful match and `unknown` otherwise. `PresenceState` tracks the
owner's absence duration and invokes its optional `on_event` callback with
`owner_returned` after the configured 30-minute absence when the owner is
next confirmed. EDITH speaks a local “Welcome back, Aman” greeting for that
event. While verification is enabled, reminders and wake-word commands are
held unless the latest camera observation confirms the owner; this is a
fail-closed privacy behavior, so an unseen face can temporarily block EDITH.
Visitor recording and study observation remain separately opt-in features;
both pause while the camera power schedule is quiet.

### Phase 3 local study-session observation (opt-in)

Set `STUDY_OBSERVER_ENABLED=1` to reuse the existing local webcam worker and
produce an in-memory `StudySessionSummary`. Configure `STUDY_DESK_REGION` and
`STUDY_BED_REGION` as normalized `x,y,width,height` rectangles. The observer
aggregates person presence, estimated desk/bed occupancy, low motion, and a
confidence score; it never saves or uploads frames and does not record
visitors. Results are probabilistic and explicitly inconclusive when the detector, frame
visibility, or motion estimate is insufficient. They are not proof of studying,
sleeping, or absence. A planned slot is treated as a study slot when its JSON
has `study: true` (or `is_study: true`), or when its title contains one of the
case-insensitive fragments in `STUDY_SLOT_TITLE_PATTERNS`. At slot start the
observer resets; at slot end only the aggregate `study_summary` is written to
the dated plan JSON. No frames or frame signatures are persisted.

The end-of-slot check-in reports desk/bed/motion estimates and confidence, and
explicitly says when the result is inconclusive. A negative response records
the user's reason, then offers to reschedule later today or tomorrow. Existing
non-study reminders and owner gating are unchanged.

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
### Visitor recording (explicit opt-in)

Visitor recording is disabled by default. Set `VISITOR_RECORDING_ENABLED=true` only
after configuring Google Drive OAuth and owner verification. EDITH refuses to
start visitor recording unless the owner backend is installed and an owner
encoding is enrolled. When the webcam detects a person who is not confirmed as
the owner, EDITH announces recording, writes a bounded MP4 to the
operating system temporary directory, and uploads it to the configured
`VISITOR_DRIVE_COLLECTION`. The temporary file is deleted after both successful and
failed uploads; there is no local or silent cloud fallback. Owner-confirmed sessions
are never recorded. `VISITOR_MAX_SESSION_DURATION_SECONDS`, announcement text, and
retention metadata are configurable.

By default recordings remain video-only. Set `VISITOR_AUDIO_ENABLED=true` to open
a separate `sounddevice` microphone stream only for the visitor recording
interval. The stream writes a temporary WAV, and `ffmpeg` (configured with
`VISITOR_FFMPEG_PATH`) muxes it into the MP4 using the configured
`VISITOR_AUDIO_CODEC` (default `aac`). If the audio device is busy/unavailable,
sounddevice is missing, or ffmpeg/muxing fails, EDITH logs the reason and uploads
the video-only MP4 instead. The normal conversational `AudioInput` is not changed,
but Raspberry Pi/PortAudio devices may not permit two simultaneous streams.
`VISITOR_AUDIO_DEVICE` and `VISITOR_AUDIO_SAMPLE_RATE` are configurable.
WAV, source video, and muxed intermediates are always deleted after upload or
failure. Audio starts after the first video writer frame is initialized, so exact
sub-frame synchronization is not guaranteed; muxing uses `-shortest`. MP4
encoding still depends on an installed OpenCV codec. Google Drive requires OAuth
credentials and the `drive.file` scope; upload failures are logged and the
recording is discarded.
