# Hotel Booking Voice Agent

A LiveKit voice agent that acts as the front desk of a fictional hotel, designed for testing real-time STT and turn-detection configurations over telephony. Every session writes a timestamped JSONL to `transcripts/` with STT interims/finals (word-level timing, confidences, language tags), VAD state, agent state, and false-interruption signals — useful for post-hoc analysis of what tripped which turn/interrupt gate.

Two variants share all other components (Cartesia TTS, Anthropic Claude, Silero VAD, LiveKit English turn detection, Krisp BVC Telephony noise cancellation):

- `hotel_agent.py` — AssemblyAI Universal Streaming (`u3-rt-pro`)
- `hotel_agent_deepgram.py` — Deepgram Flux (`flux-general-en`)

## Prerequisites

- Python 3.10+
- Accounts / API keys for:
  - LiveKit Cloud (URL, API key, API secret)
  - Anthropic
  - Cartesia
  - AssemblyAI — only if running the AAI variant
  - Deepgram — only if running the Deepgram variant
- A phone number dispatched to the agent via a LiveKit SIP inbound trunk (see below)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and fill in your keys
```

Download the model weights used by the Silero VAD, LiveKit turn detector, and Krisp noise-cancellation plugins (run once per variant — or once, it caches):

```bash
python hotel_agent.py download-files
python hotel_agent_deepgram.py download-files   # if using the DG variant
```

## Run

```bash
python hotel_agent.py dev            # AssemblyAI variant
python hotel_agent_deepgram.py dev   # Deepgram variant
```

Both variants register under the same `agent_name="hotel-booking-agent"`, so a single LiveKit dispatch rule (and phone number) serves whichever variant is currently running. Run one at a time.

## Wire up a phone number

The quickest path is [LiveKit Phone Numbers](https://docs.livekit.io/telephony/start/phone-numbers/) — buy a US number directly through LiveKit, no separate SIP provider needed. Every LiveKit Cloud plan includes one free local number.

Requires the [LiveKit CLI](https://docs.livekit.io/reference/developer-tools/livekit-cli/) (`lk`), authenticated against the same project as your `.env` keys.

### 0. Install and authenticate the LiveKit CLI

```bash
# macOS
brew install livekit-cli

# Linux
curl -sSL https://get.livekit.io/cli | bash
```

Then authenticate against your LiveKit Cloud project:

```bash
lk cloud auth
```

### 1. Buy a phone number

```bash
# search for available numbers in your area code
lk number search --country-code US --area-code 415

# purchase one
lk number purchase --numbers +14155550100
```

### 2. Create the dispatch rule

A template is included at [`dispatch-rule.json`](./dispatch-rule.json). It routes every inbound call into its own room (prefix `hotel-`) and dispatches `hotel-booking-agent` into that room.

Create it and note the returned ID:

```bash
lk sip dispatch create dispatch-rule.json
```

### 3. Assign the dispatch rule to your number

```bash
lk number update --number +14155550100 --sip-dispatch-rule-id <DISPATCH_RULE_ID>
```

### 4. Call it

Start the agent (`python hotel_agent.py dev` or `python hotel_agent_deepgram.py dev`) and call the phone number you bought. You should hear: _"I'm from The Grandview Hotel, how can I help?"_

> Both scripts register as `agent_name="hotel-booking-agent"`, so a single dispatch rule and phone number serve whichever variant is currently running — just don't run both at once.

## Transcripts and recordings

Each session writes a pair of artifacts into its own timestamped folder:

```
transcripts/
├── assembly/
│   └── 2026-04-22_143012/
│       ├── session.jsonl
│       └── session.wav
└── deepgram/
    └── 2026-04-22_143012/
        ├── session.jsonl
        └── session.wav
```

- `session.jsonl` — one JSON object per line. Event types include `START_OF_SPEECH`, `END_OF_SPEECH`, `interim_transcript`, `preflight_transcript`, `final_transcript`, `user_state_changed`, `agent_state_changed`, `user_input_transcribed`, `agent_false_interruption`, `speech_created`, and `config_update` (when the booking-mode tools fire).
- `session.wav` — 16 kHz stereo PCM. Caller on the left channel, agent on the right. Useful for playing back alongside the JSONL to see which gate fired when.

## Tuning knobs

The interesting parameters to adjust while testing live in `entrypoint()`:

- `assemblyai.STT(...)` / `deepgram.STTv2(...)` — STT-level VAD and turn-silence thresholds
- `silero.VAD.load(...)` — LiveKit-side VAD
- `turn_handling.endpointing` — `min_delay` / `max_delay` for turn finalization
- `turn_handling.interruption` — `min_duration`, `min_words`, `false_interruption_timeout`, etc.

The two function tools (`set_booking_mode`, `set_general_mode`) also live there and demonstrate runtime STT reconfiguration during a call.
