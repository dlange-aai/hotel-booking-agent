"""
Hotel Booking Voice Agent
=========================

A LiveKit voice agent for testing AssemblyAI STT configurations over telephony.
Uses LiveKit's EnglishModel for turn detection, and exposes tools to
dynamically update AssemblyAI's min/max_turn_silence during the booking
collection phase.

See README.md for setup. Environment variables live in .env (see .env.example).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, room_io
from livekit.agents.stt import SpeechEvent, SpeechEventType
from livekit.agents.voice import ModelSettings
from livekit.agents.voice.events import (
    AgentFalseInterruptionEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
)
from livekit.plugins import (
    anthropic,
    assemblyai,
    cartesia,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.english import EnglishModel

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
logger = logging.getLogger("hotel-booking-agent")
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Transcript logger — writes to both console and a timestamped JSONL file
# ---------------------------------------------------------------------------
TRANSCRIPTS_DIR = Path(__file__).resolve().parent / "transcripts"
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
LOG_FILE = TRANSCRIPTS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.jsonl"


def log_event(event_type: str, **data):
    """Log an event to console and append to JSONL file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data,
    }
    logger.info("[%s] %s", event_type, json.dumps(data, default=str))
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def log_stt_event(ev: SpeechEvent):
    """Log a raw STT event with all available data."""
    if ev.type == SpeechEventType.START_OF_SPEECH:
        log_event("START_OF_SPEECH")

    elif ev.type == SpeechEventType.END_OF_SPEECH:
        log_event("END_OF_SPEECH")

    elif ev.type == SpeechEventType.RECOGNITION_USAGE:
        log_event(
            "RECOGNITION_USAGE",
            audio_duration=ev.recognition_usage.audio_duration if ev.recognition_usage else None,
        )

    elif ev.type in (
        SpeechEventType.INTERIM_TRANSCRIPT,
        SpeechEventType.PREFLIGHT_TRANSCRIPT,
        SpeechEventType.FINAL_TRANSCRIPT,
    ):
        alt = ev.alternatives[0] if ev.alternatives else None
        if not alt:
            return

        words = None
        if alt.words:
            words = [
                {
                    "word": str(w),
                    "start_time": w.start_time,
                    "end_time": w.end_time,
                    "confidence": w.confidence,
                }
                for w in alt.words
            ]

        log_event(
            ev.type.value,
            text=alt.text,
            confidence=alt.confidence,
            language=str(alt.language) if alt.language else None,
            start_time=alt.start_time,
            end_time=alt.end_time,
            speaker_id=alt.speaker_id,
            words=words,
        )


# ---------------------------------------------------------------------------
# Hotel Booking Agent
# ---------------------------------------------------------------------------

HOTEL_SYSTEM_PROMPT = """\
You are the front desk at The Grandview Hotel in downtown San Francisco. \
Keep replies short and conversational — one or two sentences. Follow the \
caller's lead: answer what they ask, don't lecture.

Greeting: "I'm from The Grandview Hotel, how can I help?"

Hotel facts (only mention what's asked):
- 850 Market Street, SF. Check-in 3pm, check-out 11am.
- Standard $199/night, Deluxe $299, Suite $499. WiFi and breakfast included.
- Rooftop pool, fitness center, spa, restaurant, valet parking ($45/night).
- Pet-friendly in Standard and Deluxe ($50/night). Airport shuttle $35 each way.

If the caller wants to book, call set_booking_mode first, then collect \
one item at a time — don't read the whole list up front:
name, check-in date, check-out date, guests, room type, phone, email, \
credit card number, expiration, CVV.

Read back the details, confirm, then call set_general_mode and wrap up \
with "You're all set — a confirmation email is on its way."

If the caller interrupts or corrects you, go with it. Never restart the \
booking from the top unless they ask.\
"""


class HotelBookingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=HOTEL_SYSTEM_PROMPT)

    async def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ):
        async for ev in Agent.default.stt_node(self, audio, model_settings):
            log_stt_event(ev)
            yield ev

    @agents.function_tool()
    async def set_booking_mode(self, context: agents.RunContext[None]) -> str:
        """Call this before starting to collect booking details (name, dates, phone, email, credit card)."""
        stt_instance: assemblyai.STT = context.session.stt
        stt_instance.update_options(
            min_turn_silence=500
        )
        log_event(
            "config_update",
            step="booking_mode",
            min_turn_silence=500
        )
        return "Turn detection adjusted for booking detail collection. Proceed with collecting the caller's information."

    @agents.function_tool()
    async def set_general_mode(self, context: agents.RunContext[None]) -> str:
        """Call this after booking details are fully collected and confirmed."""
        stt_instance: assemblyai.STT = context.session.stt
        stt_instance.update_options(
            min_turn_silence=200,
            max_turn_silence=1500
        )
        context.session.update_options(min_endpointing_delay=1)
        log_event(
            "config_update",
            step="general_mode",
            min_turn_silence=200,
            max_turn_silence=1500
        )
        return "Turn detection restored to general conversation settings."


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=assemblyai.STT(
            model="u3-rt-pro",
            min_turn_silence=200,
            max_turn_silence=1500,
            vad_threshold=0.3,
            prompt="Transcribe verbatim with standard punctuation. Include filler words and incomplete utterances.",
            keyterms_prompt=[
                "SpringHill",
                "TownePlace",
                "WoodSpring",
                "DoubleTree",
                "Hilton",
                "Marriott",
                "Bonvoy",
                "suite",
                "valet",
            ],
        ),
        tts=cartesia.TTS(model="sonic-3", voice="607167f6-9bf2-473c-accc-ac7b3b66b30b"),
        llm=anthropic.LLM(model="claude-sonnet-4-5-20250929"),
        vad=silero.VAD.load(
            activation_threshold=0.3
        ),
        turn_handling={
            "turn_detection": EnglishModel(),
            "endpointing": {"min_delay": 1.0, "max_delay": 4.0},
            "interruption": {
                "enabled": True,
                "min_duration": 0.5,
                "min_words": 2,
                "resume_false_interruption": True,
                "false_interruption_timeout": 1.5,
            },
        },
        allow_interruptions=True,
        max_tool_steps=10,
    )

    @session.on("user_state_changed")
    def _on_user_state(ev: UserStateChangedEvent):
        log_event(
            "user_state_changed",
            old=ev.old_state,
            new=ev.new_state,
            created_at=ev.created_at,
        )

    @session.on("user_input_transcribed")
    def _on_user_input(ev: UserInputTranscribedEvent):
        log_event(
            "user_input_transcribed",
            transcript=ev.transcript,
            is_final=ev.is_final,
            created_at=ev.created_at,
        )

    @session.on("agent_false_interruption")
    def _on_false_interruption(ev: AgentFalseInterruptionEvent):
        log_event(
            "agent_false_interruption",
            resumed=ev.resumed,
            created_at=ev.created_at,
        )

    @session.on("speech_created")
    def _on_speech_created(ev: SpeechCreatedEvent):
        log_event(
            "speech_created",
            source=ev.source,
            user_initiated=ev.user_initiated,
        )

    await session.start(
        room=ctx.room,
        agent=HotelBookingAgent(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        ),
    )

    log_event("session_started", room=ctx.room.name)

    await session.generate_reply(
        instructions='Say exactly: "I\'m from The Grandview Hotel, how can I help?"'
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="hotel-booking-agent",
        )
    )
