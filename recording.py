"""Per-session artifacts: stereo WAV + JSONL event log.

Each call writes into transcripts/<vendor>/<stamp>/:
    session.jsonl — one event per line (STT, state changes, etc.)
    session.wav   — 16 kHz stereo, caller on left, agent on right

feed_left() and feed_right() accept LiveKit AudioFrames at any sample rate;
frames are resampled to 16 kHz on arrival and written as interleaved stereo
every 20 ms, padding with silence on whichever side is idle that window.
"""

from __future__ import annotations

import asyncio
import json
import logging
import wave
from datetime import datetime, timezone
from pathlib import Path

from livekit import rtc
from livekit.agents.stt import SpeechEvent, SpeechEventType

TARGET_SAMPLE_RATE = 16_000
FRAME_MS = 20
SAMPLES_PER_FRAME = TARGET_SAMPLE_RATE * FRAME_MS // 1000
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2  # int16 mono

logger = logging.getLogger("hotel-booking-agent")


class Session:
    """Owns the per-call WAV + JSONL files in transcripts/<vendor>/<stamp>/."""

    def __init__(self, transcripts_root: Path, vendor: str):
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.dir = transcripts_root / vendor / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "session.jsonl"
        self.wav_path = self.dir / "session.wav"
        self.recorder = StereoRecorder(self.wav_path)

    def log(self, event_type: str, **data) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **data,
        }
        logger.info("[%s] %s", event_type, json.dumps(data, default=str))
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def log_stt_event(self, ev: SpeechEvent) -> None:
        if ev.type == SpeechEventType.START_OF_SPEECH:
            self.log("START_OF_SPEECH")
        elif ev.type == SpeechEventType.END_OF_SPEECH:
            self.log("END_OF_SPEECH")
        elif ev.type == SpeechEventType.RECOGNITION_USAGE:
            self.log(
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
            self.log(
                ev.type.value,
                text=alt.text,
                confidence=alt.confidence,
                language=str(alt.language) if alt.language else None,
                start_time=alt.start_time,
                end_time=alt.end_time,
                speaker_id=alt.speaker_id,
                words=words,
            )


class StereoRecorder:
    def __init__(self, path: Path):
        self._wf = wave.open(str(path), "wb")
        self._wf.setnchannels(2)
        self._wf.setsampwidth(2)
        self._wf.setframerate(TARGET_SAMPLE_RATE)

        self._left = bytearray()
        self._right = bytearray()
        self._left_rs: rtc.AudioResampler | None = None
        self._right_rs: rtc.AudioResampler | None = None
        self._closed = False
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._writer_loop())

    def feed_left(self, frame: rtc.AudioFrame) -> None:
        self._feed(frame, left=True)

    def feed_right(self, frame: rtc.AudioFrame) -> None:
        self._feed(frame, left=False)

    def _feed(self, frame: rtc.AudioFrame, left: bool) -> None:
        if self._closed:
            return

        if frame.sample_rate != TARGET_SAMPLE_RATE:
            attr = "_left_rs" if left else "_right_rs"
            rs = getattr(self, attr)
            if rs is None:
                rs = rtc.AudioResampler(
                    input_rate=frame.sample_rate,
                    output_rate=TARGET_SAMPLE_RATE,
                    num_channels=1,
                )
                setattr(self, attr, rs)
            pcm = b"".join(bytes(f.data) for f in rs.push(frame))
        else:
            pcm = bytes(frame.data)

        (self._left if left else self._right).extend(pcm)

    async def _writer_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(FRAME_MS / 1000)
                self._write_frame()
        except asyncio.CancelledError:
            pass

    def _write_frame(self) -> None:
        l = self._pop(self._left)
        r = self._pop(self._right)
        stereo = bytearray(BYTES_PER_FRAME * 2)
        for i in range(SAMPLES_PER_FRAME):
            stereo[i * 4 : i * 4 + 2] = l[i * 2 : i * 2 + 2]
            stereo[i * 4 + 2 : i * 4 + 4] = r[i * 2 : i * 2 + 2]
        self._wf.writeframes(bytes(stereo))

    @staticmethod
    def _pop(buf: bytearray) -> bytes:
        if len(buf) >= BYTES_PER_FRAME:
            chunk = bytes(buf[:BYTES_PER_FRAME])
            del buf[:BYTES_PER_FRAME]
            return chunk
        chunk = bytes(buf) + b"\x00" * (BYTES_PER_FRAME - len(buf))
        buf.clear()
        return chunk

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        for rs, buf in ((self._left_rs, self._left), (self._right_rs, self._right)):
            if rs is not None:
                for f in rs.flush():
                    buf.extend(bytes(f.data))

        while self._left or self._right:
            self._write_frame()

        self._wf.close()
