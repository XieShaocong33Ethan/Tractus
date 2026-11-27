"""
Streaming Meeting Simulator - Multi-Speaker Conversational Audio/Text Streaming.

Simulates 3~4 participant discussions with conversational filler, turn-taking, and task requests.
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterator
from enum import Enum


class SpeakerType(Enum):
    """Speaker role types."""
    HOST = "Host (PM)"
    PARTICIPANT_1 = "Participant A (Marketing)"
    PARTICIPANT_2 = "Participant B (Engineering)"
    PARTICIPANT_3 = "Participant C (Industrial Design)"


@dataclass
class StreamChunk:
    """Streaming conversational utterance chunk."""
    speaker: SpeakerType
    text: str
    timestamp_ms: float
    is_filler: bool = False  # Conversational filler vs substantive
    duration_s: float = 1.5
    audio_url: str = ""
    speaker_id: str = ""
    task_title: str = ""
    intent: str = ""


class StreamSimulator:
    """Simulates real-time meeting utterance stream."""

    def __init__(self, meeting_transcript: List[StreamChunk] = None):
        if meeting_transcript is None:
            self.meeting_transcript = self._load_authentic_or_demo_transcript()
        else:
            self.meeting_transcript = meeting_transcript
        self.current_index = 0

    def _load_authentic_or_demo_transcript(self) -> List[StreamChunk]:
        """Loads Edinburgh AMI authentic dataset chunks if available, else fallback."""
        manifest_paths = [
            Path(__file__).resolve().parent / "audio" / "manifest.json",
            Path(__file__).resolve().parent.parent / "web_ui" / "static" / "audio" / "manifest.json",
        ]
        for manifest_path in manifest_paths:
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    chunks = data.get("demo2_meeting", {}).get("chunks", [])
                    if chunks:
                        transcript = []
                        speaker_map = {
                            "HOST": SpeakerType.HOST,
                            "PARTICIPANT_1": SpeakerType.PARTICIPANT_1,
                            "PARTICIPANT_2": SpeakerType.PARTICIPANT_2,
                            "PARTICIPANT_3": SpeakerType.PARTICIPANT_3,
                        }
                        for c in chunks:
                            spk_role = speaker_map.get(c.get("role_key"), SpeakerType.PARTICIPANT_1)
                            transcript.append(StreamChunk(
                                speaker=spk_role,
                                text=c["text"],
                                timestamp_ms=c["start_s"] * 1000.0,
                                is_filler=c.get("is_filler", False),
                                duration_s=c.get("duration_s", 2.0),
                                audio_url=c.get("url", ""),
                                speaker_id=c.get("speaker_id", "A"),
                                task_title=c.get("task_title", ""),
                                intent=c.get("intent", ""),
                            ))
                        return transcript
                except Exception:
                    pass
        return self._create_demo_transcript()

    def _create_demo_transcript(self) -> List[StreamChunk]:
        """Creates fallback demo meeting transcript."""
        transcript = []
        base_time = 0.0

        # Host introduction
        transcript.append(StreamChunk(
            SpeakerType.HOST,
            "Okay, today we are going to review feature priorities and budget limits.",
            base_time,
            is_filler=False
        ))
        base_time += 3500

        # Participant 1 filler and question
        transcript.append(StreamChunk(
            SpeakerType.PARTICIPANT_1,
            "Um",
            base_time,
            is_filler=True
        ))
        base_time += 400

        transcript.append(StreamChunk(
            SpeakerType.PARTICIPANT_1,
            "You know",
            base_time,
            is_filler=True
        ))
        base_time += 500

        transcript.append(StreamChunk(
            SpeakerType.PARTICIPANT_1,
            "I want to ask what the Q3 active user retention curves look like.",
            base_time,
            is_filler=False
        ))
        base_time += 2800

        # Participant 2 filler
        transcript.append(StreamChunk(
            SpeakerType.PARTICIPANT_2,
            "Right, exactly.",
            base_time,
            is_filler=True
        ))
        base_time += 600

        # Host response
        transcript.append(StreamChunk(
            SpeakerType.HOST,
            "The active user retention is holding at approximately seventy percent.",
            base_time,
            is_filler=False
        ))
        base_time += 2000

        # Participant 3 interjection
        transcript.append(StreamChunk(
            SpeakerType.PARTICIPANT_3,
            "Wait a second",
            base_time,
            is_filler=False
        ))
        base_time += 500

        transcript.append(StreamChunk(
            SpeakerType.PARTICIPANT_3,
            "What about the BOM calculation? Please retrieve the prototype BOM cost model.",
            base_time,
            is_filler=False
        ))
        base_time += 3200

        # Participant 1 follow-up
        transcript.append(StreamChunk(
            SpeakerType.PARTICIPANT_1,
            "Yeah sure",
            base_time,
            is_filler=True
        ))
        base_time += 400

        transcript.append(StreamChunk(
            SpeakerType.PARTICIPANT_1,
            "Yes, and we also need to incorporate the tiered scale discount curve.",
            base_time,
            is_filler=False
        ))
        base_time += 2500

        # Host conclusion
        transcript.append(StreamChunk(
            SpeakerType.HOST,
            "Agreed, let us record these action items on the milestone board.",
            base_time,
            is_filler=False
        ))
        base_time += 2800

        # Participant 2 acknowledgment
        transcript.append(StreamChunk(
            SpeakerType.PARTICIPANT_2,
            "Mm-hmm",
            base_time,
            is_filler=True
        ))
        base_time += 300

        transcript.append(StreamChunk(
            SpeakerType.PARTICIPANT_2,
            "Sounds good.",
            base_time,
            is_filler=True
        ))
        base_time += 400

        return transcript

    def stream(self, simulate_realtime: bool = False) -> Iterator[StreamChunk]:
        """
        Streams utterance chunks.

        Args:
            simulate_realtime: Whether to simulate real-time sleep intervals.

        Yields:
            StreamChunk instances.
        """
        last_time = 0.0

        for chunk in self.meeting_transcript:
            if simulate_realtime and self.current_index > 0:
                delay = (chunk.timestamp_ms - last_time) / 1000.0
                time.sleep(max(0, delay))

            yield chunk
            last_time = chunk.timestamp_ms
            self.current_index += 1

    def reset(self):
        """Resets simulator index."""
        self.current_index = 0

    def get_total_chunks(self) -> int:
        """Returns total chunk count."""
        return len(self.meeting_transcript)
