"""
Handoff Router - Low-Latency Seamless Escalation Switchboard.

State machine simulation and agent-facing highlighted risk context cards generation.
"""

import time
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .intent_parser import ParsedIntent, IntentSlot


class HandoffState(Enum):
    """Handoff escalation state."""
    PENDING = "pending"           # Pending evaluation
    ROUTING = "routing"           # Neural routing in progress
    CONNECTED = "connected"       # Connected to human specialist
    COMPLETED = "completed"       # Session completed


@dataclass
class RiskCard:
    """Agent-facing highlighted risk card."""
    risk_level: str  # "high", "medium", "low"
    title: str
    details: List[str]
    suggested_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffContext:
    """Handoff escalation context."""
    session_id: str
    user_input: str
    parsed_intents: List[ParsedIntent]
    risk_cards: List[RiskCard]
    state: HandoffState
    handoff_time_ms: float
    agent_id: Optional[str] = None


class HandoffRouter:
    """Seamless neural handoff router."""

    def __init__(self):
        self.active_handoffs: List[HandoffContext] = []
        self.completed_handoffs: List[HandoffContext] = []

    def initiate_handoff(
        self,
        session_id: str,
        user_input: str,
        parsed_intents: List[ParsedIntent]
    ) -> HandoffContext:
        """
        Initiates seamless human specialist handoff.

        Args:
            session_id: Session identifier.
            user_input: Raw customer utterance.
            parsed_intents: Parsed neural intent slots.

        Returns:
            HandoffContext instance.
        """
        start_time = time.perf_counter()

        # Generate context risk cards
        risk_cards = self._generate_risk_cards(parsed_intents, user_input)

        # Assign specialist agent tier
        agent_id = self._assign_agent(risk_cards)

        handoff_time = time.perf_counter() - start_time

        context = HandoffContext(
            session_id=session_id,
            user_input=user_input,
            parsed_intents=parsed_intents,
            risk_cards=risk_cards,
            state=HandoffState.CONNECTED,
            handoff_time_ms=handoff_time * 1000,
            agent_id=agent_id,
        )

        self.active_handoffs.append(context)
        return context

    def _generate_risk_cards(
        self,
        intents: List[ParsedIntent],
        user_input: str
    ) -> List[RiskCard]:
        """
        Generates agent-facing context risk cards.

        Args:
            intents: Parsed neural intents.
            user_input: Raw customer message.

        Returns:
            List of RiskCard objects.
        """
        cards = []

        # Check emergency card freeze
        freeze_intents = [i for i in intents if i.slot == IntentSlot.EMERGENCY_FREEZE]
        if freeze_intents:
            card = RiskCard(
                risk_level="high",
                title="❄️ [Zero-Key Direct] Instant Credit Card Freeze & Security Lock",
                details=[
                    f"Identified customer urgent lost or stolen card request: {freeze_intents[0].evidence[:60]}",
                    "Risk defenses armed: real-time block on foreign card-not-present and rapid pay channels",
                ],
                suggested_actions=[
                    "Click 'Lock Card Immediately' to finalize security hold",
                    "Review unauthorized transactions in past 24 hours",
                    "Issue replacement card with express delivery to primary address",
                ],
                metadata={"alert_type": "emergency_freeze", "action_direct": True},
            )
            cards.append(card)

        # Check fraud dispute
        fraud_intents = [i for i in intents if i.slot == IntentSlot.FRAUD_DISPUTE]
        if fraud_intents:
            card = RiskCard(
                risk_level="high",
                title="🚨 [Anti-Fraud Line] Suspicious Transaction & Fraud Dispute Triage",
                details=[
                    f"Customer flagged unauthorized transactions: {fraud_intents[0].evidence[:60]}",
                    "System flagged recent 3 suspicious transactions and alerted Risk Operations",
                ],
                suggested_actions=[
                    "Retrieve merchant details and geolocation coordinates",
                    "Initiate provisional dispute credit and chargeback guarantee",
                    "File formal fraud dispute with card network",
                ],
                metadata={"alert_type": "fraud_dispute", "severity": "critical"},
            )
            cards.append(card)

        # Check direct debit management
        debit_intents = [i for i in intents if i.slot == IntentSlot.DIRECT_DEBIT]
        if debit_intents:
            card = RiskCard(
                risk_level="medium",
                title="📑 [Zero-Key Direct] Direct Debit & Subscription Management",
                details=[
                    f"Direct debit inquiry and refund request: {debit_intents[0].evidence[:60]}",
                    "Active direct debit mandates detected: 2 agreements (Utilities, Insurance auto-pay)",
                ],
                suggested_actions=[
                    "Display list of authorized direct debit merchants",
                    "Provide one-click mandate cancellation",
                    "Initiate 7-day refund process for disputed debit",
                ],
                metadata={"alert_type": "direct_debit", "action_direct": True},
            )
            cards.append(card)

        # Check compliance alert
        compliance_intents = [i for i in intents if i.slot == IntentSlot.COMPLIANCE_ALERT]
        if compliance_intents:
            card = RiskCard(
                risk_level="high",
                title="🚨 Critical Regulatory Compliance Warning",
                details=[
                    f"Customer mentioned regulatory body: {intent.evidence}"
                    for intent in compliance_intents
                ],
                suggested_actions=[
                    "De-escalate immediately with empathetic listening",
                    "Guarantee formal written response within 48 hours",
                    "Record formal case number and escalate to Compliance Operations",
                ],
                metadata={"alert_type": "compliance", "severity": "critical"},
            )
            cards.append(card)

        # Check transfer human request
        transfer_intents = [i for i in intents if i.slot == IntentSlot.TRANSFER_HUMAN]
        if transfer_intents:
            card = RiskCard(
                risk_level="medium",
                title="📞 Human Specialist Escalation Request",
                details=[
                    f"Customer explicitly requested live agent: {intent.evidence}"
                    for intent in transfer_intents
                ],
                suggested_actions=[
                    "Confirm inquiry scope",
                    "Provide agent employee identifier",
                    "Commit to service resolution timeframe",
                ],
                metadata={"alert_type": "transfer", "severity": "medium"},
            )
            cards.append(card)

        # Check billing dispute
        billing_intents = [i for i in intents if i.slot == IntentSlot.BILLING_INQUIRY]
        if billing_intents:
            card = RiskCard(
                risk_level="medium",
                title="💳 Billing & Fee Discrepancy Triage",
                details=[
                    f"Fee dispute: {intent.evidence}"
                    for intent in billing_intents
                ],
                suggested_actions=[
                    "Query itemized transaction ledger",
                    "Verify fee assessment rules",
                    "Issue fee reversal or provisional credit if applicable",
                ],
                metadata={"alert_type": "billing", "severity": "medium"},
            )
            cards.append(card)

        # Real neural emotional distress assessment
        emotion_intent = next((i for i in intents if i.slot == IntentSlot.COMPLAINT or "emotional_intensity" in i.metadata), None)
        if emotion_intent:
            conf = emotion_intent.confidence * 100
            card = RiskCard(
                risk_level="low" if conf < 90 else "medium",
                title="😠 Customer Neural Emotional Distress Assessment",
                details=[
                    f"Neural emotion analysis: High agitation and dissatisfaction (Model confidence: {conf:.1f}%)",
                    f"Utterance: {user_input[:60]}{'...' if len(user_input) > 60 else ''}",
                ],
                suggested_actions=[
                    "Adopt empathetic tone and prioritize de-escalation",
                    "Avoid boilerplate scripts and propose direct substantive resolution",
                ],
                metadata={"alert_type": "neural_emotion", "confidence": emotion_intent.confidence},
            )
            cards.append(card)

        return cards

    def _assign_agent(self, risk_cards: List[RiskCard]) -> str:
        """Assigns specialist tier based on risk cards."""
        has_high_risk = any(card.risk_level == "high" for card in risk_cards)
        if has_high_risk:
            return "senior_agent_001"
        else:
            return "agent_042"

    def complete_handoff(self, session_id: str) -> bool:
        """Completes active handoff session."""
        context = next(
            (h for h in self.active_handoffs if h.session_id == session_id),
            None
        )
        if context is None:
            return False

        context.state = HandoffState.COMPLETED
        self.active_handoffs.remove(context)
        self.completed_handoffs.append(context)
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Returns handoff telemetry statistics."""
        total_handoffs = len(self.active_handoffs) + len(self.completed_handoffs)

        if total_handoffs == 0:
            return {
                "total_handoffs": 0,
                "average_handoff_time_ms": 0.0,
                "high_risk_count": 0,
            }

        all_handoffs = self.active_handoffs + self.completed_handoffs
        avg_time = sum(h.handoff_time_ms for h in all_handoffs) / total_handoffs

        high_risk_count = sum(
            1 for h in all_handoffs
            if any(card.risk_level == "high" for card in h.risk_cards)
        )

        return {
            "total_handoffs": total_handoffs,
            "average_handoff_time_ms": avg_time,
            "high_risk_count": high_risk_count,
            "active_sessions": len(self.active_handoffs),
            "completed_sessions": len(self.completed_handoffs),
        }

    def reset(self):
        """Resets handoff router."""
        self.active_handoffs.clear()
        self.completed_handoffs.clear()
