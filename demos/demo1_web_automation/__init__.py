"""Demo 1: Ultra-Fast Autonomous Web/OS Agent (UltraJev Multi-Scenario Engine)."""

from .mock_web_app import MockWebApp, HeadlessWebEnvironment, FormNavigationEnvironment, FormField, FormState
from .automation_executor import AutomationExecutor, Action, ActionType
from .ultrajev_scenarios import (
    UltraJevScenario,
    UltraJevStepDef,
    UltraJevDomain,
    ULTRAJEV_SCENARIOS_REGISTRY,
    get_scenario,
    list_available_scenarios,
)

__all__ = [
    'MockWebApp',
    'HeadlessWebEnvironment',
    'FormNavigationEnvironment',
    'FormField',
    'FormState',
    'AutomationExecutor',
    'Action',
    'ActionType',
    'UltraJevScenario',
    'UltraJevStepDef',
    'UltraJevDomain',
    'ULTRAJEV_SCENARIOS_REGISTRY',
    'get_scenario',
    'list_available_scenarios',
]
