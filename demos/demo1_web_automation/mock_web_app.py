"""
Mock Web Application DOM Environment - Multi-Step Form Simulation.

Simulates enterprise cross-platform flight booking and expense reimbursement workflows.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class FormField:
    """Form field specification."""
    name: str
    field_type: str  # text, select, checkbox, date
    required: bool = True
    options: List[str] = field(default_factory=list)  # for select
    validation_rule: Optional[str] = None


@dataclass
class FormState:
    """Form state tracking."""
    fields: Dict[str, Any] = field(default_factory=dict)
    current_step: int = 1
    total_steps: int = 5
    errors: List[str] = field(default_factory=list)
    submitted: bool = False


class MockWebApp:
    """Headless Web Application DOM environment for autonomous form navigation benchmarks."""

    def __init__(self):
        self.form_definition = self._create_form_definition()
        self.state = FormState(total_steps=len(self.form_definition))

    def _create_form_definition(self) -> Dict[int, List[FormField]]:
        """Creates multi-step form definition (20+ fields)."""
        return {
            1: [  # Stage 1: Departure Information
                FormField("departure_city", "select", options=["San Francisco", "New York", "London", "Tokyo", "Beijing", "Shanghai", "Guangzhou", "Shenzhen"]),
                FormField("departure_date", "date"),
                FormField("departure_time", "select", options=["Morning", "Afternoon", "Evening"]),
            ],
            2: [  # Stage 2: Arrival Information
                FormField("arrival_city", "select", options=["Zurich", "Frankfurt", "Paris", "Singapore", "Chengdu", "Hangzhou", "Xi'an", "Wuhan"]),
                FormField("arrival_date", "date"),
                FormField("cabin_class", "select", options=["Economy", "Business", "First"]),
            ],
            3: [  # Stage 3: Passenger Information
                FormField("passenger_name", "text"),
                FormField("id_card", "text", validation_rule="18_digits"),
                FormField("phone", "text", validation_rule="11_digits"),
                FormField("email", "text", validation_rule="email"),
            ],
            4: [  # Stage 4: Expense & Reimbursement
                FormField("company_name", "text"),
                FormField("department", "select", options=["Engineering", "Marketing", "Finance", "Operations", "Human Resources"]),
                FormField("cost_center", "text"),
                FormField("project_code", "text"),
                FormField("invoice_type", "select", options=["Corporate Tax Invoice", "Standard Receipt", "VAT Special Invoice", "Standard VAT Invoice"]),
            ],
            5: [  # Stage 5: Attachments & Final Submission
                FormField("receipt_uploaded", "checkbox"),
                FormField("itinerary_uploaded", "checkbox"),
                FormField("boarding_pass_uploaded", "checkbox"),
                FormField("agreement_checked", "checkbox"),
            ],
        }

    def get_current_step_fields(self) -> List[FormField]:
        """Gets form fields for current step."""
        return self.form_definition.get(self.state.current_step, [])

    def fill_field(self, field_name: str, value: Any) -> bool:
        """
        Fills a form field.

        Returns:
            True if filled successfully, False otherwise.
        """
        current_fields = self.get_current_step_fields()
        field = next((f for f in current_fields if f.name == field_name), None)

        if field is None:
            self.state.errors.append(f"Field {field_name} does not exist in current step")
            return False

        # Validation
        if field.field_type == "select" and value not in field.options:
            self.state.errors.append(f"Field {field_name} value {value} not in options")
            return False

        if field.validation_rule:
            if not self._validate_value(value, field.validation_rule):
                self.state.errors.append(f"Field {field_name} validation failed")
                return False

        self.state.fields[field_name] = value
        return True

    def _validate_value(self, value: str, rule: str) -> bool:
        """Validates field value according to rule."""
        if rule == "18_digits":
            return isinstance(value, str) and len(value) == 18 and value.isalnum()
        elif rule == "11_digits":
            return isinstance(value, str) and len(value) == 11 and value.isdigit()
        elif rule == "email":
            return isinstance(value, str) and "@" in value and "." in value
        return True

    def next_step(self) -> bool:
        """
        Advances to next form step.

        Returns:
            True if advanced successfully, False otherwise.
        """
        # Verify all required fields in current step
        current_fields = self.get_current_step_fields()
        for field in current_fields:
            if field.required and field.name not in self.state.fields:
                self.state.errors.append(f"Required field {field.name} not filled")
                return False

        if self.state.current_step < self.state.total_steps:
            self.state.current_step += 1
            return True
        return False

    def submit(self) -> bool:
        """
        Submits final form.

        Returns:
            True if submitted successfully, False otherwise.
        """
        if self.state.current_step != self.state.total_steps:
            self.state.errors.append("All steps must be completed before submission")
            return False

        # Verify final step
        if not self._validate_final_step():
            return False

        self.state.submitted = True
        return True

    def _validate_final_step(self) -> bool:
        """Verifies all required checkboxes in the final step."""
        current_fields = self.get_current_step_fields()
        for field in current_fields:
            if field.required and not self.state.fields.get(field.name, False):
                self.state.errors.append(f"Required checkbox {field.name} not checked")
                return False
        return True

    def get_state_summary(self) -> Dict[str, Any]:
        """Gets comprehensive state summary."""
        return {
            "current_step": self.state.current_step,
            "total_steps": self.state.total_steps,
            "filled_fields": len(self.state.fields),
            "total_fields": sum(len(fields) for fields in self.form_definition.values()),
            "errors": self.state.errors.copy(),
            "submitted": self.state.submitted,
        }


# Enterprise Environment Aliases
HeadlessWebEnvironment = MockWebApp
FormNavigationEnvironment = MockWebApp
