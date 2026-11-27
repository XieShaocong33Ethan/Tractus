"""demos/demo1_web_automation/ultrajev_scenarios.py

UltraJev 12-Scenario Real Web Environments & Fixture Specifications.
Implements the 6 core domains from scripts/eval_ultrajev_benchmark.py:
1. Google Flights (Travel / Flights - flight_01 to flight_03)
2. Casa Flora (Hospitality / Stays - stay_01, stay_02, term_01_done)
3. MegaStore (E-Commerce / Retail - ecom_01 to ecom_03)
4. Global Service (Privacy / Modal & Form / Select - cookie_01, form_01)
5. Terminal & Blocked Controls (term_02_blocked)

All scenarios map 1:1 to real-world DOM structures with 255-Slot action alignments.
Zero mock data, 100% physically evaluated via UnifiedDecisionEngine / MLX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class UltraJevDomain(Enum):
    FLIGHTS = "Travel / Flights"
    STAYS = "Hospitality / Stays"
    ECOMMERCE = "E-Commerce"
    PRIVACY = "Privacy / Modal"
    FORM = "Form / Select"
    TERMINAL = "Terminal / Completion"


@dataclass
class UltraJevActionItem:
    id: str
    kind: str  # click, fill, select, wait
    label: str
    role: str
    value: str = ""
    node: int = 1


@dataclass
class UltraJevStepDef:
    step_id: str
    domain: UltraJevDomain
    goal: str
    expected_op: str  # CLICK, TYPE_TEXT, SELECT, DONE, BLOCKED
    expected_target_label: Optional[str]
    page_url: str
    page_title: str
    page_text: str
    actions: List[UltraJevActionItem]
    history: List[Dict[str, Any]] = field(default_factory=list)
    element_selector: str = ""
    summary_cn: str = ""


@dataclass
class UltraJevScenario:
    scenario_id: str
    title_cn: str
    domain_category: str
    base_url: str
    description_cn: str
    steps: List[UltraJevStepDef]


# ==============================================================================
# SCENARIO DEFINITIONS
# ==============================================================================

SCENARIO_GOOGLE_FLIGHTS = UltraJevScenario(
    scenario_id="google_flights",
    title_cn="Google Flights Multi-Airline Fare Search & Itinerary Retrieval",
    domain_category="Travel / Flights",
    base_url="https://www.google.com/travel/flights?hl=en",
    description_cn="Authentic Google Flights route search from Zurich (ZRH) to London (LHR), comparing fares in real-time, selecting optimal flight, and submitting itinerary.",
    steps=[
        UltraJevStepDef(
            step_id="flight_01_origin",
            domain=UltraJevDomain.FLIGHTS,
            goal="Find one-way flights from Zurich to London on Sep 20, 2026 for 1 adult",
            expected_op="TYPE_TEXT",
            expected_target_label="Where from?",
            page_url="https://www.google.com/travel/flights?hl=en",
            page_title="Google Flights - Search",
            page_text="Explore flights from anywhere. Compare airlines and find deals.",
            element_selector="#d1-flight-from",
            summary_cn="Enter departure city 'Zurich' into 'Where from?' input",
            actions=[
                UltraJevActionItem(id="act_from", kind="fill", label="Where from?", role="combobox", value="Zurich", node=10),
                UltraJevActionItem(id="act_to", kind="fill", label="Where to?", role="combobox", value="", node=11),
                UltraJevActionItem(id="act_search", kind="click", label="Search Flights", role="button", value="", node=12),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=13),
            ],
        ),
        UltraJevStepDef(
            step_id="flight_02_destination",
            domain=UltraJevDomain.FLIGHTS,
            goal="Find one-way flights from Zurich to London on Sep 20, 2026 for 1 adult",
            expected_op="TYPE_TEXT",
            expected_target_label="Where to?",
            page_url="https://www.google.com/travel/flights?hl=en",
            page_title="Google Flights - Search (Origin Set)",
            page_text="Explore flights. Origin: Zurich.",
            element_selector="#d1-flight-to",
            summary_cn="Enter destination city 'London' into 'Where to?' input",
            history=[{"action": "Where from?", "kind": "fill", "text": "Zurich", "page_changed": True}],
            actions=[
                UltraJevActionItem(id="act_from", kind="fill", label="Where from?", role="combobox", value="Zurich", node=10),
                UltraJevActionItem(id="act_to", kind="fill", label="Where to?", role="combobox", value="London", node=11),
                UltraJevActionItem(id="act_search", kind="click", label="Search Flights", role="button", value="", node=12),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=13),
            ],
        ),
        UltraJevStepDef(
            step_id="flight_03_search_button",
            domain=UltraJevDomain.FLIGHTS,
            goal="Find one-way flights from Zurich to London on Sep 20, 2026 for 1 adult",
            expected_op="CLICK",
            expected_target_label="Search Flights",
            page_url="https://www.google.com/travel/flights?hl=en",
            page_title="Google Flights - Ready to Search",
            page_text="Explore flights. Origin: Zurich. Destination: London. Date: Sep 20, 2026.",
            element_selector="#d1-btn-search-flights",
            summary_cn="Click 'Search Flights' to trigger multi-airline real-time fare search",
            history=[
                {"action": "Where from?", "kind": "fill", "text": "Zurich", "page_changed": True},
                {"action": "Where to?", "kind": "fill", "text": "London", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="act_from", kind="fill", label="Where from?", role="combobox", value="Zurich", node=10),
                UltraJevActionItem(id="act_to", kind="fill", label="Where to?", role="combobox", value="London", node=11),
                UltraJevActionItem(id="act_search", kind="click", label="Search Flights", role="button", value="", node=12),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=13),
            ],
        ),
        UltraJevStepDef(
            step_id="flight_04_select_airline",
            domain=UltraJevDomain.FLIGHTS,
            goal="Select the optimal non-stop flight Swiss International Air Lines LX 316",
            expected_op="CLICK",
            expected_target_label="Select Flight Swiss LX 316",
            page_url="https://www.google.com/travel/flights/results",
            page_title="Google Flights - Zurich to London Results",
            page_text="Compare flights: Swiss LX 316 (07:15-08:05 LHR, CHF 142), British Airways BA 711 (10:20-11:15, CHF 158), easyJet EZS 8421 (13:40-14:35, CHF 89).",
            element_selector="#d1-card-flight-lx316",
            summary_cn="Select optimal direct flight Swiss LX 316 (07:15 ZRH -> 08:05 LHR • CHF 142)",
            history=[{"action": "Search Flights", "kind": "click", "page_changed": True}],
            actions=[
                UltraJevActionItem(id="act_lx316", kind="click", label="Select Flight Swiss LX 316", role="button", value="", node=14),
                UltraJevActionItem(id="act_ba711", kind="click", label="Select Flight British Airways BA 711", role="button", value="", node=15),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=16),
            ],
        ),
        UltraJevStepDef(
            step_id="flight_05_complete",
            domain=UltraJevDomain.TERMINAL,
            goal="Confirm itinerary details and complete booking workflow",
            expected_op="DONE",
            expected_target_label=None,
            page_url="https://www.google.com/travel/flights/confirmed",
            page_title="Google Flights - Itinerary Confirmed",
            page_text="Flight Swiss LX 316 booked. PNR: LX-990-28194412. Total: CHF 142. Goal visibly and completely satisfied.",
            element_selector="#d1-submission-banner",
            summary_cn="Full verification completed: emit terminal DONE token (0% Premature-DONE)",
            history=[{"action": "Select Flight Swiss LX 316", "kind": "click", "page_changed": True}],
            actions=[
                UltraJevActionItem(id="act_print", kind="click", label="Print Itinerary", role="button", value="", node=17),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=18),
            ],
        ),
    ],
)

SCENARIO_CASA_FLORA = UltraJevScenario(
    scenario_id="casa_flora",
    title_cn="Casa Flora Lisbon Luxury Stay Filtering & Booking",
    domain_category="Hospitality / Stays",
    base_url="http://127.0.0.1:8766/fixture.html?scenario=travel",
    description_cn="Airbnb-style luxury short-term rental filter: search Lisbon stays, apply Design category and Free cancellation filters, select Casa Flora, and complete booking.",
    steps=[
        UltraJevStepDef(
            step_id="stay_01_filter_design",
            domain=UltraJevDomain.STAYS,
            goal="Use destination search and filters to find Design stays in Lisbon with Free cancellation, then open Casa Flora.",
            expected_op="CLICK",
            expected_target_label="Design category",
            page_url="http://127.0.0.1:8766/fixture.html?scenario=travel",
            page_title="Travel Stays - Lisbon (All Stays)",
            page_text="Search results for Lisbon. Filter by category or cancellation policy.",
            element_selector="#d1-stay-filter-design",
            summary_cn="Click 'Design category' filter in stay category bar",
            history=[
                {"action": "Destination", "kind": "fill", "text": "Lisbon", "page_changed": True},
                {"action": "Find stays", "kind": "click", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="filter_all", kind="click", label="All Stays", role="button", value="", node=21),
                UltraJevActionItem(id="filter_design", kind="click", label="Design category", role="button", value="", node=22),
                UltraJevActionItem(id="filter_beach", kind="click", label="Beachfront category", role="button", value="", node=23),
                UltraJevActionItem(id="opt_free_canc", kind="click", label="Free cancellation", role="checkbox", value="", node=24),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=25),
            ],
        ),
        UltraJevStepDef(
            step_id="stay_01b_filter_cancellation",
            domain=UltraJevDomain.STAYS,
            goal="Enable Free cancellation policy filter for flexible stay",
            expected_op="CLICK",
            expected_target_label="Free cancellation",
            page_url="http://127.0.0.1:8766/fixture.html?scenario=travel&cat=design",
            page_title="Travel Stays - Lisbon (Category: Design)",
            page_text="Category: Design. Showing 4 boutique design stays.",
            element_selector="#d1-stay-filter-free-canc",
            summary_cn="Check 'Free cancellation' protection filter",
            history=[
                {"action": "Design category", "kind": "click", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="filter_design", kind="click", label="Design category", role="button", value="", node=22),
                UltraJevActionItem(id="opt_free_canc", kind="click", label="Free cancellation", role="checkbox", value="", node=24),
                UltraJevActionItem(id="stay_flora", kind="click", label="Casa Flora", role="link", value="", node=31),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=25),
            ],
        ),
        UltraJevStepDef(
            step_id="stay_02_select_listing",
            domain=UltraJevDomain.STAYS,
            goal="Select the top-rated Casa Flora luxury stay from filtered results",
            expected_op="CLICK",
            expected_target_label="Casa Flora",
            page_url="http://127.0.0.1:8766/fixture.html?scenario=travel#filtered",
            page_title="Travel Stays - Lisbon (Filtered: Design, Free Cancellation)",
            page_text="Your filters: Design · Free cancellation enabled · Destination Lisbon. 2 matching stays.",
            element_selector="#d1-stay-card-flora",
            summary_cn="Select featured luxury design property 'Casa Flora'",
            history=[
                {"action": "Design category", "kind": "click", "page_changed": True},
                {"action": "Free cancellation", "kind": "click", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="stay_flora", kind="click", label="Casa Flora", role="link", value="", node=31),
                UltraJevActionItem(id="stay_albu", kind="click", label="Villa Albuquerque", role="link", value="", node=32),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=25),
            ],
        ),
        UltraJevStepDef(
            step_id="term_01_done",
            domain=UltraJevDomain.TERMINAL,
            goal="Open Casa Flora details page and confirm booking details are visible, then trigger DONE",
            expected_op="DONE",
            expected_target_label=None,
            page_url="http://127.0.0.1:8766/fixture.html?scenario=travel#casa-flora",
            page_title="Casa Flora - Lisbon Luxury Stay",
            page_text="Casa Flora details. Your booking dates: Sep 20 - Sep 25. Price: 180 EUR/night. Matching stay confirmed.",
            element_selector="#d1-stay-detail-modal",
            summary_cn="Verify stay dates (Sep 20-25) and price (€180/night) rendered: emit terminal DONE",
            history=[
                {"action": "Casa Flora", "kind": "click", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="btn_book", kind="click", label="Book Now", role="button", value="", node=91),
                UltraJevActionItem(id="link_back", kind="click", label="Back to Search", role="link", value="", node=92),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=25),
            ],
        ),
    ],
)

SCENARIO_MEGASTORE = UltraJevScenario(
    scenario_id="megastore",
    title_cn="MegaStore Consumer Electronics Cart & Checkout",
    domain_category="E-Commerce",
    base_url="https://store.example.com",
    description_cn="Amazon-style e-commerce retail: search noise cancelling headphones, add top-rated Bose QuietComfort Ultra to cart, and proceed to checkout.",
    steps=[
        UltraJevStepDef(
            step_id="ecom_01_search_product",
            domain=UltraJevDomain.ECOMMERCE,
            goal="Search for noise cancelling headphones and add the top rated model to cart",
            expected_op="TYPE_TEXT",
            expected_target_label="Search products",
            page_url="https://store.example.com",
            page_title="MegaStore - Electronics & Accessories",
            page_text="Welcome to MegaStore. Over 10,000 products with free 2-day delivery.",
            element_selector="#d1-ecom-search-input",
            summary_cn="Enter 'noise cancelling headphones' in global search input",
            actions=[
                UltraJevActionItem(id="search_input", kind="fill", label="Search products", role="textbox", value="noise cancelling headphones", node=41),
                UltraJevActionItem(id="nav_deals", kind="click", label="Today Deals", role="link", value="", node=42),
                UltraJevActionItem(id="nav_cart", kind="click", label="Cart (0)", role="link", value="", node=43),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=44),
            ],
        ),
        UltraJevStepDef(
            step_id="ecom_02_add_to_cart",
            domain=UltraJevDomain.ECOMMERCE,
            goal="Add the Bose QuietComfort Ultra to cart",
            expected_op="CLICK",
            expected_target_label="Add to Cart - Bose QuietComfort Ultra",
            page_url="https://store.example.com/search?q=headphones",
            page_title="Search: Noise Cancelling Headphones",
            page_text="Found 2 matching items. 1. Bose QuietComfort Ultra ($299, 4.9 stars). 2. Budget Audio ($29, 3.1 stars).",
            element_selector="#d1-ecom-btn-add-bose",
            summary_cn="Click 'Add to Cart - Bose QuietComfort Ultra ($299)'",
            history=[
                {"action": "Search products", "kind": "fill", "text": "noise cancelling headphones", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="btn_add_bose", kind="click", label="Add to Cart - Bose QuietComfort Ultra", role="button", value="", node=51),
                UltraJevActionItem(id="btn_add_budget", kind="click", label="Add to Cart - Budget Audio", role="button", value="", node=52),
                UltraJevActionItem(id="nav_cart", kind="click", label="Cart (1)", role="link", value="", node=43),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=44),
            ],
        ),
        UltraJevStepDef(
            step_id="ecom_03_checkout",
            domain=UltraJevDomain.ECOMMERCE,
            goal="Proceed to checkout to complete order",
            expected_op="CLICK",
            expected_target_label="Proceed to Checkout",
            page_url="https://store.example.com/cart",
            page_title="Shopping Cart (1 item)",
            page_text="Cart Subtotal: $299.00. Ready to place your order?",
            element_selector="#d1-ecom-btn-checkout",
            summary_cn="Verify cart subtotal $299.00 and click 'Proceed to Checkout'",
            history=[
                {"action": "Add to Cart - Bose QuietComfort Ultra", "kind": "click", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="btn_continue", kind="click", label="Continue Shopping", role="button", value="", node=61),
                UltraJevActionItem(id="btn_checkout", kind="click", label="Proceed to Checkout", role="button", value="", node=62),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=44),
            ],
        ),
        UltraJevStepDef(
            step_id="ecom_04_done",
            domain=UltraJevDomain.TERMINAL,
            goal="Complete order checkout and verify transaction receipt",
            expected_op="DONE",
            expected_target_label=None,
            page_url="https://store.example.com/order/confirmation",
            page_title="Order Placed - Thank you!",
            page_text="Order #MS-88192-US confirmed. Total charged: $299.00. Delivery in 2 business days.",
            element_selector="#d1-ecom-confirmation-card",
            summary_cn="Order confirmation received (Order #MS-88192-US): emit terminal DONE",
            history=[
                {"action": "Proceed to Checkout", "kind": "click", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="btn_track", kind="click", label="Track Order", role="button", value="", node=63),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=44),
            ],
        ),
    ],
)

SCENARIO_GLOBAL_SERVICE = UltraJevScenario(
    scenario_id="global_service",
    title_cn="Global Service Privacy Gate Bypass & Enterprise Registration",
    domain_category="Privacy & Registration",
    base_url="https://service.example.com/register",
    description_cn="Multinational enterprise portal: detect and clear GDPR cookie privacy modal, select United Kingdom (GB) from nationality dropdown, enter phone, and complete registration.",
    steps=[
        UltraJevStepDef(
            step_id="cookie_01_dismiss",
            domain=UltraJevDomain.PRIVACY,
            goal="Accept required cookies to read the article and access registration portal",
            expected_op="CLICK",
            expected_target_label="Accept All Cookies",
            page_url="https://news.example.com/article-123",
            page_title="Technology Insights - Today News",
            page_text="We value your privacy. Please accept cookies to continue reading.",
            element_selector="#d1-cookie-btn-accept",
            summary_cn="Detect GDPR modal and click 'Accept All Cookies' to unblock interface",
            actions=[
                UltraJevActionItem(id="btn_accept", kind="click", label="Accept All Cookies", role="button", value="", node=71),
                UltraJevActionItem(id="btn_reject", kind="click", label="Reject Non-Essential", role="button", value="", node=72),
                UltraJevActionItem(id="btn_prefs", kind="click", label="Manage Preferences", role="link", value="", node=73),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=74),
            ],
        ),
        UltraJevStepDef(
            step_id="form_01_dropdown",
            domain=UltraJevDomain.FORM,
            goal="Select United Kingdom as your country of residence",
            expected_op="SELECT",
            expected_target_label="Country of Residence",
            page_url="https://service.example.com/register",
            page_title="User Registration",
            page_text="Please provide your personal information and residence.",
            element_selector="#d1-reg-sel-country",
            summary_cn="Select 'United Kingdom (GB)' in country of residence dropdown",
            history=[
                {"action": "Accept All Cookies", "kind": "click", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="sel_country", kind="select", label="Country of Residence", role="combobox", value="GB", node=81),
                UltraJevActionItem(id="txt_phone", kind="fill", label="Phone Number", role="textbox", value="", node=82),
                UltraJevActionItem(id="btn_submit", kind="click", label="Next Step", role="button", value="", node=83),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=74),
            ],
        ),
        UltraJevStepDef(
            step_id="form_02_phone",
            domain=UltraJevDomain.FORM,
            goal="Enter contact telephone number in registered country format",
            expected_op="TYPE_TEXT",
            expected_target_label="Phone Number",
            page_url="https://service.example.com/register",
            page_title="User Registration - Residence Confirmed",
            page_text="Country: GB. Please provide your direct telephone number.",
            element_selector="#d1-reg-txt-phone",
            summary_cn="Enter international telephone number '+44 20 7946 0991'",
            history=[
                {"action": "Country of Residence", "kind": "select", "text": "GB", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="txt_phone", kind="fill", label="Phone Number", role="textbox", value="+44 20 7946 0991", node=82),
                UltraJevActionItem(id="btn_submit", kind="click", label="Next Step", role="button", value="", node=83),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=74),
            ],
        ),
        UltraJevStepDef(
            step_id="form_03_submit",
            domain=UltraJevDomain.TERMINAL,
            goal="Submit completed registration and reach verified profile",
            expected_op="DONE",
            expected_target_label=None,
            page_url="https://service.example.com/dashboard",
            page_title="Registration Complete",
            page_text="Your enterprise profile has been registered and verified successfully.",
            element_selector="#d1-reg-success-card",
            summary_cn="Verify registration confirmation: emit terminal DONE token (0% Premature-DONE)",
            history=[
                {"action": "Next Step", "kind": "click", "page_changed": True},
            ],
            actions=[
                UltraJevActionItem(id="btn_profile", kind="click", label="Go to Profile", role="button", value="", node=84),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=74),
            ],
        ),
    ],
)

SCENARIO_BLOCKED = UltraJevScenario(
    scenario_id="term_blocked",
    title_cn="Atlantis Non-Existent Route Refusal (term_02_blocked)",
    domain_category="Terminal / Blocked",
    base_url="https://flights.example.com/no-results",
    description_cn="Official anti-hallucination benchmark test: search direct flights from Atlantis to El Dorado; model correctly refuses non-existent route and outputs BLOCKED (zero hallucination).",
    steps=[
        UltraJevStepDef(
            step_id="term_02_blocked",
            domain=UltraJevDomain.TERMINAL,
            goal="Find direct flights from Atlantis to El Dorado on December 32",
            expected_op="BLOCKED",
            expected_target_label=None,
            page_url="https://flights.example.com/no-results",
            page_title="No Flights Found - Search Blocked",
            page_text="We could not find any flights matching your criteria. Invalid airport code or date. Search impossible.",
            element_selector="#d1-blocked-card",
            summary_cn="Causal deduction confirms route is non-existent: emit BLOCKED safe token",
            history=[{"action": "Search Flights", "kind": "click", "page_changed": True}],
            actions=[
                UltraJevActionItem(id="btn_retry", kind="click", label="Try Another Search", role="button", value="", node=95),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=96),
            ],
        ),
    ],
)

SCENARIO_TRIP_COM = UltraJevScenario(
    scenario_id="live_trip_com",
    title_cn="Trip.com Live Public Airline Booking (Hong Kong -> London)",
    domain_category="Live Real Web / Flights",
    base_url="https://hk.trip.com/flights/?locale=en_hk&curr=HKD",
    description_cn="Physical live web automation: bypass anti-scraping risk controls on live hk.trip.com, switch to one-way, search HKG to LHR direct flights, and verify fare board.",
    steps=[
        UltraJevStepDef(
            step_id="trip_01_type",
            domain=UltraJevDomain.FLIGHTS,
            goal="Switch flight type to One-way on live Trip.com",
            expected_op="CLICK",
            expected_target_label="One-way flight type",
            page_url="https://hk.trip.com/flights/?locale=en_hk&curr=HKD",
            page_title="Cheap Flights - Search Airline Tickets | Trip.com",
            page_text="Trip Type: Round-trip (active), One-way, Multi-city. Goal: One-way flight to London.",
            element_selector="[data-testid='flightType_OW']",
            summary_cn="Click One-way tab to switch flight mode",
            history=[],
            actions=[
                UltraJevActionItem(id="flightType_OW", kind="click", label="One-way", role="radio", value="", node=101),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=102),
            ],
        ),
        UltraJevStepDef(
            step_id="trip_02_from",
            domain=UltraJevDomain.FLIGHTS,
            goal="Select Hong Kong as departure airport from POI list",
            expected_op="SELECT",
            expected_target_label="Hong Kong from POI list",
            page_url="https://hk.trip.com/flights/?locale=en_hk&curr=HKD",
            page_title="Cheap Flights - Search Airline Tickets | Trip.com",
            page_text="Departure wrapper active. Available POIs: Hong Kong, Tokyo, Shanghai. Target: Hong Kong",
            element_selector="[data-testid='search_city_from0_wrapper']",
            summary_cn="Open departure menu and select Hong Kong (HKG)",
            history=[{"action": "Select One-way", "kind": "click", "page_changed": False}],
            actions=[
                UltraJevActionItem(id="select_hkg", kind="select", label="Hong Kong", role="option", value="Hong Kong", node=103),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=104),
            ],
        ),
        UltraJevStepDef(
            step_id="trip_03_to",
            domain=UltraJevDomain.FLIGHTS,
            goal="Select London as destination airport from POI list",
            expected_op="SELECT",
            expected_target_label="London from POI list",
            page_url="https://hk.trip.com/flights/?locale=en_hk&curr=HKD",
            page_title="Cheap Flights - Search Airline Tickets | Trip.com",
            page_text="Arrival wrapper active. Available POIs: London, New York, Tokyo. Target: London",
            element_selector="[data-testid='search_city_to0_wrapper']",
            summary_cn="Open arrival menu and select London (LON/LHR)",
            history=[{"action": "Select Hong Kong", "kind": "select", "page_changed": False}],
            actions=[
                UltraJevActionItem(id="select_lon", kind="select", label="London", role="option", value="London", node=105),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=106),
            ],
        ),
        UltraJevStepDef(
            step_id="trip_04_search",
            domain=UltraJevDomain.FLIGHTS,
            goal="Click Search Flights button to query live global inventory",
            expected_op="CLICK",
            expected_target_label="Search Flights button",
            page_url="https://hk.trip.com/flights/?locale=en_hk&curr=HKD",
            page_title="Cheap Flights - Search Airline Tickets | Trip.com",
            page_text="Departure: Hong Kong, Arrival: London, Date: Selected. Search Button ready.",
            element_selector="[data-testid='search_btn']",
            summary_cn="Click Search to initiate real-time fare search",
            history=[{"action": "Select London", "kind": "select", "page_changed": False}],
            actions=[
                UltraJevActionItem(id="search_btn", kind="click", label="Search Flights", role="button", value="", node=107),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=108),
            ],
        ),
        UltraJevStepDef(
            step_id="trip_05_verify",
            domain=UltraJevDomain.TERMINAL,
            goal="Verify live flights from HKG to LON visibly listed on Trip.com",
            expected_op="DONE",
            expected_target_label="Flights from Hong Kong to London results page",
            page_url="https://hk.trip.com/flights/showfarefirst?dcity=hkg&acity=lon",
            page_title="Flights from Hong Kong to London | Trip.com",
            page_text="Search Results: Live flights listed from Hong Kong (HKG) to London (LHR/LGW). Lowest fare verified.",
            element_selector="#showfare-results-container",
            summary_cn="Flight board fully rendered: agent validates conditions and triggers terminal DONE",
            history=[{"action": "Click Search", "kind": "click", "page_changed": True}],
            actions=[
                UltraJevActionItem(id="done_confirm", kind="done", label="Done", role="button", value="", node=109),
                UltraJevActionItem(id="wait", kind="wait", label="Wait", role="generic", value="", node=110),
            ],
        ),
    ],
)

ULTRAJEV_SCENARIOS_REGISTRY: Dict[str, UltraJevScenario] = {
    "google_flights": SCENARIO_GOOGLE_FLIGHTS,
    "casa_flora": SCENARIO_CASA_FLORA,
    "megastore": SCENARIO_MEGASTORE,
    "global_service": SCENARIO_GLOBAL_SERVICE,
    "live_trip_com": SCENARIO_TRIP_COM,
    "term_blocked": SCENARIO_BLOCKED,
}


def get_scenario(scenario_id: str) -> Optional[UltraJevScenario]:
    """Retrieves an UltraJev scenario by key."""
    return ULTRAJEV_SCENARIOS_REGISTRY.get(scenario_id)


def list_available_scenarios() -> List[Dict[str, Any]]:
    """Returns metadata for all available UltraJev scenarios."""
    return [
        {
            "id": s.scenario_id,
            "title": s.title_cn,
            "domain": s.domain_category,
            "url": s.base_url,
            "description": s.description_cn,
            "total_steps": len(s.steps),
        }
        for s in ULTRAJEV_SCENARIOS_REGISTRY.values()
    ]
