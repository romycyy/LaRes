"""Research-idea ledger, protocol checks, advisor calls and the generated state page.

One level above :mod:`lares.search`: ideas about the *method*, each with a
falsifiable prediction, a predeclared protocol, the user's approval, a result,
and two independent verdicts that must agree before a decision is recorded.
Driven by ``scripts/research.py``.
"""

from lares.research.schemas import (  # noqa: F401
    Decision,
    Prediction,
    Protocol,
    ResearchIdea,
    ResearchSchemaError,
    Verdict,
    decide,
    find_idea,
    load_decisions,
    load_ideas,
    next_idea_id,
)
from lares.research.protocol import check_idea, start_run  # noqa: F401
from lares.research.briefing import build_briefing  # noqa: F401
from lares.research.state import render_state, write_state  # noqa: F401
from lares.research.advisor import Advisor, openai_call  # noqa: F401
