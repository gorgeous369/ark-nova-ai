from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.helpers import (  # noqa: E402
    find_action,
    make_state,
    materialize_first_action,
    set_action_strength,
    take_card_by_number,
)


@pytest.fixture
def state_factory():
    return make_state


@pytest.fixture
def action_finder():
    return find_action


@pytest.fixture
def first_materialized_action():
    return materialize_first_action


@pytest.fixture
def action_strength_setter():
    return set_action_strength


@pytest.fixture
def card_picker():
    return take_card_by_number
