"""Unit tests for pydeck_click_handler — pure click-event parsing + dedup.

render_pydeck_map wraps a single st_deckgl() call; everything else (event
parsing, object-vs-terrain classification, dedup) is pure logic. We stub
st_deckgl to return canned deck.gl events and assert the parsed result.
"""

import skiresort_planner.ui.pydeck_click_handler as pch
from skiresort_planner.ui.pydeck_click_handler import PydeckClickResult, _get_click_id


class TestPydeckClickResult:
    def test_object_click_flags(self) -> None:
        r = PydeckClickResult(clicked_object={"type": "slope", "id": "SL1"}, clicked_coordinate=[10.0, 46.0])
        assert r.is_object_click and not r.is_terrain_click

    def test_terrain_click_flags(self) -> None:
        r = PydeckClickResult(clicked_object=None, clicked_coordinate=[10.0, 46.0])
        assert r.is_terrain_click and not r.is_object_click

    def test_empty_is_neither(self) -> None:
        r = PydeckClickResult.empty()
        assert not r.is_object_click and not r.is_terrain_click


class TestGetClickId:
    def test_object_with_type_and_id(self, fake_st) -> None:
        # Object ids fold in dedup_epoch (0 here) so a regeneration makes the same target re-clickable.
        fake_st.session_state["dedup_epoch"] = 0
        cid = _get_click_id(obj={"type": "slope", "id": "SL1"}, coord=None)
        assert cid == "slope_SL1_v0"

    def test_object_without_id_uses_position(self) -> None:
        cid = _get_click_id(obj={"type": "", "position": [10.123456, 46.654321]}, coord=None)
        assert cid.startswith("pos_10.123456_46.654321")

    def test_terrain_coord_rounded(self) -> None:
        cid = _get_click_id(obj=None, coord=[10.123456, 46.654321])
        assert cid == "coord_10.12346_46.65432"

    def test_object_and_coord_combined(self, fake_st) -> None:
        fake_st.session_state["dedup_epoch"] = 0
        cid = _get_click_id(obj={"type": "lift", "id": "L1"}, coord=[1.0, 2.0])
        assert cid == "lift_L1_v0_coord_1.00000_2.00000"

    def test_object_epoch_changes_id(self, fake_st) -> None:
        # A dedup_epoch bump changes the id → the SAME target counts as a fresh click after regen.
        fake_st.session_state["dedup_epoch"] = 7
        assert _get_click_id(obj={"type": "lift", "id": "L1"}, coord=None) == "lift_L1_v7"

    def test_no_data_is_empty(self) -> None:
        assert _get_click_id(obj=None, coord=None) == ""


class _StubDeckgl:
    """Callable replacement for st_deckgl returning a fixed event dict."""

    def __init__(self, event) -> None:
        self.event = event

    def __call__(self, *args: object, **kwargs: object):
        return self.event


class TestRenderPydeckMapParsing:
    """render_pydeck_map parses the raw st_deckgl event into a result."""

    def _run(self, fake_st, monkeypatch, event, key="k"):
        monkeypatch.setattr(pch, "st_deckgl", _StubDeckgl(event))
        return pch.render_pydeck_map(deck=object(), key=key, height=600)

    def test_no_event_is_empty(self, fake_st, monkeypatch) -> None:
        result = self._run(fake_st, monkeypatch, event=None)
        assert not result.is_object_click and not result.is_terrain_click

    def test_terrain_event_parsed(self, fake_st, monkeypatch) -> None:
        result = self._run(fake_st, monkeypatch, event={"coordinate": [10.5, 46.5], "eventType": "click"})
        assert result.is_terrain_click
        assert result.clicked_coordinate == [10.5, 46.5]

    def test_object_event_parsed(self, fake_st, monkeypatch) -> None:
        event = {
            "type": "slope",
            "id": "SL1",
            "position": [10.5, 46.5, 1.0],
            "coordinate": [10.5, 46.5],
            "eventType": "click",
        }
        result = self._run(fake_st, monkeypatch, event=event)
        assert result.is_object_click and not result.is_terrain_click
        assert result.clicked_coordinate == [10.5, 46.5]
        assert result.clicked_object is not None
        assert result.clicked_object["type"] == "slope"
        assert result.clicked_object["id"] == "SL1"
        assert result.clicked_object["position"] == [10.5, 46.5, 1.0]  # object fields survive
        assert "coordinate" not in result.clicked_object  # only coordinate stripped
        assert "eventType" not in result.clicked_object  # only eventType stripped

    def test_duplicate_click_deduplicated(self, fake_st, monkeypatch) -> None:
        event = {"type": "slope", "id": "SL1", "coordinate": [10.5, 46.5], "eventType": "click"}
        monkeypatch.setattr(pch, "st_deckgl", _StubDeckgl(event))

        first = pch.render_pydeck_map(deck=object(), key="dup", height=600)
        second = pch.render_pydeck_map(deck=object(), key="dup", height=600)  # same event again
        assert first.is_object_click
        assert not second.is_object_click and not second.is_terrain_click  # deduped
