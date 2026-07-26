from litert_studio.training.fixtures import SMOKE_MODEL_ID, SMOKE_MODEL_REVISION


def test_smoke_fixture_is_revision_pinned() -> None:
    assert SMOKE_MODEL_ID == "fxmarty/tiny-random-GemmaForCausalLM"
    assert len(SMOKE_MODEL_REVISION) == 40
    assert all(character in "0123456789abcdef" for character in SMOKE_MODEL_REVISION)
