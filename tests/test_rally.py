import pytest

from game_mechanics.game_manager import GameManager
from game_mechanics.rally import RallyGame


def test_correct_submission_extends_streak_and_advances_prompt():
    rally = RallyGame(words=["EAT", "DRINK"], round_seconds=60.0, prompt_timeout_seconds=4.0)
    first_prompt = rally.start(now=0.0)

    result = rally.submit(first_prompt, now=1.5)

    assert result["correct"] is True
    assert result["streak"] == 1
    assert result["best_streak"] == 1
    assert result["response_time_seconds"] == 1.5
    assert result["finished"] is False
    assert result["next_prompt"] in {"EAT", "DRINK"}
    assert rally.current_prompt == result["next_prompt"]


def test_wrong_submission_resets_streak_but_round_continues():
    rally = RallyGame(words=["EAT", "DRINK"], round_seconds=60.0, prompt_timeout_seconds=4.0)
    prompt = rally.start(now=0.0)
    rally.submit(prompt, now=1.0)  # streak -> 1

    result = rally.submit("NOT-THE-PROMPT", now=2.0)

    assert result["correct"] is False
    assert result["streak"] == 0
    assert result["best_streak"] == 1  # best streak is remembered
    assert result["finished"] is False
    assert rally.current_prompt is not None


def test_check_timeout_registers_a_miss_and_moves_on():
    rally = RallyGame(words=["EAT", "DRINK"], round_seconds=60.0, prompt_timeout_seconds=2.0)
    rally.start(now=0.0)

    assert rally.check_timeout(now=1.0) is None  # not yet expired

    result = rally.check_timeout(now=2.5)

    assert result is not None
    assert result["timed_out"] is True
    assert result["correct"] is False
    assert result["misses"] == 1
    assert rally.current_prompt is not None  # a new prompt was queued


def test_round_finishes_when_time_runs_out():
    rally = RallyGame(words=["EAT", "DRINK"], round_seconds=5.0, prompt_timeout_seconds=10.0)
    prompt = rally.start(now=0.0)

    result = rally.submit(prompt, now=6.0)

    assert result["finished"] is True
    assert result["finish_reason"] == "time_up"
    assert rally.current_prompt is None
    assert "next_prompt" not in result


def test_round_finishes_after_max_misses():
    rally = RallyGame(words=["EAT", "DRINK"], round_seconds=600.0, prompt_timeout_seconds=10.0, max_misses=2)
    rally.start(now=0.0)

    rally.submit("WRONG", now=1.0)
    result = rally.submit("WRONG", now=2.0)

    assert result["finished"] is True
    assert result["finish_reason"] == "out_of_lives"


def test_submit_without_start_raises():
    rally = RallyGame(words=["EAT", "DRINK"])
    with pytest.raises(ValueError):
        rally.submit("EAT")


def test_summary_reflects_recorded_attempts():
    rally = RallyGame(words=["EAT", "DRINK"], round_seconds=60.0, prompt_timeout_seconds=4.0)
    prompt = rally.start(now=0.0)
    rally.submit(prompt, now=1.0)
    rally.submit("WRONG", now=2.0)

    summary = rally.summary()

    assert summary["total_attempts"] == 2
    assert summary["correct_attempts"] == 1
    assert summary["accuracy_percent"] == 50.0
    assert summary["best_streak"] == 1
    assert summary["final_streak"] == 0


def test_game_manager_process_rally_sign_updates_progress():
    manager = GameManager()
    prompt = manager.start_rally_session(words=["EAT", "DRINK"], round_seconds=60.0, prompt_timeout_seconds=4.0).current_prompt

    result = manager.process_rally_sign(prompt)

    assert result["correct"] is True
    assert manager.progress.total_attempts == 1
    assert manager.progress.total_correct == 1


def test_game_manager_process_rally_sign_requires_active_session():
    manager = GameManager()
    with pytest.raises(ValueError):
        manager.process_rally_sign("EAT")


def test_game_manager_check_rally_timeout_without_session_returns_none():
    manager = GameManager()
    assert manager.check_rally_timeout() is None
