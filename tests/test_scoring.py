from game_mechanics.game_manager import GameManager


def test_user_response_evaluation_tracks_accuracy_and_time():
    manager = GameManager()
    result = manager.evaluate_user_response(
        expected_answer="HELLO",
        submitted_answer="HELLO",
        started_at=10.0,
        completed_at=11.5,
    )

    assert result["accuracy_score"] == 1.0
    assert result["correct"] is True
    assert result["response_time_seconds"] == 1.5
    assert 0 <= result["performance_score"] <= 100
