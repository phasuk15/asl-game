import csv

from game_mechanics.study_session import StudySession


def test_reset_for_next_player_saves_and_clears_state(tmp_path):
    study = StudySession(results_dir=tmp_path)

    manager = study.start_participant("P001")
    lesson = manager.tutorial.current_lesson()
    manager.tutorial.mark_completed(lesson)
    manager.progress.total_correct += 1
    manager.progress.total_attempts += 1
    study.record_tutorial_result(lesson.word, True, 2.5)

    study.record_wordle_guess(manager.wordle.target_word, manager.wordle.target_word, True, 4.0)

    assert study.is_active is True
    summary = study.reset_for_next_player()

    assert study.is_active is False
    assert study.manager is None
    assert summary["participant_id"] == "P001"
    assert summary["tutorial_lessons_completed"] == 1
    assert summary["tutorial_avg_response_time_seconds"] == 2.5
    assert summary["wordle_attempts"] == 1
    assert summary["wordle_correct_guesses"] == 1
    assert summary["accuracy_percent"] == 100.0

    log_path = tmp_path / "performance_log.csv"
    assert log_path.exists()
    with log_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["participant_id"] == "P001"

    participant_files = list((tmp_path / "participants").glob("P001_*.json"))
    assert len(participant_files) == 1


def test_starting_a_new_participant_auto_saves_and_does_not_carry_over_progress(tmp_path):
    study = StudySession(results_dir=tmp_path)

    first_manager = study.start_participant("P001")
    first_manager.tutorial.mark_completed(first_manager.tutorial.current_lesson())

    second_manager = study.start_participant("P002")

    assert second_manager is not first_manager
    assert second_manager.tutorial.current_lesson().completed is False
    assert study.participant_id == "P002"

    log_path = tmp_path / "performance_log.csv"
    with log_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["participant_id"] for row in rows] == ["P001"]


def test_auto_generated_participant_ids_are_unique(tmp_path):
    study = StudySession(results_dir=tmp_path)

    study.start_participant("")
    first_id = study.participant_id
    study.start_participant(None)
    second_id = study.participant_id

    assert first_id is not None
    assert first_id != second_id


def test_end_participant_without_an_active_session_returns_none(tmp_path):
    study = StudySession(results_dir=tmp_path)

    assert study.end_participant() is None
    assert study.reset_for_next_player() is None


def test_rally_attempts_are_logged_and_summarised(tmp_path):
    study = StudySession(results_dir=tmp_path)

    manager = study.start_participant("P001")
    prompt = manager.start_rally_session(words=["EAT", "DRINK"], round_seconds=60.0, prompt_timeout_seconds=4.0).current_prompt

    correct_result = manager.process_rally_sign(prompt)
    study.record_rally_attempt(
        target=correct_result["prompt"],
        submitted=correct_result["submitted"],
        correct=correct_result["correct"],
        response_time_seconds=correct_result["response_time_seconds"],
    )
    wrong_result = manager.process_rally_sign("NOT-THE-PROMPT")
    study.record_rally_attempt(
        target=wrong_result["prompt"],
        submitted=wrong_result["submitted"],
        correct=wrong_result["correct"],
        response_time_seconds=wrong_result["response_time_seconds"],
    )

    summary = study.reset_for_next_player()

    assert summary["rally_attempts"] == 2
    assert summary["rally_correct_attempts"] == 1
    assert summary["rally_best_streak"] == 1


def test_fingerspelling_results_are_logged_and_summarised(tmp_path):
    study = StudySession(results_dir=tmp_path)

    manager = study.start_participant("P001")
    lesson = manager.fingerspelling.current_lesson()
    manager.fingerspelling.mark_completed(lesson)
    study.record_fingerspelling_result(lesson.word, True, 1.2)

    next_lesson = manager.fingerspelling.current_lesson()
    study.record_fingerspelling_result(next_lesson.word, False, 3.0)

    summary = study.reset_for_next_player()

    assert summary["fingerspelling_lessons_completed"] == 1
    assert summary["fingerspelling_total_lessons"] == 26
    assert summary["fingerspelling_avg_response_time_seconds"] == 2.1
    assert summary["fingerspelling_complete"] is False
