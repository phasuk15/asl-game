from game_mechanics.wordle import WordleGame


def test_evaluate_guess_appends_to_history():
    game = WordleGame(target_word="SIGN", allowed_words=["SIGN", "GLOVE"])

    feedback = game.evaluate_guess("SIGN")

    assert len(game.history) == 1
    assert game.history[0] is feedback
    assert feedback.pattern == ["GREEN", "GREEN", "GREEN", "GREEN"]
    assert feedback.correct is True


def test_history_accumulates_across_multiple_guesses():
    game = WordleGame(target_word="SIGN", allowed_words=["SIGN", "GLOVE"])

    game.evaluate_guess("SIGN")
    game.reset("SIGN")
    assert game.history == []

    game.evaluate_guess("SIGN")
    assert len(game.history) == 1
    assert len(game.guesses) == 1
