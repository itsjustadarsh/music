from src.mood.mood_tagger import Mood, tag_mood


def test_high_valence_high_energy_is_happy_or_excited():
    result = tag_mood(valence=0.9, energy=0.9)
    assert result.mood in (Mood.HAPPY, Mood.EXCITED)


def test_low_valence_low_energy_is_sad_or_tired():
    result = tag_mood(valence=0.1, energy=0.1)
    assert result.mood in (Mood.SAD, Mood.TIRED)


def test_high_valence_low_energy_is_relaxed_or_content():
    result = tag_mood(valence=0.9, energy=0.1)
    assert result.mood in (Mood.RELAXED, Mood.CONTENT)


def test_low_valence_high_energy_is_tense_or_angry():
    result = tag_mood(valence=0.1, energy=0.9)
    assert result.mood in (Mood.TENSE, Mood.ANGRY)


def test_center_point_is_neutral():
    result = tag_mood(valence=0.5, energy=0.5)
    assert result.mood == Mood.NEUTRAL


def test_valence_and_energy_are_recentered_to_minus_one_one():
    result = tag_mood(valence=1.0, energy=1.0)
    assert result.valence == 1.0
    assert result.arousal == 1.0


def test_out_of_range_inputs_are_clamped():
    result = tag_mood(valence=1.5, energy=-0.5)
    assert -1.0 <= result.valence <= 1.0
    assert -1.0 <= result.arousal <= 1.0


def test_batch_matches_single():
    from src.mood.mood_tagger import tag_mood_batch

    records = [{"valence": 0.9, "energy": 0.9}, {"valence": 0.1, "energy": 0.1}]
    results = tag_mood_batch(records)
    assert results[0].mood == tag_mood(0.9, 0.9).mood
    assert results[1].mood == tag_mood(0.1, 0.1).mood
