import pandas as pd
import pytest

from src.curator.playlist_agent import PlaylistCurator, parse_prompt
from src.mood.mood_tagger import Mood


@pytest.fixture
def tracks_df():
    rows = []
    artists = ["Artist A", "Artist B", "Artist C", "Artist D"]
    genres = ["pop", "rock"]
    for i in range(40):
        rows.append(
            {
                "track_id": f"t{i}",
                "artists": artists[i % len(artists)],
                "track_name": f"Track {i}",
                "track_genre": genres[i % len(genres)],
                "valence": 0.1 + (i % 10) * 0.09,
                "energy": 0.1 + (i % 7) * 0.13,
                "tempo": 80 + (i % 5) * 20,
                "key": i % 12,
                "mode": i % 2,
                "popularity": (i * 7) % 100,
            }
        )
    return pd.DataFrame(rows)


def test_requires_columns():
    with pytest.raises(ValueError):
        PlaylistCurator(pd.DataFrame({"track_id": [1]}))


def test_genre_filter_applied(tracks_df):
    curator = PlaylistCurator(tracks_df)
    playlist = curator.generate_playlist(genre="rock", n=10)
    assert len(playlist) > 0
    assert all(t.genre.lower() == "rock" for t in playlist)


def test_unknown_genre_falls_back_to_full_pool(tracks_df):
    curator = PlaylistCurator(tracks_df)
    playlist = curator.generate_playlist(genre="nonexistent-genre", n=10)
    assert len(playlist) == 10


def test_no_back_to_back_same_artist(tracks_df):
    curator = PlaylistCurator(tracks_df)
    playlist = curator.generate_playlist(genre="pop", n=15)
    artists = [t.artist for t in playlist]
    for a, b in zip(artists, artists[1:]):
        assert a != b


def test_respects_requested_length(tracks_df):
    curator = PlaylistCurator(tracks_df)
    playlist = curator.generate_playlist(genre="pop", n=5)
    assert len(playlist) == 5


def test_prompt_parsing_extracts_genre_and_mood():
    genre, mood = parse_prompt("give me some upbeat happy pop", known_genres=["pop", "rock"])
    assert genre == "pop"
    assert mood == Mood.HAPPY


def test_prompt_with_no_genre_returns_none_genre():
    genre, mood = parse_prompt("something relaxed and mellow", known_genres=["pop", "rock"])
    assert genre is None
    assert mood == Mood.RELAXED


def test_resolve_request_prefers_explicit_over_prompt(tracks_df):
    curator = PlaylistCurator(tracks_df)
    genre, mood = curator.resolve_request(genre="rock", mood="sad", prompt="upbeat happy pop")
    assert genre == "rock"
    assert mood == Mood.SAD


def test_parse_prompt_prefers_real_genre_over_mood_named_genre():
    """The Spotify Tracks Dataset has genres literally named "happy"/"sad"/"chill",
    which must not shadow the actual genre named in the prompt."""
    genre, mood = parse_prompt("upbeat happy pop", known_genres=["happy", "pop", "sad"])
    assert genre == "pop"
    assert mood is Mood.HAPPY


def test_parse_prompt_falls_back_to_mood_named_genre_when_its_the_only_match():
    genre, mood = parse_prompt("happy songs", known_genres=["happy", "pop", "sad"])
    assert genre == "happy"
    assert mood is Mood.HAPPY
