from metric import generation


def test_generation_tokenizer_falls_back_without_nltk_punkt(monkeypatch):
    def _missing_punkt(_path):
        raise LookupError("punkt is not installed")

    monkeypatch.setattr(generation, "nltk_load", _missing_punkt)

    tokenizer = generation.Tokenizer_gen("en")

    assert tokenizer.tokenize("Clinical text, works.") == [
        "Clinical",
        "text",
        ",",
        "works",
        ".",
    ]
