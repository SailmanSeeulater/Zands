from zands.chunk import chunk_text, split_paragraphs


def para(i: int, n: int) -> str:
    """A paragraph of n words made of sentences of 10 words, tagged with i."""
    sentences = []
    for s in range(n // 10):
        sentences.append(" ".join(f"p{i}s{s}w{j}" for j in range(9)) + f" p{i}s{s}end.")
    return " ".join(sentences)


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    text = "First paragraph here.\n\nSecond paragraph here."
    assert chunk_text(text) == ["First paragraph here.\n\nSecond paragraph here."]


def test_single_newlines_are_paragraphs_when_no_blank_lines():
    assert split_paragraphs("one\ntwo\n\n") == ["one", "two"]
    assert split_paragraphs("one\nstill one\n\ntwo") == ["one still one", "two"]


def test_long_text_chunks_within_bounds():
    text = "\n\n".join(para(i, 100) for i in range(30))  # 3000 words
    chunks = chunk_text(text, min_words=300, max_words=800, overlap_words=40)
    assert len(chunks) >= 4
    for c in chunks:
        n = len(c.split())
        assert n <= 800
    for c in chunks[:-1]:
        assert len(c.split()) >= 300


def test_chunks_split_on_paragraph_boundaries():
    paras = [para(i, 100) for i in range(20)]
    chunks = chunk_text("\n\n".join(paras), overlap_words=0)
    # with no overlap, every paragraph of every chunk is an original, whole paragraph
    for c in chunks:
        for p in c.split("\n\n"):
            assert p in paras
    # and together they reproduce the text exactly once
    assert sum(len(c.split("\n\n")) for c in chunks) == len(paras)


def test_overlap_repeats_last_sentences():
    paras = [para(i, 100) for i in range(20)]
    chunks = chunk_text("\n\n".join(paras), overlap_words=40)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        first = nxt.split("\n\n")[0]
        assert len(first.split()) <= 40
        assert prev.endswith(first)


def test_huge_paragraph_is_split_by_sentence():
    text = para(0, 2000)  # one 2000-word paragraph
    chunks = chunk_text(text, overlap_words=0)
    assert len(chunks) >= 3
    assert all(len(c.split()) <= 800 for c in chunks)
    assert all(c.endswith("end.") for c in chunks)  # cut between sentences


def test_monster_sentence_is_split_by_words():
    text = " ".join(f"w{i}" for i in range(2000))  # no punctuation at all
    chunks = chunk_text(text, overlap_words=0)
    assert all(len(c.split()) <= 800 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_short_tail_is_merged():
    text = "\n\n".join([para(0, 500), para(1, 200), para(2, 50)])
    chunks = chunk_text(text, overlap_words=0)
    # 500 + 200 fits, 50 more still fits: everything in one chunk
    assert len(chunks) == 1
    text = "\n\n".join([para(0, 700), para(1, 100)])
    chunks = chunk_text(text, overlap_words=0)
    # 700 + 100 = 800 still fits under max
    assert len(chunks) == 1
    text = "\n\n".join([para(0, 750), para(1, 100)])
    chunks = chunk_text(text, overlap_words=0)
    # can't merge without exceeding max, so the short tail stays separate
    assert [len(c.split()) for c in chunks] == [750, 100]
