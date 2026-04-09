import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """Naive sentence-aware chunking. Returns list of text chunks.

    This implementation prefers to split on sentences if spaCy is available,
    otherwise falls back to a simple sentence splitter.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        # fallback: split on period followed by space
        sentences = [s.strip() for s in text.split('. ') if s.strip()]

    chunks = []
    cur = []
    cur_len = 0
    for sent in sentences:
        words = sent.split()
        l = len(words)
        if cur_len + l <= chunk_size:
            cur.append(sent)
            cur_len += l
        else:
            if cur:
                chunks.append(' '.join(cur))
            # start new chunk
            # allow long sentence to form its own chunk
            cur = [sent]
            cur_len = l

    if cur:
        chunks.append(' '.join(cur))

    # apply overlap by merging adjacent chunks if needed
    if overlap > 0 and len(chunks) > 1:
        out = []
        for i in range(len(chunks)):
            if i == 0:
                out.append(chunks[i])
            else:
                prev_words = out[-1].split()
                add_words = chunks[i].split()
                # take last `overlap` words from prev and prefix to current
                prefix = ' '.join(prev_words[-overlap:]) if len(prev_words) > overlap else ' '.join(prev_words)
                out.append(prefix + ' ' + chunks[i])
        chunks = out

    return chunks
