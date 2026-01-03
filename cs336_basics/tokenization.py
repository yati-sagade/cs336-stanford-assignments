import sys
import os
import logging
import regex as re
from pprint import pformat
from collections import Counter, defaultdict
from collections.abc import Iterator

log = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()))

PRETOK_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
SPECIAL_TOK_PAT = re.compile(r"<\|endoftext\|>")
END_OF_TEXT_TOKEN = b"<|endoftext|>"

Token = bytes
TokenID = int
TokenPair = tuple[TokenID, TokenID]
Pretoken = tuple[TokenID, ...]


class BytePairEncodingTokenizer:
    def __init__(self, *, max_vocab_size: int = None, num_merges: int = None, special_tokens: list[bytes] = None):
        assert (num_merges is not None) != (max_vocab_size is not None), (
            "Exactly one of num_merges and max_vocab_size must be provided"
        )
        # Each vocab item is called a Token, and is just a sequence of bytes.
        self._vocab: list[Token] = []
        # Map from vocab item to its index in self._vocab.
        self._btoi: dict[Token, TokenID] = {}
        # Each pretoken from the corpus is represented as a list of TokenIDs. A
        # TokenID is just the index of the corresponding vocabulary item.
        self._corpus: list[Pretoken] = []
        # self._courpus_counts[i] is the count of the pretoken self._corpus[i].
        self._corpus_counts: list[int] = []
        # Tracks merges performed so far.
        self._merges: list[tuple[TokenID, TokenID, TokenID]] = []
        # Seed vocab with special tokens and all bytes.
        if special_tokens is not None:
            for tok in special_tokens:
                self._add_to_vocab(tok)
        for b in range(256):
            token = bytes([b])
            self._add_to_vocab(token)
        if max_vocab_size is None:
            self._max_allowed_vocab_size = self._vocab_size() + num_merges
        else:
            self._max_allowed_vocab_size = max_vocab_size
        # Bigram data structures.
        # Given two vocab items (a, b)...
        # self._pair_postings[a, b] is the set of indices of pretokens that contain the (a, b) bigram.
        self._pair_postings: dict[TokenPair, set[int]] = defaultdict(set)
        # self._pair_counts[a, b] is the count of the (a, b) bigram.
        self._pair_counts: dict[TokenPair, int] = defaultdict(int)

    def _add_to_vocab(self, token: Token) -> TokenID:
        assert token not in self._btoi, f"Token {token} already exists in vocab"
        idx = len(self._vocab)
        self._vocab.append(token)
        self._btoi[token] = idx
        return idx

    def _vocab_size(self) -> int:
        return len(self._vocab)

    def _num_remaining_merges(self) -> int:
        return max(0, self._max_allowed_vocab_size - self._vocab_size())

    def fit(self, pretokens: Iterator[bytes]):
        for pretoken, count in Counter(pretokens).items():
            pretoken_enc = tuple(self._btoi[bytes([b])] for b in pretoken.encode("utf-8"))
            pretoken_idx = len(self._corpus)
            self._corpus.append(pretoken_enc)
            self._corpus_counts.append(count)
            for a, b in zip(pretoken_enc[:-1], pretoken_enc[1:]):
                self._pair_postings[a, b].add(pretoken_idx)
                self._pair_counts[a, b] += count
        imerge = 0
        while self._num_remaining_merges() > 0:
            log.debug(f"====== Merge round {imerge}")
            self._merge()
            imerge += 1
        log.debug(f"*** Final vocab {pformat(self._vocab)}")
        log.debug(f"*** Final merges {pformat(self.merges())}")

    def merges(self) -> list[tuple[bytes, bytes]]:
        return [(self._vocab[a], self._vocab[b]) for a, b, _ in self._merges]

    def vocab(self) -> dict[int, bytes]:
        return {i: b for i, b in enumerate(self._vocab)}

    def _maxkey(self, t: tuple[tuple[TokenID, TokenID], int]):
        (a, b), count = t
        return (count, (self._vocab[a], self._vocab[b]))

    def _merge(self):
        best_pair, _ = max(self._pair_counts.items(), key=self._maxkey)
        besta, bestb = best_pair
        new_token_id = self._add_to_vocab(self._vocab[besta] + self._vocab[bestb])
        self._merges.append((besta, bestb, new_token_id))
        log.debug(f"*** Going to merge pair {pformat(best_pair)}")
        best_pair_postings: list[TokenID] = self._pair_postings[best_pair].copy()
        for itok in best_pair_postings:
            pretoken = self._corpus[itok]
            rewritten_pretoken = self._rewrite_pretoken(itok, best_pair, new_token_id)
            log.debug(f"*** Rewrote {pretoken} to {rewritten_pretoken}")
            self._corpus[itok] = rewritten_pretoken
        # Finally, remove the best pair
        self._pair_postings.pop(best_pair)
        self._pair_counts.pop(best_pair)
        return best_pair

    def _rewrite_pretoken(
        self, pretoken_id: int, pair_to_merge: tuple[TokenID, TokenID], new_token_id: TokenID
    ) -> tuple[TokenID, ...]:
        merged = []
        skip_one = False
        pretoken = self._corpus[pretoken_id]
        pretoken_count = self._corpus_counts[pretoken_id]
        for a, b in zip(pretoken[:-1], pretoken[1:]):
            # Treat all bigrams of the old version of the pretoken as stale, and
            # remove this pretoken from the postings of those bigrams. Later, we add
            # the pretoken back to the new bigrams formed from the rewritten version
            # of the pretoken.
            try:
                self._pair_postings[a, b].remove(pretoken_id)
            except KeyError:
                # A given bigram can appear more than once in the same pretoken, e.g.,
                # ('e', 's') in 'testes'.
                pass
            self._pair_counts[a, b] -= pretoken_count
            # Previous iteration of the loop was a hit and thus `a` is already
            # accounted for, so skip to the next iteration.
            if skip_one:
                skip_one = False
                continue
            # Hit! Signal to skip the next iteration to prevent `b` from being double
            # processed.
            if (a, b) == pair_to_merge:
                merged.append(new_token_id)
                skip_one = True
            else:
                merged.append(a)
        if not skip_one and len(pretoken) > 0:
            # If the loop above does not end with a hit, we have an unprocessed token.
            merged.append(pretoken[-1])
        # Now add the pretoken ID to all new bigrams of the rewritten version.
        for a, b in zip(merged[:-1], merged[1:]):
            self._pair_postings[a, b].add(pretoken_id)
            self._pair_counts[a, b] += pretoken_count
        return tuple(merged)


class TextChunker:
    def __init__(self, input_path, special_tokens):
        special_tok_pat = re.compile("|".join(re.escape(tok) for tok in special_tokens))
        with open(input_path, "rb") as fp:
            text_bytes = fp.read()
        print(f"Read {len(text_bytes)} bytes")
        text = text_bytes.decode("utf-8")
        self._chunks = special_tok_pat.split(text)
        print(f"Split into {len(self._chunks)} chunks")
        self._next_idx = 0

    def next_chunk(self):
        if (idx := self._next_idx) < len(self._chunks):
            self._next_idx += 1
            return self._chunks[idx]
        return None


def pretokenize(text):
    return (m.group() for m in PRETOK_PAT.finditer(text))


if __name__ == "__main__":
    try:
        data_path = sys.argv[1]
    except IndexError:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data/TinyStoriesV2-GPT4-valid.txt")
    special_tokens = ["<|endoftext|>"]
    chunker = TextChunker(data_path, special_tokens=special_tokens)
    tok = BytePairEncodingTokenizer(num_merges=6, special_tokens=special_tokens)
    pretokens = []
    while (chunk := chunker.next_chunk()) is not None:
        pretokens.extend(pretokenize(chunk))
    tok.fit(pretokens)
