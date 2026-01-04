import sys
import os
import logging
import regex as re
import time
import struct
import itertools
import tempfile
from dataclasses import dataclass
from pprint import pformat
from collections import Counter, defaultdict
from collections.abc import Iterator
from typing import BinaryIO
import multiprocessing

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
        log.debug("*** BPE initialized, starting merges")
        imerge = 0
        while self._num_remaining_merges() > 0:
            log.debug(f"====== Merge round {imerge}")
            self._merge()
            imerge += 1
        # log.debug(f"*** Final vocab {pformat(self._vocab)}")
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
        # log.debug(f"*** Going to merge pair {pformat(best_pair)}")
        best_pair_postings: list[TokenID] = self._pair_postings[best_pair].copy()
        for itok in best_pair_postings:
            rewritten_pretoken = self._rewrite_pretoken(itok, best_pair, new_token_id)
            # log.debug(f"*** Rewrote {pretoken} to {rewritten_pretoken}")
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


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


@dataclass
class ChunkParams:
    input_path: str
    output_path: str
    startpos: int
    limitpos: int
    special_tokens: list[str]


def _process_chunk(input: tuple[int, ChunkParams]):
    idx, params = input
    pid = os.getpid()

    def _log(msg):
        log.debug(f"[process {idx}/pid {pid}] {msg}")

    spre = re.compile("|".join(map(re.escape, params.special_tokens)))
    _log(f"Chunk processing started on range {params.startpos}..{params.limitpos}")
    with open(params.input_path, "rb") as inp, open(params.output_path, "wb") as outp:
        inp.seek(params.startpos)
        chunk = inp.read(params.limitpos - params.startpos).decode("utf-8")
        _log(f"Read {len(chunk)} char long chunk")
        subchunks = spre.split(chunk)
        _log(f"Processing {len(subchunks)} sub-chunks")
        finds = [PRETOK_PAT.finditer(subchunk) for subchunk in subchunks]
        for m in itertools.chain(*finds):
            b = m.group().encode("utf-8")
            outp.write(struct.pack("<I", len(b)))
            outp.write(b)
    _log(f"Chunk reading ended on range {params.startpos}..{params.limitpos}, wrote {params.output_path}")


def read_pretokens(input_path: os.PathLike, separator: str, special_tokens: list[str], /, parallelism: int = 8):
    with open(input_path, "rb") as fp:
        chunk_boundaries = find_chunk_boundaries(fp, parallelism, separator.encode("utf-8"))
        log.debug(f"Chunk boundaries for {input_path}: {chunk_boundaries} ({len(chunk_boundaries)} items)")
    # Each of the N (=parallelism) processes outputs pretokens in a separate file in a temp location.
    # Each pretoken is encoded as:
    #   <4-byte-little-endian-length><pretoken-utf-8-bytes>
    # See _process_chunk().
    tempdir = tempfile.TemporaryDirectory(delete=False)
    chunk_inputs = [
        ChunkParams(
            input_path=input_path,
            output_path=os.path.join(tempdir.name, f"bpe-chunk-{startpos}-{limitpos}.txt"),
            startpos=startpos,
            limitpos=limitpos,
            special_tokens=special_tokens,
        )
        for startpos, limitpos in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
    ]
    with multiprocessing.Pool(parallelism) as p:
        tstart = time.time()
        p.map(_process_chunk, enumerate(chunk_inputs))
        tend = time.time()
        log.debug(f"Pretokenization complete in {tend - tstart}s, parallelism={parallelism}")

    def _iter_file(path):
        with open(path, "rb") as fp:
            while True:
                header = fp.read(4)
                if not header:
                    break
                assert len(header) == 4, f"Unexpected EOF in {path}"
                (length,) = struct.unpack("<I", header)
                pretoken = fp.read(length)
                yield pretoken.decode("utf-8")

    gens = [_iter_file(chip.output_path) for chip in chunk_inputs]
    return itertools.chain(*gens), tempdir.cleanup


if __name__ == "__main__":
    try:
        data_path = sys.argv[1]
    except IndexError:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data/TinyStoriesV2-GPT4-valid.txt")
    special_tokens = ["<|endoftext|>"]
    tok = BytePairEncodingTokenizer(num_merges=6, special_tokens=special_tokens)
    pretokens, cleanup = read_pretokens(data_path, "<|endoftext|>", ["<|endoftext|>"], parallelism=8)
    log.debug("Pretokenization done, starting BPE training")
    tok.fit(pretokens)
    cleanup()
