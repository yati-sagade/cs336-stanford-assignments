import sys
import os
import logging
import regex as re
from pprint import pformat
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, ByteString, Iterator, Set

log = logging.getLogger(__name__)
logging.basicConfig(
  level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()))

PRETOK_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
SPECIAL_TOK_PAT = re.compile(r"<\|endoftext\|>")
END_OF_TEXT_TOKEN = b'<|endoftext|>'

class TokenState:
  def __init__(self, pretokens: Iterator[bytes]):
    # pretoken_bytes[i] is the list of vocabulary items that makes up the ith pretoken.
    self.pretoken_bytes: List[Tuple[bytes]] = []
    # pretoken_counts[i] is the count of pretoken_bytes[i].
    self.pretoken_counts: List[int] = []
    for pretoken, count in Counter(pretokens).items():
      self.pretoken_bytes.append(tuple(bytes([b]) for b in pretoken.encode('utf-8')))
      self.pretoken_counts.append(count)
    # Bigram data structures.
    # Given two vocab items (a, b)...
    # pair_postings[a, b] is the set of pretokens that contain the (a, b) bigram.
    self.pair_postings: Dict[Tuple[bytes], Set[int]] = defaultdict(set)
    # pair_counts[a, b] is the count of the (a, b) bigram.
    self.pair_counts: Dict[Tuple[bytes], int] = defaultdict(int)
    log.debug(f'*** Pretoken bytes {pformat(self.pretoken_bytes)}')
    log.debug(f'*** Pretoken counts {pformat(self.pretoken_counts)}')
    
    for i, (pretoken, count) in enumerate(zip(self.pretoken_bytes,
                                              self.pretoken_counts)):
      for a, b in zip(pretoken[:-1], pretoken[1:]):
        k = (a, b)
        self.pair_postings[k].add(i)
        self.pair_counts[k] += count
    
    log.debug(f'*** Pretoken bytes {pformat(self.pretoken_bytes)}')
    log.debug(f'*** Pretoken counts {pformat(self.pretoken_counts)}')
    log.debug(f'*** Pair postings {pformat(self.pair_postings)}')
    log.debug(f'*** Pair counts {pformat(self.pair_counts)}')
  
  def merge(self, key=None):
    if key is None:
      key = lambda t: (t[1], t[0])
    best_pair, _ = max(self.pair_counts.items(), key=key)
    syllable = best_pair[0] + best_pair[1]
    log.debug(f'*** Going to merge pair {pformat(best_pair)}')
    for itok in self.pair_postings[best_pair]:
      merged = self._merge_one(self.pretoken_bytes[itok], best_pair, syllable)
      log.debug(f'*** Merged {self.pretoken_bytes[itok]} to {merged}')
      self.pretoken_bytes[itok] = merged
      pre, post = self._get_neighbors(merged, syllable)
      for p in pre:
        new_pair = (p, syllable)
        self.pair_postings[new_pair].add(itok)
        self.pair_counts[new_pair] += self.pretoken_counts[itok]

        old_pair = (p, best_pair[0])
        self.pair_postings[old_pair].remove(itok)
        self.pair_counts[old_pair] -= self.pretoken_counts[itok]
      for p in post:
        new_pair = (syllable, p)
        self.pair_postings[new_pair].add(itok)
        self.pair_counts[new_pair] += self.pretoken_counts[itok]

        old_pair = (best_pair[1], p)
        self.pair_postings[old_pair].remove(itok)
        self.pair_counts[old_pair] -= self.pretoken_counts[itok]
    
    # Finally, remove the best pair
    self.pair_postings.pop(best_pair)
    self.pair_counts.pop(best_pair)
    return best_pair

  def _get_neighbors(self, pretoken: List[bytes], syllable: bytes):
    pre, post = [], []
    for i, b in enumerate(pretoken):
      if b == syllable:
        if i > 0:
          pre.append(pretoken[i-1])
        if i + 1 < len(pretoken):
          post.append(pretoken[i+1])
    return pre, post

  def _merge_one(self, buf: List[bytes],
                pair_to_merge: Tuple[bytes, bytes],
                merged_syllable: bytes) -> List[bytes]:
    ret = []
    i = 0
    while i < len(buf):
      if i + 1 < len(buf) and (buf[i], buf[i+1]) == pair_to_merge:
          ret.append(merged_syllable)
          i += 2
      else:
        ret.append(buf[i])
        i += 1
    return ret


class BytePairEncodingTokenizer:
  Token = bytes
  TokenPair = Tuple[bytes]
  def __init__(self, *, max_vocab_size: int=None, num_merges: int=None):
    assert (num_merges is not None) != (max_vocab_size is not None),\
      'Exactly one of num_merges and max_vocab_size must be provided'
    self._vocab: List[bytes] = []
    self._merges: List[Tuple[bytes]] = []
    # Map from vocab item to its index in self._vocab.
    self._btoi: Dict[bytes, int] = {}
    # Seed vocab with special tokens and all bytes.
    self._add_to_vocab(END_OF_TEXT_TOKEN)
    for b in range(256):
      self._add_to_vocab(bytes([b]))
    if max_vocab_size is None:
      self._max_allowed_vocab_size = self._vocab_size() + num_merges
  
  def _add_to_vocab(self, token: bytes):
    assert token not in self._btoi,\
      f"Token {token} already exists in vocab"
    idx = len(self._vocab)
    self._vocab.append(token)
    self._btoi[token] = idx

  def _vocab_size(self):
    return len(self._vocab)

  def _num_remaining_merges(self):
    return max(0, self._max_allowed_vocab_size - self._vocab_size())
  
  @profile
  def fit(self, pretokens: Iterator[bytes]):
    state = TokenState(pretokens)
    nmerge = 6
    imerge = 0
    while self._num_remaining_merges() > 0:
      log.debug(f'====== Merge round {imerge}')
      best_pair = state.merge()
      syllable = best_pair[0] + best_pair[1]
      self._vocab.append(syllable)
      self._merges.append(best_pair)
      imerge += 1
    log.debug(f'*** Final vocab {pformat(self._vocab)}')
    log.debug(f'*** Final merges {pformat(self._merges)}')

class TextChunker:
  def __init__(self, input_path):
    with open(input_path, 'rb') as fp:
      text_bytes = fp.read()
    print(f'Read {len(text_bytes)} bytes')
    text = text_bytes.decode('utf-8')
    self._chunks = SPECIAL_TOK_PAT.split(text)
    print(f'Split into {len(self._chunks)} chunks')
    self._next_idx = 0
  
  def next_chunk(self):
    if (idx:=self._next_idx) < len(self._chunks):
      self._next_idx += 1
      return self._chunks[idx]
    return None

def pretokenize(text):
  return (m.group() for m in PRETOK_PAT.finditer(text))

if __name__ == '__main__':
  try:
    data_path = sys.argv[1]
  except IndexError:
    data_path = os.path.join(
      os.path.dirname(__file__), '..', 'data/TinyStoriesV2-GPT4-valid.txt')
  chunker = TextChunker(data_path)
  tok = BytePairEncodingTokenizer(num_merges=6)
  pretokens = []
  while (chunk:=chunker.next_chunk()) is not None:
    pretokens.extend(pretokenize(chunk))
  tok.fit(pretokens)
