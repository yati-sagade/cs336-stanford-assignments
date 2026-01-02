- What Unicode character does chr(0) return?
  - '\x00'
- How does this character’s string representation (__repr__()) differ from its printed representation?
  - `repr(chr(0))` is just the string r"'\x00'", but when printed, this character is unprintable ("control character")
- What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations.
  - The char disappears when printed.

- Why do we prefer training tokenizers on utf-8 encoded bytes as opposed to using utf-16 or utf-32?

  ```python
    >>> s = 'hello, 你好 नमस्ते வணக்கம் Привет'
    >>> len(s)
    31
    >>> len(s.encode('utf-8'))
    67
    >>> len(s.encode('utf-32'))
    128
    >>> len(s.encode('utf-16'))
    64
   ```
   So utf-32 and utf-16 are _fixed length_ encodings, which means that even for small codepoints that fit within the ASCII range, they'll take up 4 or 2 bytes per codepoint. utf-8 is variable-length, and encodes codepoints in _up to_ 4 bytes. utf-8 is also ubiquitous.

- Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
  ```py
  def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
  ```

  The Hindi word 'नमस्ते' would break this function. The reason is that each char in that word encodes to 3 bytes in utf-8. The function does not handle such byte sequences that actually should decode to a single char.

  ```python
  >>> [(c, c.encode('utf-8')) for c in 'नमस्ते']
  [('न', b'\xe0\xa4\xa8'), ('म', b'\xe0\xa4\xae'), ('स', b'\xe0\xa4\xb8'), ('्', b'\xe0\xa5\x8d'), ('त', b'\xe0\xa4\xa4'), ('े', b'\xe0\xa5\x87')]
  ```