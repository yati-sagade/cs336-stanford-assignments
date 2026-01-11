import sys
import cbor2

with open(sys.argv[1], "rb") as fp:
    print(cbor2.loads(fp.read()))
