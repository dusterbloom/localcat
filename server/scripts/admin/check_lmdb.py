#!/usr/bin/env python3
"""Check LMDB graph database for stored facts"""
import lmdb
import msgpack
import sys

try:
    # Open LMDB database
    env = lmdb.open('graph.lmdb', readonly=True)

    with env.begin() as txn:
        cursor = txn.cursor()

        print("=== LMDB Contents Sample ===")
        count = 0
        dog_related = []
        you_facts = []

        for key, value in cursor:
            count += 1
            if count <= 10:  # Show first 10 entries
                try:
                    k = key.decode('utf-8')
                    v = msgpack.unpackb(value, raw=False)
                    print(f"Key: {k}")
                    print(f"Value: {v}")
                    print("-" * 40)
                except:
                    print(f"Binary key/value: {key[:50]}...")
                    print("-" * 40)

            # Check for dog/milo/pet related entries
            try:
                k = key.decode('utf-8')
                if 'dog' in k.lower() or 'milo' in k.lower() or 'pet' in k.lower():
                    v = msgpack.unpackb(value, raw=False)
                    dog_related.append((k, v))

                # Check for 'you' facts
                if k.startswith('adj:you'):
                    v = msgpack.unpackb(value, raw=False)
                    you_facts.append((k, v))
            except:
                pass

        print(f"\nTotal entries: {count}")

        if dog_related:
            print("\n=== Dog/Pet Related Facts ===")
            for k, v in dog_related[:5]:
                print(f"Key: {k}")
                print(f"Value: {v}")
                print("-" * 40)
        else:
            print("\nNo dog/pet related facts found in LMDB")

        if you_facts:
            print(f"\n=== Sample 'you' Facts ({len(you_facts)} total) ===")
            for k, v in you_facts[:5]:
                print(f"Key: {k}")
                print(f"Value: {v}")
                print("-" * 40)

    env.close()

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)