#!/usr/bin/env python

from neuromaps.datasets import available_annotations

anns = available_annotations()
print(f"Found {len(anns)} annotations\n")

for a in anns:
    print(a)