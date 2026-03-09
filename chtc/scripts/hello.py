#!/usr/bin/env python3
"""hello.py — verify that HTCondor can run a Python script."""

import sys
import os

print(f"Hello from HTCondor!")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Files in working directory: {os.listdir('.')}")
print(f"Hostname: {os.uname().nodename}")
