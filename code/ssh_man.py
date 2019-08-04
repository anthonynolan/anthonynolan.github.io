#!/usr/bin/env python3

import subprocess

a = subprocess.run('man ssh | col -b', shell=True, capture_output=True)
print(a)
