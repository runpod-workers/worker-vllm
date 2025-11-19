import os
import sys

# Ensure `src` is on the import path so we can import the existing module
root_dir = os.path.dirname(__file__)
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import the module which already starts the Runpod serverless handler.
# The module-level code in `src/handler.py` calls `runpod.serverless.start(...)`,
# so importing it here triggers the same behavior when Runpod runs this file.
import src.handler  # noqa: F401
