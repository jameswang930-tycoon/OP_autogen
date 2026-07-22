"""Pytest root conftest.

Its presence makes pytest put the repository root on sys.path (prepend import
mode), so tests can `import control`, `import memory`, etc. Keep this file
minimal.
"""
