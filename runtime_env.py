import os


def enable_legacy_keras():
    """
    Legacy no-op kept for compatibility with older imports.
    """
    return None


def build_runtime_env():
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env
