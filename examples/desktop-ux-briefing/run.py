#!/usr/bin/env python3
"""Desktop UX Tracker — entrypoint.

All machinery lives in the shared engine (/opt/data/shared/story_tracker);
this directory holds only the topic config (config.json) and state
(desktop_ux_stories.json). The legacy desktop_ux_briefing.py is kept for
rollback but is no longer used by cron.

Cron: cd here && python3 run.py  (HERMES_TZ + .env in this directory).
"""
import os
import sys

sys.path.insert(0, os.environ.get("STORY_TRACKER_HOME", "/opt/data/shared"))
from story_tracker import engine  # noqa: E402

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(engine.main())
