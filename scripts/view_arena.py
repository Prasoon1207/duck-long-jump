"""Open the arena with an untrained duck holding its standing pose (sanity check)."""

from duckjump.env.longjump_env import LongJumpConfig
from duckjump.infra.playback import watch

if __name__ == "__main__":
    watch(None, LongJumpConfig(), episodes=1)
