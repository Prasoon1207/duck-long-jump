import os

# GLFW's Wayland backend segfaults when the MuJoCo viewer closes; X11 (XWayland) is stable.
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")
