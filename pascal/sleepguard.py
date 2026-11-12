"""Never touch the eGPU after the Mac has slept.

Two kernel panics (2026-11-09 and 2026-11-12, `LLC Bus error ... cmd=rd_ld / ncwr` at the GPU's
BAR0, panicked task `TinyGPU`) had the same cause: the Mac slept (lid closed / 1-minute idle
sleep) while the driver held the card, the Thunderbolt link went down, and on wake the next MMIO
hit a device that was no longer there.  On Apple Silicon that bus error is fatal.

  * hold_awake()   keeps the Mac out of idle sleep for as long as this process lives
                   (`caffeinate -i -s -w <pid>`; -s also covers system sleep on AC power).
                   Closing the lid can still sleep a MacBook without an external display.
  * SleepGuard     records kern.waketime at start; check() costs one sysctl (~1 us) and raises
                   SystemSlept, before any access, once the Mac has slept since.  After a sleep
                   the card is cold anyway: re-run devinit.py + fb.py + reset_engines.py + gr.py.
"""
import ctypes, ctypes.util, os, subprocess, sys

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


class _timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_int32)]


def _sysctl_timeval(name: bytes):
    tv = _timeval(); n = ctypes.c_size_t(ctypes.sizeof(tv))
    if _libc.sysctlbyname(name, ctypes.byref(tv), ctypes.byref(n), None, ctypes.c_size_t(0)):
        return None
    return (tv.tv_sec, tv.tv_usec)


class SystemSlept(BaseException):
    """Raised instead of touching the card after a sleep/wake (BaseException: not swallowed
    by the `except Exception` handlers around individual RPC ops)."""


class SleepGuard:
    def __init__(self):
        self.enabled = sys.platform == "darwin"
        self.wake0 = _sysctl_timeval(b"kern.waketime") if self.enabled else None
        self.sleep0 = _sysctl_timeval(b"kern.sleeptime") if self.enabled else None

    def slept(self) -> bool:
        if not self.enabled:
            return False
        return (_sysctl_timeval(b"kern.waketime") != self.wake0
                or _sysctl_timeval(b"kern.sleeptime") != self.sleep0)

    def check(self):
        if self.slept():
            raise SystemSlept("the Mac slept since this process opened the GPU; the Thunderbolt "
                              "link may be gone, so no further MMIO (re-init the card)")


_caffeinate = None


def hold_awake():
    """Prevent idle/system sleep until this process exits (idempotent)."""
    global _caffeinate
    if _caffeinate is None and sys.platform == "darwin":
        try:
            _caffeinate = subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
        except OSError:
            _caffeinate = False
    return _caffeinate
