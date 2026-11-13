"""Never touch the eGPU after the Mac has slept.

Two kernel panics (2026-11-09 and 2026-11-12, `LLC Bus error ... cmd=rd_ld / ncwr` at the GPU's
BAR0, panicked task `TinyGPU`) had the same cause: the Mac slept (lid closed / 1-minute idle
sleep) while the driver held the card, the Thunderbolt link went down, and on wake the next MMIO
hit a device that was no longer there.  On Apple Silicon that bus error is fatal.

  * hold_awake()   keeps the Mac out of idle sleep for as long as this process lives
                   (`caffeinate -i -s -w <pid>`; -s also covers system sleep on AC power).
                   Closing the lid can still sleep a MacBook without an external display.
  * SleepGuard     records kern.waketime at start; check() costs two sysctls (~3 us, at most every 50 ms of wall time) and raises
                   SystemSlept, before any access, once the Mac has slept since.  After a sleep
                   the card is cold anyway: re-run devinit.py + fb.py + reset_engines.py + gr.py.
  * egpu_present() asks the IO registry (no MMIO, ~20 ms) whether an NVIDIA PCI device is attached.
                   SleepGuard polls it once a second from a daemon thread, and check() raises
                   SystemSlept as soon as the card is gone: an unplugged or powered-off enclosure is
                   the same hazard as a sleep, the next access hits a device that is not there.
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


def egpu_present() -> bool:
    """True while an NVIDIA PCI device is in the IO registry. Never touches the card. On anything
    but macOS, or if ioreg itself cannot be run, the answer is True: this is a tripwire for a card
    that has gone, not a gate that an unrelated failure should close."""
    if sys.platform != "darwin":
        return True
    fake = os.environ.get("PASCAL_SIMULATE_DISCONNECT_FILE")     # test hook: absent while this file exists
    if fake and os.path.exists(fake):
        return False
    try:
        out = subprocess.run(["ioreg", "-r", "-c", "IOPCIDevice", "-d", "1", "-w0"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    return "pci10de," in out


class SystemSlept(BaseException):
    """Raised instead of touching the card after a sleep/wake (BaseException: not swallowed
    by the `except Exception` handlers around individual RPC ops)."""


class SleepGuard:
    def __init__(self):
        self.enabled = sys.platform == "darwin"
        self.wake0 = _sysctl_timeval(b"kern.waketime") if self.enabled else None
        self.sleep0 = _sysctl_timeval(b"kern.sleeptime") if self.enabled else None
        self.gone = False
        if self.enabled and os.environ.get("PASCAL_NO_PRESENCE_WATCH") != "1":
            import threading
            threading.Thread(target=self._watch_presence, daemon=True).start()

    PRESENCE_INTERVAL = 1.0

    def _watch_presence(self):
        import time
        while not self.gone:
            time.sleep(self.PRESENCE_INTERVAL)
            if not egpu_present() and not egpu_present():     # two misses in a row, not one odd read
                self.gone = True

    def slept(self) -> bool:
        if not self.enabled:
            return False
        return (_sysctl_timeval(b"kern.waketime") != self.wake0
                or _sysctl_timeval(b"kern.sleeptime") != self.sleep0)

    INTERVAL = 0.05     # re-check at most every 50 ms of *wall* time: the wall clock jumps across a
    _last = 0.0         # sleep, so the first check after a wake always runs the sysctls

    def check(self):
        import time
        if self.gone:
            raise SystemSlept("the eGPU is no longer attached (unplugged or powered off), so no "
                              "further MMIO (reconnect it, then re-init the card)")
        now = time.time()
        if 0 <= now - self._last < self.INTERVAL:
            return
        self._last = now
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
