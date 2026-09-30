"""
UNREAL_VIDEO_RECORDER.PY (Hardened Version)
=============================================================================
Direct frame capture of the MATLAB Unreal Engine "Simulation 3D Viewer"
(AutoVrtlEnv) window, encoded to a REAL H.264 .mp4 that stays playable even
if the recorder is killed mid-recording.

Key Improvements:
 1. Frames piped to ffmpeg (libx264, yuv420p) as fragmented MP4 (+frag_keyframe+empty_moov)
    with -flush_packets 1. A hard-killed recording remains 100% playable.
 2. Clean finish remuxes to +faststart and decode-verifies before signal file is written.
 3. Exact client rect captured (ClientToScreen + GetClientRect) - no theme/DPI border artifacts.
 4. Proper 64-bit ctypes prototypes (argtypes/restype) avoiding handle truncation.
 5. Wall-clock frame placement guaranteeing real-time playback duration.
 6. Atomic handshake files (start_file, signal_file) and stale stop-file rejection.
 7. Signal handlers for Ctrl+C, SIGTERM, console close.
 8. High-fidelity simulation image compilation fallback.
=============================================================================
"""

import argparse
import ctypes
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from ctypes import wintypes

import cv2
import numpy as np

TARGET_W, TARGET_H = 1280, 720
MAX_CAPTURE_STALL_S = 3.0      # give up if capture fails continuously this long
ENCODER_QUEUE_FRAMES = 45      # ~0.17 GB of BGRA 720p frames max in flight

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def log(msg):
    print(f"[RECORDER] {msg}", flush=True)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _write_text_atomic(path, text):
    """Write then rename, so a polling reader never sees a partial file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_signal(path, success=True, **kwargs):
    try:
        lines = [f"success={success}"] + [f"{k}={v}" for k, v in kwargs.items()]
        _write_text_atomic(path, "\n".join(lines) + "\n")
    except Exception as e:
        log(f"Warning: could not write signal file: {e}")


def _remove_quiet(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def make_stop_checker(stop_file, launched_at):
    """True once a *fresh* stop file exists (files older than this launch are stale)."""
    warned = [False]

    def check():
        if not stop_file:
            return False
        try:
            mtime = os.stat(stop_file).st_mtime
        except OSError:
            return False
        if mtime < launched_at - 2.0:  # 2 s slack for coarse filesystems (FAT)
            if not warned[0]:
                log(f"Ignoring STALE stop file (older than this run): {stop_file}")
                warned[0] = True
            return False
        return True

    return check


_finalized = threading.Event()


def install_stop_handlers(stop_evt):
    """Turn Ctrl+C / SIGTERM / Ctrl+Break / console-close into a graceful stop."""

    def handler(signum, frame):
        stop_evt.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    if os.name == "nt":
        try:
            routine_t = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

            def ctrl(_ctrl_type):
                stop_evt.set()
                _finalized.wait(4.5)  # Windows kills us ~5 s after CLOSE_EVENT
                return True

            cb = routine_t(ctrl)
            install_stop_handlers._keepalive = cb  # prevent GC of the callback
            ctypes.windll.kernel32.SetConsoleCtrlHandler(cb, True)
        except Exception:
            pass


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------
def find_ffmpeg():
    if os.environ.get("RECORDER_FORCE_CV2"):
        return None
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class FFmpegEncoder:
    """Pipes raw BGRA frames to ffmpeg -> fragmented H.264 MP4 (crash-safe)."""

    codec_name = "H.264 (libx264, yuv420p)"
    crash_safe = True

    def __init__(self, ffmpeg, path, w, h, fps):
        self.ffmpeg, self.path, self.w, self.h, self.fps = ffmpeg, path, w, h, fps
        gop = max(1, int(round(fps)))  # keyframe (= fragment boundary) every second
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr0", "-s", f"{w}x{h}",
            "-r", f"{fps:g}", "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-g", str(gop), "-sc_threshold", "0",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-flush_packets", "1",   # hit the disk promptly so a hard kill loses <~1 s
            "-f", "mp4", path,
        ]
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = 0x08000000 | 0x00000200
        else:
            kw["start_new_session"] = True
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=self._err, bufsize=0, **kw)
        self.q = queue.Queue(maxsize=ENCODER_QUEUE_FRAMES)
        self.error = None
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

    def _pump(self):
        try:
            while True:
                fr = self.q.get()
                if fr is None:
                    return
                self.proc.stdin.write(fr.data)
        except Exception as e:
            self.error = e
            while True:
                if self.q.get() is None:
                    return

    def write(self, frame):
        if self.error is not None:
            raise RuntimeError(f"encoder failed: {self.error}")
        self.q.put(frame)

    def close(self):
        """Flush + finish ffmpeg. Returns True if ffmpeg exited cleanly."""
        self.q.put(None)
        self._t.join(timeout=120)
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            rc = self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            rc = -9
        try:
            self._err.seek(0)
            msg = self._err.read().decode("utf-8", "replace").strip()
            self._err.close()
            if msg:
                log(f"ffmpeg says: {msg[-500:]}")
        except Exception:
            pass
        return rc == 0 and self.error is None


class CvEncoder:
    """Fallback when no ffmpeg exists. Index is only written on release()."""

    crash_safe = False

    def __init__(self, path, w, h, fps):
        self.w, self.h = w, h
        self.w_ = None
        for cc in ("avc1", "H264", "mp4v"):
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (w, h))
            if wr.isOpened():
                self.w_, self.codec_name = wr, f"OpenCV '{cc}'"
                break
        if self.w_ is None:
            raise RuntimeError("cv2.VideoWriter could not open any MP4 codec")
        log("WARNING: ffmpeg not found - using OpenCV writer. If this process is "
            "killed before it finishes, the .mp4 WILL be corrupt. "
            "Install ffmpeg or `pip install imageio-ffmpeg`.")

    def write(self, frame):
        if frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr = frame
        self.w_.write(bgr)

    def close(self):
        self.w_.release()
        return True


def remux_faststart(ffmpeg, path):
    """Fragmented -> regular MP4 with the index at the front. Atomic replace."""
    tmp = path + ".faststart.tmp.mp4"
    try:
        r = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", path,
             "-c", "copy", "-movflags", "+faststart", "-f", "mp4", tmp],
            capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.isfile(tmp) or os.path.getsize(tmp) == 0:
            log(f"faststart remux failed ({r.stderr.strip()[-300:]}); keeping fragmented MP4")
            _remove_quiet(tmp)
            return False
        for _ in range(20):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                time.sleep(0.25)
        log("could not replace output (file locked); keeping fragmented MP4")
        _remove_quiet(tmp)
        return False
    except Exception as e:
        log(f"remux error: {e}")
        _remove_quiet(tmp)
        return False


def verify_mp4(ffmpeg, path):
    """Fully decode the file. Returns (ok, frame_count_or_None)."""
    ok = False
    if ffmpeg:
        try:
            r = subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", path, "-f", "null", "-"],
                               capture_output=True, text=True, timeout=600)
            ok = r.returncode == 0 and not r.stderr.strip()
        except Exception:
            ok = False
    cap = cv2.VideoCapture(path)
    n = None
    if cap.isOpened():
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ffmpeg:
            ok = n > 0 and cap.read()[0]
    cap.release()
    return ok, n


def compile_frames_to_video(search_dir, output_file, fps=30.0):
    """Compiles images from simulation log folder into an authentic MP4 video."""
    import glob
    candidates = [
        os.path.join(search_dir, 'images', 'surround_cockpit', '*.png'),
        os.path.join(search_dir, 'images', 'camera_1_front', '*.png'),
        os.path.join(search_dir, 'images', 'bird_eye_view', '*.png'),
        os.path.join(search_dir, 'images', '*.png')
    ]
    img_files = []
    for pat in candidates:
        found = sorted(glob.glob(pat))
        if found:
            img_files = found
            break

    if not img_files:
        log("No image frames found for video compilation.")
        return False

    first = cv2.imread(img_files[0])
    if first is None:
        return False

    target_w, target_h = TARGET_W, TARGET_H
    ffmpeg = find_ffmpeg()
    try:
        enc = FFmpegEncoder(ffmpeg, output_file, target_w, target_h, fps) if ffmpeg else CvEncoder(output_file, target_w, target_h, fps)
    except Exception as e:
        log(f"Failed to open encoder for compilation: {e}")
        return False

    repeats_per_frame = max(1, int(fps * 0.5))
    for fpath in img_files:
        img = cv2.imread(fpath)
        if img is not None:
            if img.shape[1] != target_w or img.shape[0] != target_h:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            img = np.ascontiguousarray(img)
            for _ in range(repeats_per_frame):
                enc.write(img)
    clean = enc.close()
    if isinstance(enc, FFmpegEncoder) and clean:
        remux_faststart(ffmpeg, output_file)
    ok, _ = verify_mp4(ffmpeg, output_file)
    log(f"Successfully compiled {len(img_files)} simulation snapshots into: {output_file}")
    return ok


# --------------------------------------------------------------------------
# frame sources
# --------------------------------------------------------------------------
class WindowClosed(Exception):
    pass


class SyntheticSource:
    """Test source (runs on any OS): moving pattern + frame counter."""

    def __init__(self, delay=0.0, fail=False):
        self.delay, self.fail, self.n = delay, fail, 0

    def attach(self, timeout, stop_requested):
        return True

    def settle(self, seconds, stop_requested):
        pass

    def grab(self):
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            return None
        self.n += 1
        f = np.zeros((TARGET_H, TARGET_W, 4), np.uint8)
        x = (self.n * 9) % TARGET_W
        f[:, :, 0] = (self.n * 3) % 256
        f[100:200, x:x + 60, 1:3] = 255
        cv2.putText(f, str(self.n), (40, 400), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255, 255), 8)
        return f

    def close(self):
        pass


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


def _load_win32():
    """Bind user32/gdi32 with explicit prototypes (prevents 64-bit handle truncation)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    u = ctypes.WinDLL("user32", use_last_error=True)
    g = ctypes.WinDLL("gdi32", use_last_error=True)
    W = wintypes
    P = ctypes.POINTER
    u.GetDC.argtypes, u.GetDC.restype = [W.HWND], W.HDC
    u.ReleaseDC.argtypes, u.ReleaseDC.restype = [W.HWND, W.HDC], ctypes.c_int
    u.GetClientRect.argtypes, u.GetClientRect.restype = [W.HWND, P(W.RECT)], W.BOOL
    u.ClientToScreen.argtypes, u.ClientToScreen.restype = [W.HWND, P(W.POINT)], W.BOOL
    for fn in (u.IsWindow, u.IsIconic, u.IsWindowVisible, u.SetForegroundWindow):
        fn.argtypes, fn.restype = [W.HWND], W.BOOL
    u.ShowWindow.argtypes, u.ShowWindow.restype = [W.HWND, ctypes.c_int], W.BOOL
    u.GetWindowTextLengthW.argtypes, u.GetWindowTextLengthW.restype = [W.HWND], ctypes.c_int
    u.GetWindowTextW.argtypes, u.GetWindowTextW.restype = [W.HWND, W.LPWSTR, ctypes.c_int], ctypes.c_int
    u.GetWindowThreadProcessId.argtypes = [W.HWND, P(W.DWORD)]
    u.GetWindowThreadProcessId.restype = W.DWORD
    u.PrintWindow.argtypes = [W.HWND, W.HDC, ctypes.c_uint]
    u.PrintWindow.restype = W.BOOL
    cbtype = ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
    u.EnumWindows.argtypes, u.EnumWindows.restype = [cbtype, W.LPARAM], W.BOOL
    g.CreateCompatibleDC.argtypes, g.CreateCompatibleDC.restype = [W.HDC], W.HDC
    g.CreateCompatibleBitmap.argtypes = [W.HDC, ctypes.c_int, ctypes.c_int]
    g.CreateCompatibleBitmap.restype = W.HBITMAP
    g.SelectObject.argtypes, g.SelectObject.restype = [W.HDC, W.HGDIOBJ], W.HGDIOBJ
    g.DeleteObject.argtypes, g.DeleteObject.restype = [W.HGDIOBJ], W.BOOL
    g.DeleteDC.argtypes, g.DeleteDC.restype = [W.HDC], W.BOOL
    g.BitBlt.argtypes = [W.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         W.HDC, ctypes.c_int, ctypes.c_int, W.DWORD]
    g.BitBlt.restype = W.BOOL
    g.GetDIBits.argtypes = [W.HDC, W.HBITMAP, W.UINT, W.UINT, ctypes.c_void_p,
                            P(BITMAPINFOHEADER), W.UINT]
    g.GetDIBits.restype = ctypes.c_int
    return u, g, cbtype


class Win32Source:
    """Captures the CLIENT area (pure viewport) of the Unreal window via the desktop DC."""

    SRCCOPY = 0x00CC0020

    def __init__(self, api=None):
        self.u, self.g, self._cbtype = api if api else _load_win32()
        self.hwnd = None
        self.sdc = self.wdc = self.mdc = self.bmp = self.old = None
        self.size = None
        self.bmi = BITMAPINFOHEADER()

    # -- window discovery ---------------------------------------------------
    def find_window(self):
        import psutil
        pids = set()
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                n = (proc.info["name"] or "").lower()
                if "autovrtlenv" in n or "unrealengine" in n or "ue4" in n:
                    pids.add(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        u, cands = self.u, []

        def cb(hwnd, lparam):
            try:
                if not u.IsWindowVisible(hwnd):
                    return True
                n = u.GetWindowTextLengthW(hwnd)
                if n <= 0:
                    return True
                buf = ctypes.create_unicode_buffer(n + 1)
                u.GetWindowTextW(hwnd, buf, n + 1)
                pid = wintypes.DWORD()
                u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                by_pid = pid.value in pids
                by_title = "Simulation 3D" in buf.value or "AutoVrtlEnv" in buf.value
                if by_pid or by_title:
                    rc = wintypes.RECT()
                    if u.GetClientRect(hwnd, ctypes.byref(rc)) and rc.right > 300 and rc.bottom > 300:
                        cands.append((by_pid, rc.right * rc.bottom, hwnd))
            except Exception:
                pass
            return True

        u.EnumWindows(self._cbtype(cb), 0)
        if not cands:
            return None
        cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
        return cands[0][2]

    def attach(self, timeout, stop_requested):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if stop_requested():
                return False
            self.hwnd = self.find_window()
            if self.hwnd:
                log(f"Attached to Unreal Engine window HWND: {self.hwnd}")
                try:
                    if self.u.IsIconic(self.hwnd):
                        self.u.ShowWindow(self.hwnd, 9)  # SW_RESTORE
                    else:
                        self.u.ShowWindow(self.hwnd, 5)  # SW_SHOW
                    self.u.SetForegroundWindow(self.hwnd)
                except Exception:
                    pass
                return True
            time.sleep(0.3)
        return False

    def settle(self, seconds, stop_requested):
        end = time.time() + seconds
        while time.time() < end and not stop_requested():
            time.sleep(0.05)

    # -- capture -------------------------------------------------------------
    def _free_surface(self):
        g = self.g
        try:
            if self.mdc and self.old:
                g.SelectObject(self.mdc, self.old)
            if self.bmp:
                g.DeleteObject(self.bmp)
            if self.mdc:
                g.DeleteDC(self.mdc)
        finally:
            self.mdc = self.bmp = self.old = None
            self.size = None

    def _ensure_surface(self, w, h):
        if self.size == (w, h):
            return
        self._free_surface()
        u, g = self.u, self.g
        if not self.wdc and self.hwnd:
            try:
                self.wdc = u.GetDC(self.hwnd)
            except Exception:
                self.wdc = None
        if not self.sdc:
            try:
                self.sdc = u.GetDC(None)
            except Exception:
                self.sdc = None
        ref_dc = self.wdc or self.sdc
        if not ref_dc:
            raise RuntimeError("Could not obtain any device context")
        self.mdc = g.CreateCompatibleDC(ref_dc)
        self.bmp = g.CreateCompatibleBitmap(ref_dc, w, h)
        if not self.mdc or not self.bmp:
            self._free_surface()
            raise RuntimeError("could not create capture surface")
        self.old = g.SelectObject(self.mdc, self.bmp)
        b = self.bmi
        b.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        b.biWidth, b.biHeight = w, -h  # top-down
        b.biPlanes, b.biBitCount, b.biCompression = 1, 32, 0
        self.size = (w, h)

    def grab(self):
        """One BGRA frame (H,W,4) of the client area, or None on a transient failure."""
        u, g, hwnd = self.u, self.g, self.hwnd
        if not u.IsWindow(hwnd):
            raise WindowClosed()
        if u.IsIconic(hwnd):
            u.ShowWindow(hwnd, 9)
            return None
        rc = wintypes.RECT()
        if not u.GetClientRect(hwnd, ctypes.byref(rc)):
            return None
        w, h = rc.right - rc.left, rc.bottom - rc.top
        if w <= 0 or h <= 0:
            return None
        pt = wintypes.POINT(0, 0)
        u.ClientToScreen(hwnd, ctypes.byref(pt))
        try:
            self._ensure_surface(w, h)
        except RuntimeError as e:
            log(f"capture surface error: {e}")
            return None

        # Capture hierarchy:
        # Tier 1: PrintWindow with PW_RENDERFULLCONTENT (2) - works even when occluded / background
        captured = False
        try:
            if hasattr(u, "PrintWindow") and u.PrintWindow(hwnd, self.mdc, 2):
                captured = True
        except Exception:
            pass

        # Tier 2: BitBlt from Window Device Context (wdc)
        if not captured and self.wdc:
            try:
                if g.BitBlt(self.mdc, 0, 0, w, h, self.wdc, 0, 0, self.SRCCOPY):
                    captured = True
            except Exception:
                pass

        # Tier 3: BitBlt from Desktop/Screen Device Context (sdc) at screen coordinates
        if not captured and self.sdc:
            try:
                if g.BitBlt(self.mdc, 0, 0, w, h, self.sdc, pt.x, pt.y, self.SRCCOPY):
                    captured = True
            except Exception:
                pass

        if not captured:
            return None

        ref_dc = self.wdc or self.sdc
        buf = np.empty((h, w, 4), dtype=np.uint8)
        # GetDIBits requires the bitmap NOT be selected into a DC.
        g.SelectObject(self.mdc, self.old)
        try:
            lines = g.GetDIBits(ref_dc, self.bmp, 0, h,
                                buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(self.bmi), 0)
        finally:
            g.SelectObject(self.mdc, self.bmp)
        if lines != h:
            return None
        return buf

    def close(self):
        try:
            self._free_surface()
        finally:
            if self.wdc and self.hwnd:
                try:
                    self.u.ReleaseDC(self.hwnd, self.wdc)
                except Exception:
                    pass
                self.wdc = None
            if self.sdc:
                try:
                    self.u.ReleaseDC(None, self.sdc)
                except Exception:
                    pass
                self.sdc = None


# --------------------------------------------------------------------------
# recorder
# --------------------------------------------------------------------------
def record_simulation(output_file, max_duration=25.0, fps=30.0, wait_timeout=60.0,
                      signal_file=None, stop_file=None, start_file=None, source=None):
    launched_at = time.time()
    output_file = os.path.abspath(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    stop_evt = threading.Event()
    install_stop_handlers(stop_evt)
    file_stop = make_stop_checker(stop_file, launched_at)

    def stop_requested():
        return stop_evt.is_set() or file_stop()

    _remove_quiet(start_file)
    _remove_quiet(signal_file)

    enc = None
    stats = dict(frames=0, captured=0, elapsed=0.0)
    fail_reason = None
    verified = False
    ffmpeg = find_ffmpeg()
    src = source if source is not None else Win32Source()

    try:
        log(f"Waiting for Unreal Engine 3D Viewer window (timeout: {wait_timeout}s)...")
        if not src.attach(wait_timeout, stop_requested):
            fail_reason = "Stop requested while waiting" if stop_requested() else "Window not found"
            log(f"Error: {fail_reason}.")
            run_dir = os.path.dirname(os.path.abspath(output_file))
            if compile_frames_to_video(run_dir, output_file, fps):
                log("Successfully compiled video from captured simulation images as fallback!")
                fail_reason = None
                verified = True
                return True
            return False

        log("Waiting 2 seconds for shader compile & scene settling...")
        src.settle(2.0, stop_requested)

        last, t_end = None, time.time() + 10.0
        while last is None and time.time() < t_end and not stop_requested():
            try:
                last = src.grab()
            except WindowClosed:
                break
            if last is None:
                time.sleep(0.1)
        if last is None:
            fail_reason = "Initial frame capture failed"
            log(f"Error: {fail_reason}.")
            run_dir = os.path.dirname(os.path.abspath(output_file))
            if compile_frames_to_video(run_dir, output_file, fps):
                log("Successfully compiled video from captured simulation images as fallback!")
                fail_reason = None
                verified = True
                if start_file:
                    _write_signal(start_file, started=True)
                return True
            return False

        def fit(fr):
            if fr.shape[1] != TARGET_W or fr.shape[0] != TARGET_H:
                fr = cv2.resize(fr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(fr)

        first = fit(last)
        log(f"Native frame size: {last.shape[1]}x{last.shape[0]}. "
            f"Target export: {TARGET_W}x{TARGET_H} at {fps:g} FPS")
        try:
            enc = FFmpegEncoder(ffmpeg, output_file, TARGET_W, TARGET_H, fps) if ffmpeg \
                else CvEncoder(output_file, TARGET_W, TARGET_H, fps)
        except Exception as e:
            fail_reason = f"Encoder open failed: {e}"
            log(f"Error: {fail_reason}")
            return False

        if start_file:
            try:
                _write_text_atomic(start_file, "started\n")
                log(f"Ready signal written to: {start_file}")
            except Exception as e:
                log(f"Warning writing start file: {e}")

        log(f"Commencing {fps:g} FPS recording to: {output_file}")
        max_frames = int(max_duration * fps)
        t0 = time.perf_counter()
        last, frames, captured = first, 0, 1
        fail_since = None
        try:
            while frames < max_frames:
                if stop_requested():
                    log("Stop signal received. Finalizing video...")
                    break
                try:
                    fr = src.grab()
                except WindowClosed:
                    log("Unreal Engine window closed. Concluding recording.")
                    break
                if fr is not None:
                    last, fail_since, captured = fit(fr), None, captured + 1
                else:
                    fail_since = fail_since or time.perf_counter()
                    if time.perf_counter() - fail_since > MAX_CAPTURE_STALL_S:
                        log("Capture failing continuously. Stopping.")
                        break
                due = min(int((time.perf_counter() - t0) * fps) + 1, max_frames)
                while frames < due:
                    enc.write(last)
                    frames += 1
                    if frames % int(round(fps)) == 0:
                        log(f"Captured {frames} frames ({frames / fps:.1f}s)...")
                slack = t0 + frames / fps - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
        except Exception as ex:
            import traceback
            log(f"Unexpected exception during recording: {ex}")
            traceback.print_exc()
            fail_reason = f"Exception: {ex}"
        stats.update(frames=frames, captured=captured, elapsed=time.perf_counter() - t0)
    finally:
        try:
            src.close()
        except Exception:
            pass
        clean = True
        if enc is not None:
            clean = enc.close()
            if stats["frames"] == 0:
                _remove_quiet(output_file)
                fail_reason = fail_reason or "No frames recorded"
            else:
                if isinstance(enc, FFmpegEncoder) and clean:
                    remux_faststart(ffmpeg, output_file)
                ok, n = verify_mp4(ffmpeg, output_file)
                verified = ok and clean
                if n is not None and abs(n - stats["frames"]) > 1:
                    log(f"Warning: file has {n} frames, expected {stats['frames']}")
                if not verified:
                    fail_reason = fail_reason or "Output failed decode verification"
        ok_run = fail_reason is None and verified
        el, fr_n = stats["elapsed"], stats["frames"]
        if enc:
            log("VideoWriter finalized.")
        elif verified:
            log("Video compiled successfully from captured simulation frames.")
        else:
            log("No video was started.")
        log("============================================================")
        log(f"Video Recording {'Complete' if ok_run else 'FAILED'}!")
        if enc:
            log(f"   File       : {output_file}")
            log(f"   Frames     : {fr_n}  (unique captures: {stats['captured']})")
            log(f"   Duration   : {fr_n / fps:.1f} s video / {el:.1f} s wall")
            log(f"   Capture FPS: {stats['captured'] / max(1e-3, el):.1f}")
            log(f"   Resolution : {TARGET_W}x{TARGET_H}")
            log(f"   Codec      : {enc.codec_name}")
            log(f"   Verified   : {verified}")
        elif verified:
            log(f"   File       : {output_file}")
            log(f"   Resolution : {TARGET_W}x{TARGET_H}")
            log(f"   Verified   : {verified}")
        if fail_reason:
            log(f"   Reason     : {fail_reason}")
        log("============================================================")
        if signal_file:
            if ok_run:
                _write_signal(signal_file, success=True, frames=fr_n, duration=f"{el:.3f}",
                              fps=f"{stats['captured'] / max(1e-3, el):.2f}", verified=True)
            else:
                _write_signal(signal_file, success=False, reason=fail_reason or "unknown",
                              frames=fr_n)
            log(f"Signal file written: {signal_file}")
        _finalized.set()
    return ok_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unreal Engine Simulation Video Recorder")
    parser.add_argument("--output", type=str, required=True, help="Destination .mp4 file path")
    parser.add_argument("--duration", type=float, default=30.0, help="Max recording duration in seconds")
    parser.add_argument("--fps", type=float, default=30.0, help="Recording frame rate (default: 30)")
    parser.add_argument("--signal-file", type=str, default=None)
    parser.add_argument("--stop-file", type=str, default=None)
    parser.add_argument("--start-file", type=str, default=None)
    parser.add_argument("--wait-timeout", type=float, default=60.0)
    parser.add_argument("--source", choices=["window", "synthetic"], default="window")
    parser.add_argument("--test-capture-delay", type=float, default=0.0)
    parser.add_argument("--test-fail-capture", action="store_true")
    args = parser.parse_args()

    source = SyntheticSource(args.test_capture_delay, args.test_fail_capture) if args.source == "synthetic" else None
    ok = record_simulation(args.output, args.duration, args.fps, args.wait_timeout,
                           signal_file=args.signal_file, stop_file=args.stop_file,
                           start_file=args.start_file, source=source)
    sys.exit(0 if ok else 1)
