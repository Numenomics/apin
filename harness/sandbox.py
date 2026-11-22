"""Zero-dependency Linux kernel sandboxing for untrusted submission workers.

Installs two unprivileged OS-level confinement layers inside `harness.ipc` child
processes immediately before importing untrusted submission code:
  1. Linux Landlock (syscalls 444/445/446): restricts filesystem reads strictly to
     the Python runtime/site-packages and the single submission file (blocking reads
     of `models/`, `data/`, `runs/`, `~/.ssh`, and sibling files) and forbids file
     writes (except `/dev/null`).
  2. Linux seccomp-BPF: denies IPv4 (`AF_INET`) and IPv6 (`AF_INET6`) socket creation
     and `io_uring` syscalls with `EPERM`, while keeping `AF_UNIX` allowed for the
     parent-child IPC `socketpair`.

Controlled by `APIN_SANDBOX`:
  - `auto` (default): enable on Linux x86_64/aarch64; gracefully skip if unavailable.
  - `strict` / `1` / `on`: require both Landlock and seccomp; raise if unavailable.
  - `off` / `0`: disable confinement.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import socket
import stat
import sys
import sysconfig
from pathlib import Path
from typing import Iterable

ENV_SANDBOX = "APIN_SANDBOX"

# Landlock filesystem access flags (Linux UAPI <linux/landlock.h>)
_LL_EXECUTE = 1 << 0
_LL_WRITE_FILE = 1 << 1
_LL_READ_FILE = 1 << 2
_LL_READ_DIR = 1 << 3
_LL_REMOVE_DIR = 1 << 4
_LL_REMOVE_FILE = 1 << 5
_LL_MAKE_CHAR = 1 << 6
_LL_MAKE_DIR = 1 << 7
_LL_MAKE_REG = 1 << 8
_LL_MAKE_SOCK = 1 << 9
_LL_MAKE_FIFO = 1 << 10
_LL_MAKE_BLOCK = 1 << 11
_LL_MAKE_SYM = 1 << 12
_LL_REFER = 1 << 13
_LL_TRUNCATE = 1 << 14

_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2

_LANDLOCK_CREATE_RULESET = 444
_LANDLOCK_ADD_RULE = 445
_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_CREATE_RULESET_VERSION = 1

_SECCOMP_ARCH = {
    "x86_64": {"audit": 0xC000003E, "socket_nr": 41, "seccomp_nr": 317},
    "aarch64": {"audit": 0xC00000B7, "socket_nr": 198, "seccomp_nr": 277},
}
_IO_URING_NRS = (425, 426, 427)


def sandbox_mode() -> str:
    raw = (os.environ.get(ENV_SANDBOX) or "auto").strip().lower()
    if raw in ("off", "0", "false", "no", "disabled"):
        return "off"
    if raw in ("strict", "1", "true", "yes", "on", "required"):
        return "strict"
    return "auto"


def default_read_roots(submission_path: Path) -> list[str]:
    """Read-only roots needed by Python/NumPy/SciPy plus the single submission file."""
    roots: list[str] = []

    def _add(p: str | Path | None) -> None:
        if not p:
            return
        try:
            real = os.path.realpath(str(p))
        except OSError:
            return
        if real and os.path.exists(real) and real not in roots:
            roots.append(real)

    # Only the exact submission file itself — never its parent directory.
    _add(submission_path)

    # Python runtime, stdlib, and venv site-packages (not the repo root or models/data).
    for p in (
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        Path(sys.executable).resolve().parent.parent,
    ):
        _add(p)

    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        _add(sysconfig.get_path(key))

    # System shared libraries & runtime metadata (needed by BLAS/LAPACK/libc).
    for sys_root in (
        "/usr",
        "/lib",
        "/lib64",
        "/etc/ld.so.cache",
        "/etc/ld.so.conf",
        "/etc/ld.so.conf.d",
        "/proc/self",
        "/proc/cpuinfo",
        "/sys/devices/system/cpu",
        "/dev/null",
        "/dev/urandom",
        "/dev/zero",
    ):
        _add(sys_root)

    return roots


def install_landlock(read_roots: Iterable[str], write_roots: Iterable[str] = ("/dev/null",)) -> int:
    """Install an unprivileged Linux Landlock filesystem confinement ruleset. Returns ABI version."""
    if platform.system() != "Linux":
        raise RuntimeError("Landlock requires Linux")

    libc = ctypes.CDLL(None, use_errno=True)
    abi = int(libc.syscall(_LANDLOCK_CREATE_RULESET, ctypes.c_void_p(0), 0, _LANDLOCK_CREATE_RULESET_VERSION))
    if abi < 1:
        err = ctypes.get_errno()
        raise OSError(err, f"Landlock unavailable (abi={abi}, errno={err})")

    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= _LL_REFER
    if abi >= 3:
        handled |= _LL_TRUNCATE

    class _RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class _PathBeneathAttr(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    attr = _RulesetAttr(handled)
    rs = int(libc.syscall(_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0))
    if rs < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"landlock_create_ruleset failed (errno={err})")

    try:
        entries = [(p, False) for p in read_roots] + [(p, True) for p in write_roots]
        for raw_root, allow_write in entries:
            root = os.path.realpath(raw_root)
            if not os.path.exists(root):
                continue
            fd = os.open(root, os.O_PATH | os.O_CLOEXEC)
            try:
                is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
                if allow_write and is_dir:
                    rights = handled
                elif allow_write:
                    rights = _LL_EXECUTE | _LL_WRITE_FILE | _LL_READ_FILE | (_LL_TRUNCATE if abi >= 3 else 0)
                else:
                    rights = _LL_EXECUTE | _LL_READ_FILE
                if is_dir:
                    rights |= _LL_READ_DIR
                rights &= handled
                rule = _PathBeneathAttr(rights, fd)
                if int(libc.syscall(_LANDLOCK_ADD_RULE, rs, _LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(rule), 0)) < 0:
                    err = ctypes.get_errno()
                    raise OSError(err, f"landlock_add_rule failed for {root} (errno={err})")
            finally:
                os.close(fd)

        if int(libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"prctl(PR_SET_NO_NEW_PRIVS) failed (errno={err})")
        if int(libc.syscall(_LANDLOCK_RESTRICT_SELF, rs, 0)) < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"landlock_restrict_self failed (errno={err})")
    finally:
        os.close(rs)

    return abi


def install_network_guard() -> None:
    """Install a Linux seccomp-BPF filter denying AF_INET/AF_INET6 sockets & io_uring (keeping AF_UNIX)."""
    if platform.system() != "Linux":
        raise RuntimeError("seccomp network guard requires Linux")
    arch = platform.machine()
    spec = _SECCOMP_ARCH.get(arch)
    if spec is None:
        raise RuntimeError(f"seccomp network guard unsupported on architecture {arch!r}")

    class _SockFilter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint),
        ]

    class _SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]

    audit = spec["audit"]
    sock_nr = spec["socket_nr"]
    ret_allow = 0x7FFF0000
    ret_kill = 0x80000000
    ret_eperm = 0x00050000 | errno.EPERM

    ops = [
        # Verify syscall architecture matches expected native ABI
        (0x20, 0, 0, 4),          # LD arch
        (0x15, 1, 0, audit),      # JEQ audit -> +1
        (0x06, 0, 0, ret_kill),   # KILL_PROCESS
        # Load syscall number (masking x32 bit)
        (0x20, 0, 0, 0),          # LD nr
        (0x54, 0, 0, 0x3FFFFFFF), # AND 0x3fffffff
        # Block io_uring syscalls (cannot bypass socket/fs checks asynchronously)
        *(
            op
            for nr in _IO_URING_NRS
            for op in ((0x15, 0, 1, nr), (0x06, 0, 0, ret_eperm))
        ),
        # Check socket(domain, ...): block AF_INET and AF_INET6, allow AF_UNIX
        (0x15, 0, 3, sock_nr),             # JEQ socket_nr -> check domain, else ALLOW
        (0x20, 0, 0, 16),                  # LD arg0 (domain)
        (0x15, 2, 0, int(socket.AF_INET)), # JEQ AF_INET -> EPERM
        (0x15, 1, 0, int(socket.AF_INET6)),# JEQ AF_INET6 -> EPERM
        (0x06, 0, 0, ret_allow),           # ALLOW
        (0x06, 0, 0, ret_eperm),           # EPERM
    ]

    arr = (_SockFilter * len(ops))(*(_SockFilter(*x) for x in ops))
    prog = _SockFprog(len(ops), arr)
    libc = ctypes.CDLL(None, use_errno=True)
    if int(libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_NO_NEW_PRIVS) failed (errno={err})")
    if int(libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(prog), 0, 0)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"prctl(PR_SET_SECCOMP) failed (errno={err})")

    # Self-verify both IPv4 and IPv6 sockets are blocked while AF_UNIX remains usable.
    for domain in (socket.AF_INET, socket.AF_INET6):
        try:
            s = socket.socket(domain, socket.SOCK_STREAM)
            s.close()
        except PermissionError:
            pass
        else:
            raise RuntimeError(f"seccomp network guard failed to block domain {domain}")


def confine_worker(submission_path: Path, *, mode: str | None = None) -> dict:
    """Apply Landlock + seccomp confinement in the current worker process.

    Returns a status dict describing which guards were installed.
    """
    resolved_mode = (mode or sandbox_mode()).lower()
    if resolved_mode == "off":
        return {"mode": "off", "landlock_abi": None, "network_guard": False, "active": False}

    landlock_abi: int | None = None
    network_ok = False
    errors: list[str] = []

    try:
        install_network_guard()
        network_ok = True
    except Exception as exc:
        if resolved_mode == "strict":
            raise RuntimeError(f"strict sandbox network guard failed: {exc}") from exc
        errors.append(f"network: {exc}")

    try:
        landlock_abi = install_landlock(default_read_roots(submission_path))
    except Exception as exc:
        if resolved_mode == "strict":
            raise RuntimeError(f"strict sandbox Landlock guard failed: {exc}") from exc
        errors.append(f"landlock: {exc}")

    return {
        "mode": resolved_mode,
        "landlock_abi": landlock_abi,
        "network_guard": network_ok,
        "active": bool(landlock_abi is not None and network_ok),
        "errors": errors,
    }
