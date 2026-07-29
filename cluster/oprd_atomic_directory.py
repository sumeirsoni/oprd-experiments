"""Native no-replace and trusted-writer publication of sibling directories.

The implementation is intentionally stdlib-only. It supports Linux renameat2 with
RENAME_NOREPLACE and Darwin renameatx_np with RENAME_EXCL, with a reservation-based
fallback for filesystems that report the native operation as unsupported.

The reservation fallback has an explicit cooperative-writer contract. The pinned
publication parent must not be concurrently mutated by a hostile process running as
the same effective UID. POSIX plain rename cannot condition replacement on the inode
created by mkdir, so this fallback is not equivalent to kernel RENAME_NOREPLACE
against such a writer. Detected uncertainty fails closed and leaves names untouched.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import errno
import json
import os
import secrets
import stat
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union


PathLike = Union[str, os.PathLike[str]]
DirectoryIdentity = Tuple[int, int]

_FALLBACK_ERRNOS = {
    errno.EINVAL,
    errno.ENOSYS,
    errno.EOPNOTSUPP,
    errno.ENOTSUP,
}
_COLLISION_ERRNOS = {errno.EEXIST, errno.ENOTEMPTY}
_BEST_EFFORT_FSYNC_ERRNOS = {
    errno.EBADF,
    errno.EINVAL,
    errno.EOPNOTSUPP,
    errno.ENOTSUP,
    errno.EROFS,
}
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class AtomicDirectoryError(Exception):
    """Base class for all errors raised by this module."""


class DestinationCollisionError(AtomicDirectoryError, FileExistsError):
    """The destination was present before mutation or native no-replace collided."""


class UnsafePathError(AtomicDirectoryError, ValueError):
    """A path is outside the trust boundary or has an unsafe component."""


class UnsafeParentError(UnsafePathError):
    """The pinned parent is not a private, stable directory."""


class KnownNoChangeMoveError(AtomicDirectoryError):
    """The move failed and its original source is known to remain unchanged."""


class AmbiguousPostconditionError(AtomicDirectoryError):
    """Observed inode postconditions do not prove success or safe non-mutation."""


class UnsupportedPlatformError(AtomicDirectoryError):
    """The current platform cannot provide the required directory operations."""


@dataclass(frozen=True)
class AtomicDirectoryMoveResult:
    method: str
    source_identity: DirectoryIdentity
    recovered_after_error: bool


@dataclass(frozen=True)
class AtomicDirectoryProbeResult:
    move: AtomicDirectoryMoveResult
    probe_root_identity: DirectoryIdentity

    @property
    def method(self) -> str:
        return self.move.method


class AtomicDirectoryOperations:
    """Injectable syscall backend used by the public operations and tests."""

    def __init__(self, *, platform: Optional[str] = None) -> None:
        self.platform = sys.platform if platform is None else platform

    def geteuid(self) -> int:
        if not hasattr(os, "geteuid"):
            raise UnsupportedPlatformError("effective UID checks are unavailable")
        return os.geteuid()

    def lstat(self, path: PathLike, *, dir_fd: Optional[int] = None) -> os.stat_result:
        return os.stat(path, dir_fd=dir_fd, follow_symlinks=False)

    def open(
        self,
        path: PathLike,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: Optional[int] = None,
    ) -> int:
        return os.open(path, flags, mode, dir_fd=dir_fd)

    def close(self, fd: int) -> None:
        os.close(fd)

    def fstat(self, fd: int) -> os.stat_result:
        return os.fstat(fd)

    def fchmod(self, fd: int, mode: int) -> None:
        os.fchmod(fd, mode)

    def mkdir(
        self, path: PathLike, mode: int = 0o777, *, dir_fd: Optional[int] = None
    ) -> None:
        os.mkdir(path, mode, dir_fd=dir_fd)

    def rmdir(self, path: PathLike, *, dir_fd: Optional[int] = None) -> None:
        os.rmdir(path, dir_fd=dir_fd)

    def rename(
        self,
        source: PathLike,
        destination: PathLike,
        *,
        src_dir_fd: Optional[int] = None,
        dst_dir_fd: Optional[int] = None,
    ) -> None:
        os.rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    def fsync(self, fd: int) -> None:
        os.fsync(fd)

    def listdir(self, fd: int) -> Iterable[str]:
        return os.listdir(fd)

    def native_rename_noreplace(
        self,
        parent_fd: int,
        source_name: str,
        destination_name: str,
    ) -> None:
        if self.platform.startswith("linux"):
            self._linux_renameat2(parent_fd, source_name, destination_name)
            return
        if self.platform == "darwin":
            self._darwin_renameatx_np(parent_fd, source_name, destination_name)
            return
        raise UnsupportedPlatformError(
            f"atomic directory no-replace is unsupported on {self.platform!r}"
        )

    @staticmethod
    def _linux_renameat2(
        parent_fd: int, source_name: str, destination_name: str
    ) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "renameat2 is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_fd,
            os.fsencode(source_name),
            parent_fd,
            os.fsencode(destination_name),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            value = ctypes.get_errno()
            raise OSError(value, os.strerror(value))

    @staticmethod
    def _darwin_renameatx_np(
        parent_fd: int, source_name: str, destination_name: str
    ) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is None:
            raise OSError(errno.ENOSYS, "renameatx_np is unavailable")
        renameatx_np.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            parent_fd,
            os.fsencode(source_name),
            parent_fd,
            os.fsencode(destination_name),
            0x00000004,  # RENAME_EXCL
        )
        if result != 0:
            value = ctypes.get_errno()
            raise OSError(value, os.strerror(value))


@dataclass
class _PinnedDirectory:
    path: str
    fd: int
    identity: DirectoryIdentity


@dataclass
class _ValidatedMove:
    destination_path: str
    source_name: str
    destination_name: str
    parent: _PinnedDirectory
    source_fd: int
    source_identity: DirectoryIdentity


@dataclass(frozen=True)
class _MoveState:
    source: Optional[DirectoryIdentity]
    destination: Optional[DirectoryIdentity]


def _identity(value: os.stat_result) -> DirectoryIdentity:
    return (value.st_dev, value.st_ino)


def _absolute_normal_path(path: PathLike, label: str) -> str:
    try:
        value = os.fsdecode(os.fspath(path))
    except (TypeError, ValueError, UnicodeError) as error:
        raise UnsafePathError(f"{label} is not a valid filesystem path") from error
    if not value or not os.path.isabs(value):
        raise UnsafePathError(f"{label} must be absolute")
    normalized = os.path.normpath(value)
    if normalized != value:
        raise UnsafePathError(f"{label} must be lexically canonical: {value!r}")
    return value


def _is_contained(path: str, trusted_root: str) -> bool:
    try:
        return os.path.commonpath((path, trusted_root)) == trusted_root
    except ValueError:
        return False


def _lstat_optional(
    ops: AtomicDirectoryOperations,
    path: PathLike,
    *,
    dir_fd: Optional[int] = None,
) -> Optional[os.stat_result]:
    try:
        return ops.lstat(path, dir_fd=dir_fd)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == errno.ENOENT:
            return None
        raise


def _require_secure_directory_stat(
    value: os.stat_result,
    *,
    label: str,
    euid: int,
    parent_error: bool = False,
) -> None:
    error_type = UnsafeParentError if parent_error else UnsafePathError
    if not stat.S_ISDIR(value.st_mode):
        raise error_type(f"{label} is not a real directory")
    if value.st_uid != euid:
        raise error_type(
            f"{label} is owned by uid {value.st_uid}, not effective uid {euid}"
        )
    if stat.S_IMODE(value.st_mode) & 0o022:
        raise error_type(f"{label} is group- or world-writable")


def _validate_directory_chain(
    path: str,
    trusted_root: str,
    ops: AtomicDirectoryOperations,
) -> None:
    if not _is_contained(path, trusted_root):
        raise UnsafePathError(f"{path!r} is outside trusted root {trusted_root!r}")
    real_root = os.path.realpath(trusted_root)
    real_path = os.path.realpath(path)
    if not _is_contained(real_path, real_root):
        raise UnsafePathError(
            f"canonical path {real_path!r} is outside trusted root {real_root!r}"
        )

    euid = ops.geteuid()
    relative = os.path.relpath(path, trusted_root)
    components = [] if relative == "." else relative.split(os.sep)
    current = trusted_root
    for index, component in enumerate([None, *components]):
        if index:
            current = os.path.join(current, component or "")
        try:
            value = ops.lstat(current)
        except OSError as error:
            raise UnsafePathError(f"cannot inspect directory component {current!r}") from error
        _require_secure_directory_stat(
            value,
            label=f"directory component {current!r}",
            euid=euid,
            parent_error=current == path,
        )


def _close_fds_independently(
    ops: AtomicDirectoryOperations, *fds: Optional[int]
) -> None:
    """Attempt every close and preserve any already-active operation exception."""

    errors = []
    for fd in fds:
        if fd is None:
            continue
        try:
            ops.close(fd)
        except OSError as error:
            errors.append((fd, error))
    if not errors:
        return
    details = "; ".join(f"fd {fd}: {error}" for fd, error in errors)
    primary = sys.exc_info()[1]
    if primary is not None:
        if hasattr(primary, "add_note"):
            primary.add_note(f"additional close failure(s): {details}")
        return
    raise AmbiguousPostconditionError(f"file descriptor close failure(s): {details}")


def _pin_secure_directory(
    path: PathLike,
    *,
    trusted_root: PathLike,
    ops: AtomicDirectoryOperations,
) -> _PinnedDirectory:
    normalized_path = _absolute_normal_path(path, "directory")
    normalized_root = _absolute_normal_path(trusted_root, "trusted_root")
    _validate_directory_chain(normalized_path, normalized_root, ops)
    euid = ops.geteuid()
    try:
        before = ops.lstat(normalized_path)
        fd = ops.open(normalized_path, _DIRECTORY_FLAGS)
    except OSError as error:
        raise UnsafeParentError(f"cannot pin directory {normalized_path!r}") from error
    try:
        after = ops.lstat(normalized_path)
        pinned = ops.fstat(fd)
        _require_secure_directory_stat(
            before,
            label=f"directory {normalized_path!r}",
            euid=euid,
            parent_error=True,
        )
        _require_secure_directory_stat(
            after,
            label=f"directory {normalized_path!r}",
            euid=euid,
            parent_error=True,
        )
        _require_secure_directory_stat(
            pinned,
            label=f"pinned directory {normalized_path!r}",
            euid=euid,
            parent_error=True,
        )
        identities = {_identity(before), _identity(after), _identity(pinned)}
        if len(identities) != 1:
            raise UnsafeParentError(
                f"directory {normalized_path!r} changed while it was pinned"
            )
        return _PinnedDirectory(
            path=normalized_path,
            fd=fd,
            identity=_identity(pinned),
        )
    except BaseException:
        _close_fds_independently(ops, fd)
        raise


def _verify_pinned_parent(
    parent: _PinnedDirectory, ops: AtomicDirectoryOperations
) -> None:
    try:
        path_stat = ops.lstat(parent.path)
        fd_stat = ops.fstat(parent.fd)
    except OSError as error:
        raise UnsafeParentError(f"cannot revalidate parent {parent.path!r}") from error
    if _identity(path_stat) != parent.identity or _identity(fd_stat) != parent.identity:
        raise UnsafeParentError(f"parent {parent.path!r} changed after it was pinned")


def _open_verified_child_directory(
    parent_fd: int,
    name: str,
    expected: os.stat_result,
    *,
    label: str,
    ops: AtomicDirectoryOperations,
    require_mode_0700: bool = False,
) -> int:
    euid = ops.geteuid()
    _require_secure_directory_stat(expected, label=label, euid=euid)
    try:
        fd = ops.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as error:
        raise UnsafePathError(f"cannot pin {label}") from error
    try:
        path_stat = ops.lstat(name, dir_fd=parent_fd)
        fd_stat = ops.fstat(fd)
        _require_secure_directory_stat(path_stat, label=label, euid=euid)
        _require_secure_directory_stat(fd_stat, label=label, euid=euid)
        if not (
            _identity(expected) == _identity(path_stat) == _identity(fd_stat)
        ):
            raise UnsafePathError(f"{label} changed while it was pinned")
        if require_mode_0700 and stat.S_IMODE(fd_stat.st_mode) != 0o700:
            raise UnsafePathError(f"{label} is not mode 0700")
        return fd
    except BaseException:
        _close_fds_independently(ops, fd)
        raise


def _validate_move(
    source: PathLike,
    destination: PathLike,
    *,
    trusted_root: PathLike,
    ops: AtomicDirectoryOperations,
) -> _ValidatedMove:
    source_path = _absolute_normal_path(source, "source")
    destination_path = _absolute_normal_path(destination, "destination")
    root_path = _absolute_normal_path(trusted_root, "trusted_root")
    source_parent = os.path.dirname(source_path)
    destination_parent = os.path.dirname(destination_path)
    if source_path == destination_path:
        raise UnsafePathError("source and destination must be distinct")
    if source_parent != destination_parent:
        raise UnsafePathError("source and destination must be siblings")
    if not _is_contained(source_path, root_path) or not _is_contained(
        destination_path, root_path
    ):
        raise UnsafePathError("source and destination must be under trusted_root")

    parent = _pin_secure_directory(
        source_parent,
        trusted_root=root_path,
        ops=ops,
    )
    source_fd: Optional[int] = None
    try:
        source_name = os.path.basename(source_path)
        destination_name = os.path.basename(destination_path)
        source_stat = _lstat_optional(ops, source_name, dir_fd=parent.fd)
        if source_stat is None:
            raise UnsafePathError(f"source {source_path!r} does not exist")
        source_fd = _open_verified_child_directory(
            parent.fd,
            source_name,
            source_stat,
            label=f"source {source_path!r}",
            ops=ops,
        )
        pinned_source = ops.fstat(source_fd)
        if pinned_source.st_dev != parent.identity[0]:
            raise UnsafePathError("source and parent are on different devices")
        destination_stat = _lstat_optional(
            ops, destination_name, dir_fd=parent.fd
        )
        if destination_stat is not None:
            raise DestinationCollisionError(
                f"destination {destination_path!r} already exists"
            )
        _verify_pinned_parent(parent, ops)
        return _ValidatedMove(
            destination_path=destination_path,
            source_name=source_name,
            destination_name=destination_name,
            parent=parent,
            source_fd=source_fd,
            source_identity=_identity(pinned_source),
        )
    except BaseException:
        _close_fds_independently(ops, source_fd, parent.fd)
        raise


def _inspect_move_state(
    move: _ValidatedMove, ops: AtomicDirectoryOperations
) -> _MoveState:
    try:
        source_stat = _lstat_optional(
            ops, move.source_name, dir_fd=move.parent.fd
        )
        destination_stat = _lstat_optional(
            ops, move.destination_name, dir_fd=move.parent.fd
        )
    except OSError as error:
        raise AmbiguousPostconditionError("cannot inspect move postconditions") from error
    return _MoveState(
        source=None if source_stat is None else _identity(source_stat),
        destination=(
            None if destination_stat is None else _identity(destination_stat)
        ),
    )


def _confirmed_success(state: _MoveState, source: DirectoryIdentity) -> bool:
    return state.source is None and state.destination == source


def _confirmed_no_change(state: _MoveState, source: DirectoryIdentity) -> bool:
    return state.source == source and state.destination is None


def _fsync_parent_after_success(
    move: _ValidatedMove, ops: AtomicDirectoryOperations
) -> None:
    try:
        ops.fsync(move.parent.fd)
    except OSError as error:
        raise AmbiguousPostconditionError(
            "move is visible, but parent directory fsync failed"
        ) from error


def _result(
    method: str,
    move: _ValidatedMove,
    *,
    recovered_after_error: bool,
    ops: AtomicDirectoryOperations,
) -> AtomicDirectoryMoveResult:
    _fsync_parent_after_success(move, ops)
    return AtomicDirectoryMoveResult(
        method=method,
        source_identity=move.source_identity,
        recovered_after_error=recovered_after_error,
    )


def _raise_native_nonfallback_error(
    error: OSError,
    state: _MoveState,
    move: _ValidatedMove,
) -> None:
    if state.source == move.source_identity and (
        state.destination is not None or error.errno in _COLLISION_ERRNOS
    ):
        raise DestinationCollisionError(
            f"destination {move.destination_path!r} collided during rename"
        ) from error
    if _confirmed_no_change(state, move.source_identity):
        raise KnownNoChangeMoveError(
            f"native no-replace rename failed without changing the source: {error}"
        ) from error
    raise AmbiguousPostconditionError(
        f"native no-replace rename failed with inconsistent postconditions: {error}"
    ) from error


def _directory_is_exact_and_empty(
    parent_fd: int,
    name: str,
    identity: DirectoryIdentity,
    *,
    ops: AtomicDirectoryOperations,
) -> bool:
    value = _lstat_optional(ops, name, dir_fd=parent_fd)
    if value is None or _identity(value) != identity or not stat.S_ISDIR(value.st_mode):
        return False
    try:
        fd = ops.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError:
        return False
    try:
        pinned = ops.fstat(fd)
        if _identity(pinned) != identity or not stat.S_ISDIR(pinned.st_mode):
            return False
        return not list(ops.listdir(fd))
    except OSError:
        return False
    finally:
        _close_fds_independently(ops, fd)


def _remove_cooperative_empty_directory(
    parent_fd: int,
    name: str,
    identity: DirectoryIdentity,
    *,
    ops: AtomicDirectoryOperations,
) -> bool:
    """Remove a known empty probe directory under the cooperative-writer contract.

    POSIX has no expected-inode rmdir. A hostile same-UID writer can race the final
    lstat and rmdir, so this helper is used only during normal cleanup of the private
    random probe tree, never to resolve a fallback move error.
    """

    if not _directory_is_exact_and_empty(parent_fd, name, identity, ops=ops):
        return False
    value = _lstat_optional(ops, name, dir_fd=parent_fd)
    if value is None or _identity(value) != identity:
        return False
    try:
        ops.rmdir(name, dir_fd=parent_fd)
    except OSError:
        return False
    return _lstat_optional(ops, name, dir_fd=parent_fd) is None


def _pin_verified_reservation(
    move: _ValidatedMove,
    *,
    ops: AtomicDirectoryOperations,
) -> Tuple[int, DirectoryIdentity]:
    reservation_stat = _lstat_optional(
        ops, move.destination_name, dir_fd=move.parent.fd
    )
    if reservation_stat is None:
        raise AmbiguousPostconditionError("destination reservation disappeared")
    reservation_identity = _identity(reservation_stat)
    reservation_fd: Optional[int] = None
    try:
        reservation_fd = _open_verified_child_directory(
            move.parent.fd,
            move.destination_name,
            reservation_stat,
            label=f"destination reservation {move.destination_path!r}",
            ops=ops,
        )
        if ops.fstat(reservation_fd).st_dev != move.parent.identity[0]:
            raise AmbiguousPostconditionError(
                "destination reservation and parent are on different devices"
            )
        ops.fchmod(reservation_fd, 0o700)
        after_chmod = ops.fstat(reservation_fd)
        if (
            _identity(after_chmod) != reservation_identity
            or stat.S_IMODE(after_chmod.st_mode) != 0o700
            or list(ops.listdir(reservation_fd))
        ):
            raise AmbiguousPostconditionError(
                "destination reservation is not the exact empty mode-0700 directory"
            )
        return reservation_fd, reservation_identity
    except AmbiguousPostconditionError:
        _close_fds_independently(ops, reservation_fd)
        raise
    except UnsafePathError as error:
        _close_fds_independently(ops, reservation_fd)
        raise AmbiguousPostconditionError(str(error)) from error
    except OSError as error:
        _close_fds_independently(ops, reservation_fd)
        raise AmbiguousPostconditionError(
            f"destination reservation validation failed: {error}"
        ) from error


def _reservation_fallback(
    move: _ValidatedMove,
    *,
    ops: AtomicDirectoryOperations,
) -> AtomicDirectoryMoveResult:
    """Reserve then rename under the explicit trusted/cooperative-writer contract.

    The reservation makes cooperative publication races deterministic. It cannot stop
    a hostile same-UID process from replacing the empty reservation between the final
    identity check and plain rename because POSIX has no conditional rename primitive.
    """

    _verify_pinned_parent(move.parent, ops)
    try:
        ops.mkdir(move.destination_name, 0o700, dir_fd=move.parent.fd)
    except OSError as error:
        # NFS may report an error after committing mkdir. Immediate lstat cannot
        # distinguish that case from a concurrent writer, so no name is removed.
        raise AmbiguousPostconditionError(
            f"destination reservation mkdir reported an uncertain result: {error}"
        ) from error

    reservation_fd, reservation_identity = _pin_verified_reservation(
        move, ops=ops
    )
    try:
        state_before = _inspect_move_state(move, ops)
        if state_before.source != move.source_identity:
            raise AmbiguousPostconditionError(
                "source changed after validation and before fallback rename"
            )
        if state_before.destination != reservation_identity or not _directory_is_exact_and_empty(
            move.parent.fd,
            move.destination_name,
            reservation_identity,
            ops=ops,
        ):
            raise AmbiguousPostconditionError(
                "destination reservation changed before fallback rename"
            )

        try:
            ops.rename(
                move.source_name,
                move.destination_name,
                src_dir_fd=move.parent.fd,
                dst_dir_fd=move.parent.fd,
            )
        except OSError as error:
            # In particular, do not infer success or no-change from immediate NFS
            # lstat results after a reported rename error. Mutate no names afterward.
            raise AmbiguousPostconditionError(
                f"fallback plain rename reported an uncertain result: {error}"
            ) from error

        state = _inspect_move_state(move, ops)
        if not _confirmed_success(state, move.source_identity):
            raise AmbiguousPostconditionError(
                "fallback rename returned success with inconsistent source or destination inodes"
            )
        return _result(
            "reserved_plain_rename",
            move,
            recovered_after_error=False,
            ops=ops,
        )
    finally:
        _close_fds_independently(ops, reservation_fd)


def atomic_move_directory_noreplace(
    source: PathLike,
    destination: PathLike,
    *,
    trusted_root: PathLike,
    ops: Optional[AtomicDirectoryOperations] = None,
) -> AtomicDirectoryMoveResult:
    """Move one owned sibling directory under the documented writer contract.

    Native operations provide kernel no-replace semantics. The reservation fallback
    assumes the validated parent has only cooperating writers, as documented above.
    Validation and postcondition decisions are inode-based. Native no-replace errors
    may be recovered from definitive inode postconditions. Fallback plain-rename errors
    are always ambiguous because immediate network-filesystem observations are not a
    reliable commit oracle.
    """

    backend = AtomicDirectoryOperations() if ops is None else ops
    platform = getattr(backend, "platform", sys.platform)
    if not (platform.startswith("linux") or platform == "darwin"):
        raise UnsupportedPlatformError(
            f"atomic directory moves are unsupported on {platform!r}"
        )

    move = _validate_move(
        source,
        destination,
        trusted_root=trusted_root,
        ops=backend,
    )
    try:
        _verify_pinned_parent(move.parent, backend)
        try:
            backend.fsync(move.source_fd)
        except OSError as error:
            if error.errno not in _BEST_EFFORT_FSYNC_ERRNOS:
                raise KnownNoChangeMoveError(
                    f"source directory fsync failed before move mutation: {error}"
                ) from error

        native_error: Optional[OSError] = None
        try:
            backend.native_rename_noreplace(
                move.parent.fd,
                move.source_name,
                move.destination_name,
            )
        except UnsupportedPlatformError:
            raise
        except OSError as error:
            native_error = error

        state = _inspect_move_state(move, backend)
        if _confirmed_success(state, move.source_identity):
            return _result(
                "native_noreplace",
                move,
                recovered_after_error=native_error is not None,
                ops=backend,
            )
        if native_error is None:
            raise AmbiguousPostconditionError(
                "native no-replace rename returned success with inconsistent inodes"
            )
        if (
            native_error.errno in _FALLBACK_ERRNOS
            and _confirmed_no_change(state, move.source_identity)
        ):
            return _reservation_fallback(move, ops=backend)
        _raise_native_nonfallback_error(native_error, state, move)
        raise AssertionError("unreachable")
    finally:
        _close_fds_independently(backend, move.source_fd, move.parent.fd)


def directory_matches_identity(
    path: PathLike,
    identity: DirectoryIdentity,
    *,
    ops: Optional[AtomicDirectoryOperations] = None,
) -> bool:
    """Return whether path is a real directory with exactly the supplied identity."""

    backend = AtomicDirectoryOperations() if ops is None else ops
    try:
        value = backend.lstat(path)
    except (OSError, TypeError, ValueError):
        return False
    return stat.S_ISDIR(value.st_mode) and _identity(value) == tuple(identity)


def _create_private_child_directory(
    parent: _PinnedDirectory,
    name: str,
    *,
    ops: AtomicDirectoryOperations,
) -> DirectoryIdentity:
    ops.mkdir(name, 0o700, dir_fd=parent.fd)
    value = ops.lstat(name, dir_fd=parent.fd)
    identity = _identity(value)
    fd: Optional[int] = None
    try:
        fd = _open_verified_child_directory(
            parent.fd,
            name,
            value,
            label=f"private directory {name!r}",
            ops=ops,
        )
        ops.fchmod(fd, 0o700)
        after = ops.fstat(fd)
        if (
            _identity(after) != identity
            or after.st_dev != parent.identity[0]
            or stat.S_IMODE(after.st_mode) != 0o700
            or list(ops.listdir(fd))
        ):
            raise AmbiguousPostconditionError(
                f"private directory {name!r} failed identity or mode validation"
            )
        return identity
    finally:
        _close_fds_independently(ops, fd)


def _cleanup_probe_root(
    parent: _PinnedDirectory,
    probe_name: str,
    probe_identity: DirectoryIdentity,
    children: Dict[str, DirectoryIdentity],
    *,
    ops: AtomicDirectoryOperations,
) -> None:
    """Clean the private random probe tree under the cooperative-writer contract."""

    root_stat = _lstat_optional(ops, probe_name, dir_fd=parent.fd)
    if (
        root_stat is None
        or _identity(root_stat) != probe_identity
        or not stat.S_ISDIR(root_stat.st_mode)
    ):
        raise AmbiguousPostconditionError(
            "probe root changed, so cleanup was intentionally skipped"
        )
    root_fd = ops.open(probe_name, _DIRECTORY_FLAGS, dir_fd=parent.fd)
    try:
        if _identity(ops.fstat(root_fd)) != probe_identity:
            raise AmbiguousPostconditionError(
                "probe root changed while cleanup pinned it"
            )
        actual_names = set(ops.listdir(root_fd))
        if actual_names != set(children):
            raise AmbiguousPostconditionError(
                "probe root has unexpected entries, so cleanup was intentionally skipped"
            )
        for name, identity in children.items():
            if not _directory_is_exact_and_empty(root_fd, name, identity, ops=ops):
                raise AmbiguousPostconditionError(
                    f"probe child {name!r} changed, so cleanup was intentionally skipped"
                )
        for name, identity in children.items():
            if not _remove_cooperative_empty_directory(root_fd, name, identity, ops=ops):
                raise AmbiguousPostconditionError(
                    f"could not safely remove probe child {name!r}"
                )
        ops.fsync(root_fd)
    finally:
        _close_fds_independently(ops, root_fd)
    root_stat = _lstat_optional(ops, probe_name, dir_fd=parent.fd)
    if root_stat is None or _identity(root_stat) != probe_identity:
        raise AmbiguousPostconditionError(
            "probe root changed immediately before removal"
        )
    ops.rmdir(probe_name, dir_fd=parent.fd)
    if _lstat_optional(ops, probe_name, dir_fd=parent.fd) is not None:
        raise AmbiguousPostconditionError("probe root remained after rmdir")
    ops.fsync(parent.fd)


def probe_atomic_directory_move(
    parent: PathLike,
    *,
    trusted_root: PathLike,
    ops: Optional[AtomicDirectoryOperations] = None,
) -> AtomicDirectoryProbeResult:
    """Exercise the selected move method and collision behavior in a private subtree."""

    backend = AtomicDirectoryOperations() if ops is None else ops
    platform = getattr(backend, "platform", sys.platform)
    if not (platform.startswith("linux") or platform == "darwin"):
        raise UnsupportedPlatformError(
            f"atomic directory moves are unsupported on {platform!r}"
        )
    pinned_parent = _pin_secure_directory(
        parent,
        trusted_root=trusted_root,
        ops=backend,
    )
    probe_name: Optional[str] = None
    probe_identity: Optional[DirectoryIdentity] = None
    children: Dict[str, DirectoryIdentity] = {}
    result: Optional[AtomicDirectoryProbeResult] = None
    operation_error: Optional[BaseException] = None
    try:
        probe_name = f".oprd-atomic-probe-{secrets.token_hex(12)}"
        try:
            probe_identity = _create_private_child_directory(
                pinned_parent,
                probe_name,
                ops=backend,
            )
        except (AtomicDirectoryError, OSError, ValueError, TypeError) as error:
            try:
                appeared = _lstat_optional(
                    backend, probe_name, dir_fd=pinned_parent.fd
                )
            except OSError as inspect_error:
                raise AmbiguousPostconditionError(
                    "probe root allocation failed and candidate state cannot be inspected"
                ) from inspect_error
            if appeared is not None:
                raise AmbiguousPostconditionError(
                    "probe root allocation reported an error after the candidate appeared; "
                    "the candidate was left untouched"
                ) from error
            raise AmbiguousPostconditionError(
                f"probe root allocation failed before an owned root was established: {error}"
            ) from error

        probe_path = os.path.join(pinned_parent.path, probe_name)
        probe = _pin_secure_directory(
            probe_path,
            trusted_root=trusted_root,
            ops=backend,
        )
        try:
            source_name = "source"
            destination_name = "destination"
            source_identity = _create_private_child_directory(
                probe,
                source_name,
                ops=backend,
            )
            children[source_name] = source_identity
        finally:
            _close_fds_independently(backend, probe.fd)

        source_path = os.path.join(probe_path, source_name)
        destination_path = os.path.join(probe_path, destination_name)
        move_result = atomic_move_directory_noreplace(
            source_path,
            destination_path,
            trusted_root=trusted_root,
            ops=backend,
        )
        if directory_matches_identity(source_path, source_identity, ops=backend):
            raise AmbiguousPostconditionError("probe source remained after successful move")
        if not directory_matches_identity(
            destination_path, source_identity, ops=backend
        ):
            raise AmbiguousPostconditionError(
                "probe destination does not have the original source identity"
            )
        del children[source_name]
        children[destination_name] = source_identity

        probe = _pin_secure_directory(
            probe_path,
            trusted_root=trusted_root,
            ops=backend,
        )
        try:
            collision_source_name = "collision-source"
            collision_source_identity = _create_private_child_directory(
                probe,
                collision_source_name,
                ops=backend,
            )
            children[collision_source_name] = collision_source_identity
            if move_result.method == "native_noreplace":
                try:
                    backend.native_rename_noreplace(
                        probe.fd,
                        collision_source_name,
                        destination_name,
                    )
                except OSError as error:
                    if error.errno not in _COLLISION_ERRNOS:
                        raise AmbiguousPostconditionError(
                            f"native collision probe returned an unexpected error: {error}"
                        ) from error
                else:
                    raise AmbiguousPostconditionError(
                        "native collision probe unexpectedly reported success"
                    )
            else:
                try:
                    backend.mkdir(destination_name, 0o700, dir_fd=probe.fd)
                except OSError as error:
                    if error.errno not in _COLLISION_ERRNOS:
                        raise AmbiguousPostconditionError(
                            f"reservation collision probe returned an unexpected error: {error}"
                        ) from error
                else:
                    raise AmbiguousPostconditionError(
                        "reservation collision probe unexpectedly reported success"
                    )
        finally:
            _close_fds_independently(backend, probe.fd)
        collision_source_path = os.path.join(probe_path, collision_source_name)
        if not directory_matches_identity(
            collision_source_path, collision_source_identity, ops=backend
        ) or not directory_matches_identity(
            destination_path, source_identity, ops=backend
        ):
            raise AmbiguousPostconditionError(
                "selected-method collision probe changed a source or destination inode"
            )
        result = AtomicDirectoryProbeResult(
            move=move_result,
            probe_root_identity=probe_identity,
        )
    except BaseException as error:
        operation_error = error
    finally:
        cleanup_error: Optional[BaseException] = None
        if probe_name is not None and probe_identity is not None:
            try:
                _cleanup_probe_root(
                    pinned_parent,
                    probe_name,
                    probe_identity,
                    children,
                    ops=backend,
                )
            except BaseException as error:
                cleanup_error = error
        try:
            _close_fds_independently(backend, pinned_parent.fd)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            if operation_error is not None:
                raise AmbiguousPostconditionError(
                    f"probe failed and exact cleanup also failed: {cleanup_error}"
                ) from operation_error
            raise cleanup_error
    if operation_error is not None:
        raise operation_error
    if result is None:
        raise AmbiguousPostconditionError("probe completed without a result")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Native no-replace and trusted-writer directory move helper"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    move = subparsers.add_parser("move")
    move.add_argument("--source", required=True)
    move.add_argument("--destination", required=True)
    move.add_argument("--trusted-root", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--parent", required=True)
    probe.add_argument("--trusted-root", required=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        parser = _build_parser()
        arguments = parser.parse_args(None if argv is None else list(argv))
        if arguments.command == "move":
            result = atomic_move_directory_noreplace(
                arguments.source,
                arguments.destination,
                trusted_root=arguments.trusted_root,
            )
        else:
            result = probe_atomic_directory_move(
                arguments.parent,
                trusted_root=arguments.trusted_root,
            )
        print(
            json.dumps(
                {"ok": True, "result": dataclasses.asdict(result)},
                sort_keys=True,
            )
        )
        return 0
    except (AtomicDirectoryError, OSError, ValueError, TypeError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
