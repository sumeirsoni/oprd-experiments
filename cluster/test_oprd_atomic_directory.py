import contextlib
import errno
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import oprd_atomic_directory as atomic


class ScriptedOperations(atomic.AtomicDirectoryOperations):
    def __init__(self, *, native_errno=None, native_commit=False, rename_action=None):
        super().__init__(platform="linux")
        self.native_errno = native_errno
        self.native_commit = native_commit
        self.rename_action = rename_action
        self.events = []

    def native_rename_noreplace(
        self,
        parent_fd,
        source_name,
        destination_name,
    ):
        self.events.append("native")
        try:
            super().lstat(destination_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(errno.EEXIST, "destination exists")
        if self.native_commit:
            super().rename(
                source_name,
                destination_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        if self.native_errno is not None:
            raise OSError(self.native_errno, os.strerror(self.native_errno))
        if not self.native_commit:
            super().rename(
                source_name,
                destination_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )

    def mkdir(self, path, mode=0o777, *, dir_fd=None):
        self.events.append(f"mkdir:{os.fsdecode(path)}")
        return super().mkdir(path, mode, dir_fd=dir_fd)

    def rename(
        self,
        source,
        destination,
        *,
        src_dir_fd=None,
        dst_dir_fd=None,
    ):
        self.events.append("plain_rename")
        if self.rename_action is not None:
            return self.rename_action(
                self,
                source,
                destination,
                src_dir_fd,
                dst_dir_fd,
            )
        return super().rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    def fsync(self, fd):
        self.events.append("fsync")
        return super().fsync(fd)


class CloseFailureOperations(ScriptedOperations):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.close_attempts = []

    def close(self, fd):
        self.close_attempts.append(fd)
        atomic.AtomicDirectoryOperations.close(self, fd)
        raise OSError(errno.EIO, "injected close failure")


class FakeDeviceOperations(ScriptedOperations):
    def __init__(self):
        super().__init__()
        self.source_fds = set()

    @staticmethod
    def _with_device(value, device):
        fields = {
            name: getattr(value, name)
            for name in dir(value)
            if name.startswith("st_")
        }
        fields["st_dev"] = device
        return types.SimpleNamespace(**fields)

    def lstat(self, path, *, dir_fd=None):
        value = super().lstat(path, dir_fd=dir_fd)
        if dir_fd is not None and os.fsdecode(path) == "source":
            return self._with_device(value, value.st_dev + 1)
        return value

    def open(self, path, flags, mode=0o777, *, dir_fd=None):
        fd = super().open(path, flags, mode, dir_fd=dir_fd)
        if dir_fd is not None and os.fsdecode(path) == "source":
            self.source_fds.add(fd)
        return fd

    def fstat(self, fd):
        value = super().fstat(fd)
        if fd in self.source_fds:
            return self._with_device(value, value.st_dev + 1)
        return value

    def close(self, fd):
        self.source_fds.discard(fd)
        return super().close(fd)


class AtomicDirectoryTestCase(unittest.TestCase):
    def make_layout(self):
        temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(temporary.name)
        source = root / "source"
        destination = root / "destination"
        source.mkdir(mode=0o700)
        return temporary, root, source, destination

    def identity(self, path):
        value = path.lstat()
        return (value.st_dev, value.st_ino)


class AtomicMoveTests(AtomicDirectoryTestCase):
    def test_darwin_native_uses_renameatx_np_with_parent_fd_and_basenames(self):
        class FakeRenameAtX:
            def __init__(self):
                self.argtypes = None
                self.restype = None
                self.calls = []

            def __call__(self, *arguments):
                self.calls.append(arguments)
                return 0

        renameatx = FakeRenameAtX()
        fake_libc = types.SimpleNamespace(renameatx_np=renameatx)
        with mock.patch.object(atomic.ctypes, "CDLL", return_value=fake_libc):
            atomic.AtomicDirectoryOperations._darwin_renameatx_np(
                17, "source", "destination"
            )

        self.assertEqual(
            renameatx.calls,
            [(17, b"source", 17, b"destination", 0x00000004)],
        )

    def test_native_success_returns_source_inode_identity(self):
        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        source_identity = self.identity(source)
        ops = ScriptedOperations()

        result = atomic.atomic_move_directory_noreplace(
            source, destination, trusted_root=root, ops=ops
        )

        self.assertEqual(result.method, "native_noreplace")
        self.assertEqual(result.source_identity, source_identity)
        self.assertFalse(result.recovered_after_error)
        self.assertFalse(source.exists())
        self.assertTrue(atomic.directory_matches_identity(destination, source_identity))

    def test_native_error_after_commit_is_recovered(self):
        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = ScriptedOperations(native_errno=errno.EIO, native_commit=True)

        result = atomic.atomic_move_directory_noreplace(
            source, destination, trusted_root=root, ops=ops
        )

        self.assertEqual(result.method, "native_noreplace")
        self.assertTrue(result.recovered_after_error)
        self.assertFalse(source.exists())
        self.assertTrue(destination.is_dir())

    def test_only_unsupported_native_errors_use_reservation_fallback(self):
        fallback_errnos = sorted(
            {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}
        )
        for error_number in fallback_errnos:
            with self.subTest(error_number=error_number):
                temporary, root, source, destination = self.make_layout()
                with temporary:
                    ops = ScriptedOperations(native_errno=error_number)
                    result = atomic.atomic_move_directory_noreplace(
                        source, destination, trusted_root=root, ops=ops
                    )
                    self.assertEqual(result.method, "reserved_plain_rename")
                    self.assertFalse(result.recovered_after_error)
                    self.assertTrue(destination.is_dir())

    def test_permission_cross_device_and_io_errors_do_not_fallback(self):
        for error_number in (errno.EACCES, errno.EXDEV, errno.EIO):
            with self.subTest(error_number=error_number):
                temporary, root, source, destination = self.make_layout()
                with temporary:
                    ops = ScriptedOperations(native_errno=error_number)
                    with self.assertRaises(atomic.KnownNoChangeMoveError):
                        atomic.atomic_move_directory_noreplace(
                            source, destination, trusted_root=root, ops=ops
                        )
                    self.assertTrue(source.is_dir())
                    self.assertFalse(destination.exists())
                    self.assertFalse(
                        any(event == "mkdir:destination" for event in ops.events)
                    )

    def test_concurrent_native_collision_does_not_fallback(self):
        class NativeWinnerOperations(ScriptedOperations):
            def native_rename_noreplace(
                self,
                parent_fd,
                source_name,
                destination_name,
            ):
                self.events.append("native")
                atomic.AtomicDirectoryOperations.mkdir(
                    self, destination_name, 0o700, dir_fd=parent_fd
                )
                raise FileExistsError(errno.EEXIST, "winner")

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = NativeWinnerOperations()

        with self.assertRaises(atomic.DestinationCollisionError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )

        self.assertTrue(source.is_dir())
        self.assertTrue(destination.is_dir())
        self.assertFalse(any(event == "mkdir:destination" for event in ops.events))

    def test_fallback_error_after_commit_is_ambiguous_despite_lstat_success_shape(self):
        def commit_then_error(ops, source, destination, src_fd, dst_fd):
            atomic.AtomicDirectoryOperations.rename(
                ops,
                source,
                destination,
                src_dir_fd=src_fd,
                dst_dir_fd=dst_fd,
            )
            raise OSError(errno.EIO, "reported after commit")

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = ScriptedOperations(
            native_errno=errno.ENOSYS,
            rename_action=commit_then_error,
        )

        with self.assertRaises(atomic.AmbiguousPostconditionError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )

        self.assertFalse(source.exists())
        self.assertTrue(destination.is_dir())

    def test_fallback_error_before_commit_leaves_reservation_untouched(self):
        def fail_before_commit(ops, source, destination, src_fd, dst_fd):
            raise OSError(errno.EIO, "before commit")

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        source_identity = self.identity(source)
        ops = ScriptedOperations(
            native_errno=errno.EINVAL,
            rename_action=fail_before_commit,
        )

        with self.assertRaises(atomic.AmbiguousPostconditionError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )

        self.assertTrue(atomic.directory_matches_identity(source, source_identity))
        self.assertTrue(destination.is_dir())

    def test_reservation_validation_error_leaves_reservation_untouched(self):
        class ValidationErrorOperations(ScriptedOperations):
            def fchmod(self, fd, mode):
                raise OSError(errno.EIO, "reservation validation failed")

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = ValidationErrorOperations(native_errno=errno.ENOSYS)

        with self.assertRaises(atomic.AmbiguousPostconditionError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )

        self.assertTrue(source.is_dir())
        self.assertTrue(destination.is_dir())

    def test_source_fsync_error_before_mutation_is_known_no_change(self):
        class SourceFsyncErrorOperations(ScriptedOperations):
            def fsync(self, fd):
                self.events.append("fsync")
                raise OSError(errno.EIO, "source fsync failed")

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = SourceFsyncErrorOperations(native_errno=errno.ENOSYS)

        with self.assertRaises(atomic.KnownNoChangeMoveError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )

        self.assertTrue(source.is_dir())
        self.assertFalse(destination.exists())
        self.assertNotIn("native", ops.events)
        self.assertNotIn("plain_rename", ops.events)

    def test_unsupported_source_fsync_errors_remain_best_effort(self):
        class BestEffortFsyncOperations(ScriptedOperations):
            def __init__(self, fsync_errno):
                super().__init__()
                self.fsync_errno = fsync_errno
                self.fsync_calls = 0

            def fsync(self, fd):
                self.events.append("fsync")
                self.fsync_calls += 1
                if self.fsync_calls == 1:
                    raise OSError(
                        self.fsync_errno, os.strerror(self.fsync_errno)
                    )
                return atomic.AtomicDirectoryOperations.fsync(self, fd)

        for error_number in sorted(atomic._BEST_EFFORT_FSYNC_ERRNOS):
            with self.subTest(error_number=error_number):
                temporary, root, source, destination = self.make_layout()
                with temporary:
                    result = atomic.atomic_move_directory_noreplace(
                        source,
                        destination,
                        trusted_root=root,
                        ops=BestEffortFsyncOperations(error_number),
                    )
                    self.assertEqual(result.method, "native_noreplace")
                    self.assertTrue(destination.is_dir())

    def test_existing_destination_of_every_kind_is_a_collision_and_untouched(self):
        kinds = ("directory", "file", "symlink", "fifo")
        for kind in kinds:
            with self.subTest(kind=kind):
                temporary, root, source, destination = self.make_layout()
                with temporary:
                    if kind == "directory":
                        destination.mkdir()
                    elif kind == "file":
                        destination.write_text("winner")
                    elif kind == "symlink":
                        destination.symlink_to(root / "missing")
                    else:
                        if not hasattr(os, "mkfifo"):
                            self.skipTest("mkfifo unavailable")
                        os.mkfifo(destination)
                    destination_stat = destination.lstat()
                    ops = ScriptedOperations()
                    with self.assertRaises(atomic.DestinationCollisionError):
                        atomic.atomic_move_directory_noreplace(
                            source, destination, trusted_root=root, ops=ops
                        )
                    self.assertEqual(
                        self.identity(destination),
                        (destination_stat.st_dev, destination_stat.st_ino),
                    )
                    self.assertTrue(source.is_dir())
                    self.assertNotIn("native", ops.events)

    def test_reservation_mkdir_error_with_destination_is_ambiguous_and_untouched(self):
        class WinnerOperations(ScriptedOperations):
            def mkdir(self, path, mode=0o777, *, dir_fd=None):
                if dir_fd is not None and os.fsdecode(path) == "destination":
                    atomic.AtomicDirectoryOperations.mkdir(
                        self, path, 0o700, dir_fd=dir_fd
                    )
                    raise FileExistsError(errno.EEXIST, "winner")
                return super().mkdir(path, mode, dir_fd=dir_fd)

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = WinnerOperations(native_errno=errno.ENOSYS)

        with self.assertRaises(atomic.AmbiguousPostconditionError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )

        self.assertTrue(source.is_dir())
        self.assertTrue(destination.is_dir())

    def test_replaced_reservation_is_ambiguous_and_not_removed(self):
        replacement_identity = None

        def replace_reservation(ops, source, destination, src_fd, dst_fd):
            nonlocal replacement_identity
            atomic.AtomicDirectoryOperations.rmdir(
                ops, destination, dir_fd=dst_fd
            )
            atomic.AtomicDirectoryOperations.mkdir(
                ops, destination, 0o700, dir_fd=dst_fd
            )
            replacement_identity = atomic._identity(
                atomic.AtomicDirectoryOperations.lstat(
                    ops, destination, dir_fd=dst_fd
                )
            )
            raise OSError(errno.EIO, "reservation replaced")

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = ScriptedOperations(
            native_errno=errno.ENOSYS,
            rename_action=replace_reservation,
        )

        with self.assertRaises(atomic.AmbiguousPostconditionError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )

        self.assertTrue(source.is_dir())
        self.assertTrue(
            atomic.directory_matches_identity(destination, replacement_identity)
        )

    def test_hostile_same_uid_empty_reservation_swap_cannot_be_prevented(self):
        replacement_identity = None

        def swap_after_last_check_then_rename(
            ops, source, destination, src_fd, dst_fd
        ):
            nonlocal replacement_identity
            # This simulates a hostile same-UID writer in the unavoidable interval
            # between the fallback's last identity check and POSIX plain rename.
            atomic.AtomicDirectoryOperations.rmdir(
                ops, destination, dir_fd=dst_fd
            )
            atomic.AtomicDirectoryOperations.mkdir(
                ops, destination, 0o700, dir_fd=dst_fd
            )
            replacement_identity = atomic._identity(
                atomic.AtomicDirectoryOperations.lstat(
                    ops, destination, dir_fd=dst_fd
                )
            )
            atomic.AtomicDirectoryOperations.rename(
                ops,
                source,
                destination,
                src_dir_fd=src_fd,
                dst_dir_fd=dst_fd,
            )

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        source_identity = self.identity(source)
        ops = ScriptedOperations(
            native_errno=errno.ENOSYS,
            rename_action=swap_after_last_check_then_rename,
        )

        result = atomic.atomic_move_directory_noreplace(
            source, destination, trusted_root=root, ops=ops
        )

        self.assertEqual(result.method, "reserved_plain_rename")
        self.assertTrue(atomic.directory_matches_identity(destination, source_identity))
        self.assertFalse(
            atomic.directory_matches_identity(destination, replacement_identity)
        )

    def test_nonempty_reservation_is_ambiguous_and_not_removed(self):
        def fill_reservation(ops, source, destination, src_fd, dst_fd):
            destination_fd = atomic.AtomicDirectoryOperations.open(
                ops, destination, atomic._DIRECTORY_FLAGS, dir_fd=dst_fd
            )
            try:
                marker = atomic.AtomicDirectoryOperations.open(
                    ops,
                    "marker",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=destination_fd,
                )
                atomic.AtomicDirectoryOperations.close(ops, marker)
            finally:
                atomic.AtomicDirectoryOperations.close(ops, destination_fd)
            raise OSError(errno.EIO, "reservation made nonempty")

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = ScriptedOperations(
            native_errno=errno.ENOSYS,
            rename_action=fill_reservation,
        )

        with self.assertRaises(atomic.AmbiguousPostconditionError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )

        self.assertTrue(source.is_dir())
        self.assertEqual([path.name for path in destination.iterdir()], ["marker"])

    def test_both_paths_absent_after_fallback_error_is_ambiguous(self):
        def remove_both(ops, source, destination, src_fd, dst_fd):
            atomic.AtomicDirectoryOperations.rmdir(ops, source, dir_fd=src_fd)
            atomic.AtomicDirectoryOperations.rmdir(ops, destination, dir_fd=dst_fd)
            raise OSError(errno.EIO, "both paths disappeared")

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = ScriptedOperations(
            native_errno=errno.ENOSYS,
            rename_action=remove_both,
        )

        with self.assertRaises(atomic.AmbiguousPostconditionError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )

        self.assertFalse(source.exists())
        self.assertFalse(destination.exists())

    def test_source_replacement_is_ambiguous_and_reservation_remains(self):
        def replace_source(ops, source, destination, src_fd, dst_fd):
            atomic.AtomicDirectoryOperations.rmdir(ops, source, dir_fd=src_fd)
            atomic.AtomicDirectoryOperations.mkdir(
                ops, source, 0o700, dir_fd=src_fd
            )
            raise OSError(errno.EIO, "source replaced")

        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        original_identity = self.identity(source)
        ops = ScriptedOperations(
            native_errno=errno.ENOSYS,
            rename_action=replace_source,
        )

        with self.assertRaises(atomic.AmbiguousPostconditionError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )

        self.assertTrue(source.is_dir())
        self.assertNotEqual(self.identity(source), original_identity)
        self.assertTrue(destination.is_dir())

    def test_fallback_fsync_order_is_source_then_rename_then_parent(self):
        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = ScriptedOperations(native_errno=errno.ENOSYS)

        atomic.atomic_move_directory_noreplace(
            source, destination, trusted_root=root, ops=ops
        )

        native_index = ops.events.index("native")
        mkdir_index = ops.events.index("mkdir:destination")
        rename_index = ops.events.index("plain_rename")
        fsync_indexes = [
            index for index, event in enumerate(ops.events) if event == "fsync"
        ]
        self.assertEqual(len(fsync_indexes), 2)
        self.assertLess(fsync_indexes[0], native_index)
        self.assertLess(native_index, mkdir_index)
        self.assertLess(mkdir_index, rename_index)
        self.assertLess(rename_index, fsync_indexes[1])

    def test_close_failures_attempt_both_fds_without_masking_primary_error(self):
        cases = (
            (None, atomic.AmbiguousPostconditionError, False, True),
            (errno.EIO, atomic.KnownNoChangeMoveError, True, False),
        )
        for native_errno, error_type, source_exists, destination_exists in cases:
            with self.subTest(native_errno=native_errno):
                temporary, root, source, destination = self.make_layout()
                with temporary:
                    ops = CloseFailureOperations(native_errno=native_errno)
                    with self.assertRaises(error_type):
                        atomic.atomic_move_directory_noreplace(
                            source, destination, trusted_root=root, ops=ops
                        )
                    self.assertEqual(len(ops.close_attempts), 2)
                    self.assertEqual(source.exists(), source_exists)
                    self.assertEqual(destination.exists(), destination_exists)


class SafetyValidationTests(AtomicDirectoryTestCase):
    def test_rejects_non_siblings_and_paths_outside_root(self):
        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        nested = root / "nested"
        nested.mkdir()
        outside = root.parent / f"{root.name}-outside"
        outside.mkdir(mode=0o700)
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))

        with self.assertRaises(atomic.UnsafePathError):
            atomic.atomic_move_directory_noreplace(
                source, nested / "destination", trusted_root=root,
                ops=ScriptedOperations(),
            )
        with self.assertRaises(atomic.UnsafePathError):
            atomic.atomic_move_directory_noreplace(
                source, outside / "destination", trusted_root=root,
                ops=ScriptedOperations(),
            )

    def test_rejects_symlink_components_and_symlink_source(self):
        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        real_parent = root / "real"
        real_parent.mkdir(mode=0o700)
        real_source = real_parent / "source"
        real_source.mkdir(mode=0o700)
        linked_parent = root / "linked"
        linked_parent.symlink_to(real_parent, target_is_directory=True)

        with self.assertRaises(atomic.UnsafePathError):
            atomic.atomic_move_directory_noreplace(
                linked_parent / "source",
                linked_parent / "destination",
                trusted_root=root,
                ops=ScriptedOperations(),
            )

        source.rmdir()
        source.symlink_to(real_source, target_is_directory=True)
        with self.assertRaises(atomic.UnsafePathError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ScriptedOperations()
            )

    def test_rejects_group_or_world_writable_parent_and_source(self):
        for unsafe_target in ("parent", "source"):
            with self.subTest(unsafe_target=unsafe_target):
                temporary, root, source, destination = self.make_layout()
                with temporary:
                    target = root if unsafe_target == "parent" else source
                    target.chmod(0o777)
                    try:
                        with self.assertRaises(atomic.UnsafePathError):
                            atomic.atomic_move_directory_noreplace(
                                source,
                                destination,
                                trusted_root=root,
                                ops=ScriptedOperations(),
                            )
                    finally:
                        target.chmod(0o700)

    def test_accepts_readable_nonwritable_group_and_world_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            root.chmod(0o750)
            publication_parent = root / "publication"
            publication_parent.mkdir(mode=0o755)
            publication_parent.chmod(0o755)
            source = publication_parent / "source"
            destination = publication_parent / "destination"
            source.mkdir(mode=0o755)
            source.chmod(0o755)

            result = atomic.atomic_move_directory_noreplace(
                source,
                destination,
                trusted_root=root,
                ops=ScriptedOperations(),
            )

            self.assertEqual(result.method, "native_noreplace")
            self.assertTrue(destination.is_dir())

    def test_rejects_source_device_mismatch(self):
        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        with self.assertRaises(atomic.UnsafePathError):
            atomic.atomic_move_directory_noreplace(
                source,
                destination,
                trusted_root=root,
                ops=FakeDeviceOperations(),
            )

    def test_rejects_unsupported_platform_without_mutation(self):
        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        ops = atomic.AtomicDirectoryOperations(platform="win32")
        with self.assertRaises(atomic.UnsupportedPlatformError):
            atomic.atomic_move_directory_noreplace(
                source, destination, trusted_root=root, ops=ops
            )
        self.assertTrue(source.is_dir())
        self.assertFalse(destination.exists())

    def test_directory_matches_identity_rejects_files_symlinks_and_wrong_inode(self):
        temporary, root, source, destination = self.make_layout()
        self.addCleanup(temporary.cleanup)
        identity = self.identity(source)
        self.assertTrue(atomic.directory_matches_identity(source, identity))
        self.assertFalse(
            atomic.directory_matches_identity(source, (identity[0], identity[1] + 1))
        )
        destination.write_text("file")
        self.assertFalse(atomic.directory_matches_identity(destination, self.identity(destination)))
        link = root / "link"
        link.symlink_to(source, target_is_directory=True)
        self.assertFalse(atomic.directory_matches_identity(link, identity))


class ProbeAndCliTests(AtomicDirectoryTestCase):
    def test_probe_exercises_move_collision_and_leaves_no_residue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            before = set(root.iterdir())
            ops = ScriptedOperations()
            result = atomic.probe_atomic_directory_move(
                root,
                trusted_root=root,
                ops=ops,
            )
            self.assertEqual(result.method, "native_noreplace")
            self.assertEqual(ops.events.count("native"), 2)
            self.assertEqual(set(root.iterdir()), before)

    def test_probe_uses_fallback_when_native_reports_unsupported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            ops = ScriptedOperations(native_errno=errno.ENOSYS)
            result = atomic.probe_atomic_directory_move(
                root,
                trusted_root=root,
                ops=ops,
            )
            self.assertEqual(result.method, "reserved_plain_rename")
            self.assertEqual(ops.events.count("mkdir:destination"), 2)
            self.assertEqual(list(root.iterdir()), [])

    def test_probe_root_mkdir_error_after_creation_is_ambiguous_and_left_untouched(self):
        class ProbeMkdirErrorOperations(ScriptedOperations):
            def mkdir(self, path, mode=0o777, *, dir_fd=None):
                name = os.fsdecode(path)
                if dir_fd is not None and name.startswith(".oprd-atomic-probe-"):
                    atomic.AtomicDirectoryOperations.mkdir(
                        self, path, mode, dir_fd=dir_fd
                    )
                    raise OSError(errno.EIO, "reported after probe mkdir")
                return super().mkdir(path, mode, dir_fd=dir_fd)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with self.assertRaises(atomic.AmbiguousPostconditionError):
                atomic.probe_atomic_directory_move(
                    root,
                    trusted_root=root,
                    ops=ProbeMkdirErrorOperations(),
                )
            residue = list(root.iterdir())
            self.assertEqual(len(residue), 1)
            self.assertTrue(residue[0].name.startswith(".oprd-atomic-probe-"))
            self.assertTrue(residue[0].is_dir())

    def test_cli_move_and_collision_emit_json_with_exit_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir(mode=0o700)
            command = [
                sys.executable,
                str(pathlib.Path(atomic.__file__).resolve()),
                "move",
                "--source",
                str(source),
                "--destination",
                str(destination),
                "--trusted-root",
                str(root),
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["ok"])
            self.assertIn(
                payload["result"]["method"],
                ("native_noreplace", "reserved_plain_rename"),
            )
            self.assertNotIn("destination_identity", payload["result"])

            source.mkdir(mode=0o700)
            collided = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(collided.returncode, 1, collided.stderr)
            error_payload = json.loads(collided.stdout)
            self.assertFalse(error_payload["ok"])
            self.assertEqual(
                error_payload["error"]["type"], "DestinationCollisionError"
            )

    def test_cli_serializes_expected_raw_exceptions_as_json(self):
        for error in (
            OSError(errno.EIO, "raw io error"),
            ValueError("raw value error"),
            TypeError("raw type error"),
        ):
            with self.subTest(error_type=type(error).__name__):
                output = io.StringIO()
                with mock.patch.object(
                    atomic,
                    "atomic_move_directory_noreplace",
                    side_effect=error,
                ), contextlib.redirect_stdout(output):
                    status = atomic.main(
                        [
                            "move",
                            "--source",
                            "/tmp/source",
                            "--destination",
                            "/tmp/destination",
                            "--trusted-root",
                            "/tmp",
                        ]
                    )
                self.assertEqual(status, 1)
                payload = json.loads(output.getvalue())
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["error"]["type"], type(error).__name__)

    def test_cli_probe_emits_json_and_cleans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(pathlib.Path(atomic.__file__).resolve()),
                    "probe",
                    "--parent",
                    str(root),
                    "--trusted-root",
                    str(root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertTrue(payload["ok"])
            self.assertIn(
                payload["result"]["move"]["method"],
                ("native_noreplace", "reserved_plain_rename"),
            )
            self.assertEqual(len(payload["result"]["probe_root_identity"]), 2)
            self.assertNotIn("method", payload["result"])
            self.assertNotIn("collision_verified", payload["result"])
            self.assertNotIn("cleaned", payload["result"])
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
