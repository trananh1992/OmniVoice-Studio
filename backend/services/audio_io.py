"""Atomic disk writes for audio files.

A direct ``torchaudio.save(path, ...)`` writes bytes to ``path`` as the
encoder produces them. If the process is killed mid-write (SIGKILL, OOM
kill, power loss, Tauri sidecar reap), the file at ``path`` exists but is
truncated. Downstream tools — ffmpeg in the dub mux step, NLEs the user
imports the WAV into — often happily read a truncated RIFF (the header
appears first, then the data chunk gets cut short), producing silently
corrupt audio later in the pipeline. That is the root of issue #48.

``atomic_save_wav`` writes to a sibling temp file in the same directory and
``os.replace()`` it into place once encoding completes. POSIX guarantees
``rename(2)`` is atomic on the same filesystem; ``os.replace()`` makes the
same guarantee portable to Windows, including the case where the target
path already exists.

Either the new file fully exists at ``target_path`` after the call returns,
or the target keeps its previous contents (or never existed). There is no
intermediate window where a partial WAV is visible at ``target_path``.

Closes #48.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

import torch
import torchaudio

logger = logging.getLogger("omnivoice.audio_io")


def atomic_save_wav(
    target_path: str,
    audio: torch.Tensor,
    sample_rate: int,
    **kwargs: Any,
) -> None:
    """Write a WAV to ``target_path`` atomically.

    Implementation: write to a sibling temp file in the same directory, then
    ``os.replace()`` into place. Cross-filesystem renames are *not* atomic
    on POSIX, so the temp file must live next to the target — that is why
    we use ``dir=target_dir`` instead of the system temp dir.

    Args:
        target_path: Final destination. Parent directory must already exist.
        audio: ``(channels, samples)`` tensor — the same shape
            ``torchaudio.save`` expects.
        sample_rate: WAV sample rate in Hz.
        **kwargs: Forwarded to ``torchaudio.save``.

    Raises:
        Whatever ``torchaudio.save`` raises. The temp file is unlinked on
        failure so we do not leak ``.tmp`` files in ``DUB_DIR``.
    """
    target_dir = os.path.dirname(target_path) or "."
    target_base = os.path.basename(target_path)
    # The temp file must end in ``.wav`` even though it is conceptually a
    # ``.tmp`` file. torchaudio.save infers the output format from the path
    # suffix and *ignores* the ``format=`` kwarg with the soundfile backend
    # — a ``.tmp`` suffix raises ``ValueError: Unsupported format: tmp``.
    # The leading dot + ``target_base`` prefix still marks the file as
    # transient and groups it next to its target in directory listings.
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{target_base}.",
        suffix=".wav",
        dir=target_dir,
    )
    os.close(fd)  # torchaudio reopens by path; we just needed a unique name.
    try:
        torchaudio.save(tmp_path, audio, sample_rate, **kwargs)
        os.replace(tmp_path, target_path)
    except BaseException:
        # BaseException so we clean up on KeyboardInterrupt + SystemExit too.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
