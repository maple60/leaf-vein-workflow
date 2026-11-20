"""Relay progress from a worker thread to the dock's progress bar.

Long steps (Auto-assign 1°/2°, Export) run in napari worker threads, where
Qt widgets must not be touched.  A plain callback ``report(fraction, message)``
is handed to the computation; it emits a Qt signal, which Qt delivers on the
main thread because the reporter object lives there.
"""

from qtpy.QtCore import QObject, Signal


class ProgressReporter(QObject):
    """Callable that forwards ``(fraction, message)`` to the main thread."""

    progressed = Signal(float, str)

    def __call__(self, fraction: float, message: str) -> None:
        self.progressed.emit(float(fraction), str(message))
