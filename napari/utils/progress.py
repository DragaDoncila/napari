import inspect
from typing import Iterable, Optional

from qtpy import QtCore
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QWidget,
)
from tqdm import tqdm

from .._qt.utils import get_viewer_instance


def get_pbar(viewer_instance, **kwargs):
    """Adds ProgressBar to viewer Activity Dock and returns it.

    Parameters
    ----------
    viewer_instance : qtViewer
        current napari qtViewer instance

    Returns
    -------
    ProgressBar
        progress bar to associate with current iterable
    """
    pbar = ProgressBar(**kwargs)
    viewer_instance.activityDock.widget().layout.addWidget(pbar)

    return pbar


def get_calling_function_name(max_depth: int):
    """Inspect stack up to max_depth and return first function name outside of progress.py"""
    for finfo in inspect.stack()[2:max_depth]:
        if not finfo.filename.endswith("progress.py"):
            return finfo.function

    return None


_tqdm_kwargs = {
    p.name
    for p in inspect.signature(tqdm.__init__).parameters.values()
    if p.kind is not inspect.Parameter.VAR_KEYWORD and p.name != "self"
}


class progress(tqdm):
    """This class inherits from tqdm and provides an interface for
    progress bars in the napari viewer. Progress bars can be created
    directly by wrapping an iterable or by providing a total number
    of expected updates.

    See tqdm.tqdm API for valid args and kwargs: https://tqdm.github.io/docs/tqdm/

    Also, any keyword arguments to the :class:`ProgressBar` `QWidget`
    are also accepted and will be passed to the ``ProgressBar``.

    Examples
    --------

    >>> def long_running(steps=10, delay=0.1):
    ...     for i in progress(range(steps)):
    ...         sleep(delay)

    it can also be used as a context manager:

    >>> def long_running(steps=10, repeats=4, delay=0.1):
    ...     with progress(range(steps)) as pbr:
    ...         for i in pbr:
    ...             sleep(delay)

    or for manual updates:

    >>> def manual_updates(total):
    ...     pbr = progress(total=total)
    ...     sleep(10)
    ...     pbr.set_description("Step 1 Complete")
    ...     pbr.update(1)

    """

    def __init__(
        self,
        iterable: Optional[Iterable] = None,
        desc: Optional[str] = None,
        total: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:

        # check if there's a napari viewer instance
        viewer = get_viewer_instance()
        self.has_viewer = viewer is not None
        if self.has_viewer:
            kwargs['gui'] = True

        kwargs = kwargs.copy()
        pbar_kwargs = {k: kwargs.pop(k) for k in set(kwargs) - _tqdm_kwargs}

        super().__init__(iterable, desc, total, *args, **kwargs)
        if not self.has_viewer:
            return

        self._pbar = get_pbar(viewer, **pbar_kwargs)
        if self.total is not None:
            self._pbar.setRange(self.n, self.total)
            self._pbar._set_value(self.n)
        else:
            self._pbar.setRange(0, 0)

        if desc:
            self.set_description(desc)
        else:
            desc = get_calling_function_name(max_depth=5)
            if desc:
                self.set_description(desc)
            else:
                # TODO: pick a better default
                self.set_description("Progress Bar")

        self.show()
        QApplication.processEvents()

    def display(self, msg: str = None, pos: int = None) -> None:
        """Update the display."""
        if not self.has_viewer:
            return super().display(msg=msg, pos=pos)

        eta_params = {
            k: self.format_dict[k]
            for k in [
                'n',
                'total',
                'elapsed',
                'unit',
                'unit_scale',
                'rate',
                'unit_divisor',
                'initial',
            ]
        }
        etas = self.format_time(**eta_params)

        self._pbar._set_value(self.n)
        self._pbar._set_eta(etas)
        QApplication.processEvents()

    def set_description(self, desc):
        """Update progress bar description"""
        super().set_description(desc, refresh=True)
        if self.has_viewer:
            self._pbar._set_description(self.desc)

    @staticmethod
    def format_time(
        n,
        total,
        elapsed,
        unit='it',
        unit_scale=False,
        rate=None,
        unit_divisor=1000,
        initial=0,
    ):
        """Formats iteration and time estimates for display.

        Taken from tqdm.format_meter, this function computes and formats estimates
        for iterations per second, iterations remaining and current elapsed time.

        All parameters are filtered from self.format_dict

        Parameters
        ----------
        n  : int or float
            Number of finished iterations.
        total  : int or float
            The expected total number of iterations. If meaningless (None),
            only basic progress statistics are displayed (no ETA).
        elapsed  : float
            Number of seconds passed since start.
        unit  : str, optional
            The iteration unit [default: 'it'].
        unit_scale  : bool or int or float, optional
            If 1 or True, the number of iterations will be printed with an
            appropriate SI metric prefix (k = 10^3, M = 10^6, etc.)
            [default: False]. If any other non-zero number, will scale
            `total` and `n`.
        rate  : float, optional
            Manual override for iteration rate.
            If [default: None], uses n/elapsed.
        unit_divisor  : float, optional
            [default: 1000], ignored unless `unit_scale` is True.
        initial  : int or float, optional
            The initial counter value [default: 0].

        Returns
        -------
        str
            formatted estimates ready for display
        """

        # sanity check: total
        if total and n >= (total + 0.5):  # allow float imprecision (#849)
            total = None

        # apply custom scale if necessary
        if unit_scale and unit_scale not in (True, 1):
            if total:
                total *= unit_scale
            n *= unit_scale
            if rate:
                rate *= (
                    unit_scale  # by default rate = self.avg_dn / self.avg_dt
                )
            unit_scale = False

        elapsed_str = tqdm.format_interval(elapsed)
        # if unspecified, attempt to use rate = average speed
        # (we allow manual override since predicting time is an arcane art)
        if rate is None and elapsed:
            rate = (n - initial) / elapsed
        inv_rate = 1 / rate if rate else None
        format_sizeof = tqdm.format_sizeof
        rate_noinv_fmt = (
            (
                (format_sizeof(rate) if unit_scale else f'{rate:5.2f}')
                if rate
                else '?'
            )
            + unit
            + '/s'
        )
        rate_inv_fmt = (
            (
                (format_sizeof(inv_rate) if unit_scale else f'{inv_rate:5.2f}')
                if inv_rate
                else '?'
            )
            + 's/'
            + unit
        )
        rate_fmt = (
            rate_inv_fmt if inv_rate and inv_rate > 1 else rate_noinv_fmt
        )

        if unit_scale:
            n_fmt = format_sizeof(n, divisor=unit_divisor)
            total_fmt = (
                format_sizeof(total, divisor=unit_divisor)
                if total is not None
                else '?'
            )
        else:
            n_fmt = str(n)
            total_fmt = str(total) if total is not None else '?'

        remaining = (total - n) / rate if rate and total else 0
        remaining_str = tqdm.format_interval(remaining) if rate else '?'

        bar_etas = ' {}/{} [{}<{}, {}]'.format(
            n_fmt, total_fmt, elapsed_str, remaining_str, rate_fmt
        )

        return bar_etas

    def hide(self):
        """Hide the progress bar"""
        if self.has_viewer:
            self._pbar.hide()

    def show(self):
        """Show the progress bar"""
        if self.has_viewer:
            self._pbar.show()

    def close(self):
        """Closes and deletes the progress bar widget"""
        if self.disable:
            return
        if self.has_viewer:
            self._pbar.close()
        super().close()


class ProgressBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.pbar = QProgressBar()
        self.description_label = QLabel()
        self.eta_label = QLabel()

        layout = QHBoxLayout()
        layout.addWidget(self.description_label)
        layout.addWidget(self.pbar)
        layout.addWidget(self.eta_label)
        self.setLayout(layout)

    def setRange(self, min, max):
        self.pbar.setRange(min, max)

    def _set_value(self, value):
        self.pbar.setValue(value)

    def _get_value(self):
        return self.pbar.value()

    def _set_description(self, desc):
        self.description_label.setText(desc)

    def _set_eta(self, eta):
        self.eta_label.setText(eta)