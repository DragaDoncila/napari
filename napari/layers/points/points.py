from collections.abc import Sequence
from copy import copy, deepcopy
from typing import (
    TYPE_CHECKING,
    Optional,
    Union,
)

import numpy as np
import numpy.typing as npt
from psygnal.containers import Selection
from scipy.stats import gmean

from napari.layers.base._base_constants import ActionType
from napari.layers.points._coordinates import _Coordinates
from napari.layers.points._points_constants import (
    Mode,
)
from napari.layers.points._points_utils import (
    _create_box_from_corners_3d,
    create_box,
)
from napari.layers.points._slice import _PointSliceRequest, _PointSliceResponse
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.layers.utils.layer_utils import (
    _features_to_properties,
)
from napari.utils.events.event_utils import mirror_event
from napari.utils.events.migrations import deprecation_warning_event
from napari.utils.migrations import add_deprecated_property, rename_argument
from napari.utils.status_messages import generate_layer_coords_status
from napari.utils.transforms import Affine

if TYPE_CHECKING:
    from napari.components.dims import Dims

DEFAULT_COLOR_CYCLE = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])


class Points(_Coordinates):
    """Points layer.

    Parameters
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    ndim : int
        Number of dimensions for shapes. When data is not None, ndim must be D.
        An empty points layer can be instantiated with arbitrary ndim.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    antialiasing: float
        Amount of antialiasing in canvas pixels.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', 'translucent_no_depth', 'additive', and 'minimum'}.
    border_color : str, array-like, dict
        Color of the point marker border. Numeric color values should be RGB(A).
    border_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to border_color if a
        categorical attribute is used color the vectors.
    border_colormap : str, napari.utils.Colormap
        Colormap to set border_color if a continuous attribute is used to set face_color.
    border_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    border_width : float, array
        Width of the symbol border in pixels.
    border_width_is_relative : bool
        If enabled, border_width is interpreted as a fraction of the point size.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    canvas_size_limits : tuple of float
        Lower and upper limits for the size of points in canvas pixels.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.
    face_color : str, array-like, dict
        Color of the point marker body. Numeric color values should be RGB(A).
    face_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to face_color if a
        categorical attribute is used color the vectors.
    face_colormap : str, napari.utils.Colormap
        Colormap to set face_color if a continuous attribute is used to set face_color.
    face_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    feature_defaults : dict[str, Any] or DataFrame
        The default value of each feature in a table with one row.
    features : dict[str, array-like] or DataFrame
        Features table where each row corresponds to a point and each column
        is a feature.
    metadata : dict
        Layer metadata.
    n_dimensional : bool
        This property will soon be deprecated in favor of 'out_of_slice_display'.
        Use that instead.
    name : str
        Name of the layer. If not provided then will be guessed using heuristics.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    out_of_slice_display : bool
        If True, renders points not just in central plane but also slightly out of slice
        according to specified point marker size.
    projection_mode : str
        How data outside the viewed dimensions but inside the thick Dims slice will
        be projected onto the viewed dimensions. Must fit to cls._projectionclass.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
    property_choices : dict {str: array (N,)}
        possible values for each property.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    scale : tuple of float
        Scale factors for the layer.
    shading : str, Shading
        Render lighting and shading on points. Options are:

        * 'none'
          No shading is added to the points.
        * 'spherical'
          Shading and depth buffer are changed to give a 3D spherical look to the points
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    shown : 1-D array of bool
        Whether to show each point.
    size : float, array
        Size of the point marker in data pixels. If given as a scalar, all points are made
        the same size. If given as an array, size must be the same or broadcastable
        to the same shape as the data.
    symbol : str, array
        Symbols to be used for the point markers. Must be one of the
        following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x.
    text : str, dict
        Text to be displayed with the points. If text is set to a key in properties,
        the value of that property will be displayed. Multiple properties can be
        composed using f-string-like syntax (e.g., '{property_1}, {float_property:.2f}).
        A dictionary can be provided with keyword arguments to set the text values
        and display properties. See TextManager.__init__() for the valid keyword arguments.
        For example usage, see /napari/examples/add_points_with_text.py.
    translate : tuple of float
        Translation values for the layer.
    visible : bool
        Whether the layer visual is currently being displayed.

    Attributes
    ----------
    data : array (N, D)
        Coordinates for N points in D dimensions.
    features : DataFrame-like
        Features table where each row corresponds to a point and each column
        is a feature.
    feature_defaults : DataFrame-like
        Stores the default value of each feature in a table with one row.
    properties : dict {str: array (N,)} or DataFrame
        Annotations for each point. Each property should be an array of length N,
        where N is the number of points.
    text : str
        Text to be displayed with the points. If text is set to a key in properties, the value of
        that property will be displayed. Multiple properties can be composed using f-string-like
        syntax (e.g., '{property_1}, {float_property:.2f}).
        For example usage, see /napari/examples/add_points_with_text.py.
    symbol : array of str
        Array of symbols for each point.
    size : array (N,)
        Array of sizes for each point. Must have the same shape as the layer `data`.
    border_width : array (N,)
        Width of the marker borders in pixels for all points
    border_width : array (N,)
        Width of the marker borders for all points as a fraction of their size.
    border_color : Nx4 numpy array
        Array of border color RGBA values, one for each point.
    border_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to border_color if a
        categorical attribute is used color the vectors.
    border_colormap : str, napari.utils.Colormap
        Colormap to set border_color if a continuous attribute is used to set face_color.
    border_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    face_color : Nx4 numpy array
        Array of face color RGBA values, one for each point.
    face_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to face_color if a
        categorical attribute is used color the vectors.
    face_colormap : str, napari.utils.Colormap
        Colormap to set face_color if a continuous attribute is used to set face_color.
    face_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    current_symbol : Symbol
        Symbol for the next point to be added or the currently selected points.
    current_size : float
        Size of the marker for the next point to be added or the currently
        selected point.
    current_border_width : float
        Border width of the marker for the next point to be added or the currently
        selected point.
    current_border_color : str
        Border color of the marker border for the next point to be added or the currently
        selected point.
    current_face_color : str
        Face color of the marker border for the next point to be added or the currently
        selected point.
    out_of_slice_display : bool
        If True, renders points not just in central plane but also slightly out of slice
        according to specified point marker size.
    selected_data : Selection
        Integer indices of any selected points.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In ADD mode clicks of the cursor add points at the clicked location.

        In SELECT mode the cursor can select points by clicking on them or
        by dragging a box around them. Once selected points can be moved,
        have their properties edited, or be deleted.
    face_color_mode : str
        Face color setting mode.

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
    border_color_mode : str
        Border color setting mode.

        DIRECT (default mode) allows each point to be set arbitrarily

        CYCLE allows the color to be set via a color cycle over an attribute

        COLORMAP allows color to be set via a color map over an attribute
    shading : Shading
        Shading mode.
    antialiasing: float
        Amount of antialiasing in canvas pixels.
    canvas_size_limits : tuple of float
        Lower and upper limits for the size of points in canvas pixels.
    shown : 1-D array of bool
        Whether each point is shown.

    Notes
    -----
    _view_data : array (M, D)
        coordinates of points in the currently viewed slice.
    _view_size : array (M, )
        Size of the point markers in the currently viewed slice.
    _view_symbol : array (M, )
        Symbols of the point markers in the currently viewed slice.
    _view_border_width : array (M, )
        Border width of the point markers in the currently viewed slice.
    _indices_view : array (M, )
        Integer indices of the points in the currently viewed slice and are shown.
    _selected_view :
        Integer indices of selected points in the currently viewed slice within
        the `_view_data` array.
    _selected_box : array (4, 2) or None
        Four corners of any box either around currently selected points or
        being created during a drag action. Starting in the top left and
        going clockwise.
    _drag_start : list or None
        Coordinates of first cursor click during a drag action. Gets reset to
        None after dragging is done.
    """

    # TODO  write better documentation for border_color and face_color

    @rename_argument(
        'edge_width', 'border_width', since_version='0.5.0', version='0.6.0'
    )
    @rename_argument(
        'edge_width_is_relative',
        'border_width_is_relative',
        since_version='0.5.0',
        version='0.6.0',
    )
    @rename_argument(
        'edge_color', 'border_color', since_version='0.5.0', version='0.6.0'
    )
    @rename_argument(
        'edge_color_cycle',
        'border_color_cycle',
        since_version='0.5.0',
        version='0.6.0',
    )
    @rename_argument(
        'edge_colormap',
        'border_colormap',
        since_version='0.5.0',
        version='0.6.0',
    )
    @rename_argument(
        'edge_contrast_limits',
        'border_contrast_limits',
        since_version='0.5.0',
        version='0.6.0',
    )
    def __init__(
        self,
        data=None,
        ndim=None,
        *,
        affine=None,
        antialiasing=1,
        blending='translucent',
        border_color='dimgray',
        border_color_cycle=None,
        border_colormap='viridis',
        border_contrast_limits=None,
        border_width=0.05,
        border_width_is_relative=True,
        cache=True,
        canvas_size_limits=(2, 10000),
        experimental_clipping_planes=None,
        face_color='white',
        face_color_cycle=None,
        face_colormap='viridis',
        face_contrast_limits=None,
        feature_defaults=None,
        features=None,
        metadata=None,
        n_dimensional=None,
        name=None,
        opacity=1.0,
        out_of_slice_display=False,
        projection_mode='none',
        properties=None,
        property_choices=None,
        rotate=None,
        scale=None,
        shading='none',
        shear=None,
        shown=True,
        size=10,
        symbol='o',
        text=None,
        translate=None,
        visible=True,
    ) -> None:
        super().__init__(
            data=data,
            ndim=ndim,
            affine=affine,
            antialiasing=antialiasing,
            blending=blending,
            border_color=border_color,
            border_color_cycle=border_color_cycle,
            border_colormap=border_colormap,
            border_contrast_limits=border_contrast_limits,
            border_width=border_width,
            border_width_is_relative=border_width_is_relative,
            cache=cache,
            canvas_size_limits=canvas_size_limits,
            experimental_clipping_planes=experimental_clipping_planes,
            face_color=face_color,
            face_color_cycle=face_color_cycle,
            face_colormap=face_colormap,
            face_contrast_limits=face_contrast_limits,
            feature_defaults=feature_defaults,
            features=features,
            metadata=metadata,
            n_dimensional=n_dimensional,
            name=name,
            opacity=opacity,
            out_of_slice_display=out_of_slice_display,
            projection_mode=projection_mode,
            properties=properties,
            property_choices=property_choices,
            rotate=rotate,
            scale=scale,
            shading=shading,
            shear=shear,
            shown=shown,
            size=size,
            symbol=symbol,
            text=text,
            translate=translate,
            visible=visible,
        )

        deprecated_events = {}
        for attr in [
            '{}_width',
            'current_{}_width',
            '{}_width_is_relative',
            '{}_color',
            'current_{}_color',
        ]:
            old_attr = attr.format('edge')
            new_attr = attr.format('border')
            old_emitter = deprecation_warning_event(
                'layer.events',
                old_attr,
                new_attr,
                since_version='0.5.0',
                version='0.6.0',
            )
            getattr(self.events, new_attr).connect(old_emitter)
            deprecated_events[old_attr] = old_emitter

        self.events.add(**deprecated_events)
        mirror_event(
            self.events.coordinates,
            self.events.data,
            ['data_indices', 'vertex_indices', 'action', 'value'],
        )

    @classmethod
    def _add_deprecated_properties(cls) -> None:
        """Adds deprecated properties to class."""
        deprecated_properties = [
            'edge_width',
            'edge_width_is_relative',
            'current_edge_width',
            'edge_color',
            'edge_color_cycle',
            'edge_colormap',
            'edge_contrast_limits',
            'current_edge_color',
            'edge_color_mode',
        ]
        for old_property in deprecated_properties:
            new_property = old_property.replace('edge', 'border')
            add_deprecated_property(
                cls,
                old_property,
                new_property,
                since_version='0.5.0',
                version='0.6.0',
            )

    @property
    def data(self) -> np.ndarray:
        """(N, D) array: coordinates for N points in D dimensions."""
        return self._coordinates

    @data.setter
    def data(self, data: Optional[np.ndarray]) -> None:
        """Set the data array and emit a corresponding event."""
        self.coordinates = data

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = self._get_base_state()
        state.update(
            {
                'symbol': (
                    self.symbol if self.data.size else [self.current_symbol]
                ),
                'border_width': self.border_width,
                'border_width_is_relative': self.border_width_is_relative,
                'face_color': (
                    self.face_color
                    if self.data.size
                    else [self.current_face_color]
                ),
                'face_color_cycle': self.face_color_cycle,
                'face_colormap': self.face_colormap.dict(),
                'face_contrast_limits': self.face_contrast_limits,
                'border_color': (
                    self.border_color
                    if self.data.size
                    else [self.current_border_color]
                ),
                'border_color_cycle': self.border_color_cycle,
                'border_colormap': self.border_colormap.dict(),
                'border_contrast_limits': self.border_contrast_limits,
                'properties': self.properties,
                'property_choices': self.property_choices,
                'text': self.text.dict(),
                'out_of_slice_display': self.out_of_slice_display,
                'n_dimensional': self.out_of_slice_display,
                'size': self.size,
                'ndim': self.ndim,
                'data': self.data,
                'coordinates': self.coordinates,
                'features': self.features,
                'feature_defaults': self.feature_defaults,
                'shading': self.shading,
                'antialiasing': self.antialiasing,
                'canvas_size_limits': self.canvas_size_limits,
                'shown': self.shown,
            }
        )
        return state

    @property
    def _view_data(self) -> np.ndarray:
        """Get the coords of the points in view

        Returns
        -------
        view_data : (N x D) np.ndarray
            Array of coordinates for the N points in view
        """
        return self._view_coordinates

    def _set_view_slice(self) -> None:
        """Sets the view given the indices to slice with."""

        # The new slicing code makes a request from the existing state and
        # executes the request on the calling thread directly.
        # For async slicing, the calling thread will not be the main thread.
        request = self._make_slice_request_internal(
            self._slice_input, self._data_slice
        )
        response = request()
        self._update_slice_response(response)

    def _make_slice_request(self, dims: 'Dims') -> _PointSliceRequest:
        """Make a Points slice request based on the given dims and these data."""
        slice_input = self._make_slice_input(dims)
        # See Image._make_slice_request to understand why we evaluate this here
        # instead of using `self._data_slice`.
        data_slice = slice_input.data_slice(self._data_to_world.inverse)
        return self._make_slice_request_internal(slice_input, data_slice)

    def _make_slice_request_internal(
        self, slice_input: _SliceInput, data_slice: _ThickNDSlice
    ) -> _PointSliceRequest:
        return _PointSliceRequest(
            slice_input=slice_input,
            data=self.data,
            data_slice=data_slice,
            projection_mode=self.projection_mode,
            out_of_slice_display=self.out_of_slice_display,
            size=self.size,
        )

    def _update_slice_response(self, response: _PointSliceResponse) -> None:
        """Handle a slicing response."""
        self._slice_input = response.slice_input
        indices = response.indices
        scale = response.scale

        # Update the _view_size_scale in accordance to the self._indices_view setter.
        # If out_of_slice_display is False, scale is a number and not an array.
        # Therefore we have an additional if statement checking for
        # self._view_size_scale being an integer.
        if not isinstance(scale, np.ndarray):
            self._view_size_scale = scale
        elif len(self._shown) == 0:
            self._view_size_scale = np.empty(0, int)
        else:
            self._view_size_scale = scale[self.shown[indices]]

        self._indices_view = np.array(indices, dtype=int)
        # get the selected points that are in view
        self._selected_view = list(
            np.intersect1d(
                np.array(list(self._selected_data)),
                self._indices_view,
                return_indices=True,
            )[2]
        )
        with self.events.highlight.blocker():
            self._set_highlight(force=True)

    def _set_highlight(self, force: bool = False) -> None:
        """Render highlights of shapes including boundaries, vertices,
        interaction boxes, and the drag selection box when appropriate.
        Highlighting only occurs in Mode.SELECT.

        Parameters
        ----------
        force : bool
            Bool that forces a redraw to occur when `True`
        """
        # Check if any point ids have changed since last call
        if (
            self.selected_data == self._selected_data_stored
            and self._value == self._value_stored
            and np.array_equal(self._drag_box, self._drag_box_stored)
        ) and not force:
            return
        self._selected_data_stored = Selection(self.selected_data)
        self._value_stored = copy(self._value)
        self._drag_box_stored = copy(self._drag_box)

        if self._value is not None or len(self._selected_view) > 0:
            if len(self._selected_view) > 0:
                index = copy(self._selected_view)
                # highlight the hovered point if not in adding mode
                if (
                    self._value in self._indices_view
                    and self._mode == Mode.SELECT
                    and not self._is_selecting
                ):
                    hover_point = list(self._indices_view).index(self._value)
                    if hover_point not in index:
                        index.append(hover_point)
                index.sort()
            else:
                # only highlight hovered points in select mode
                if (
                    self._value in self._indices_view
                    and self._mode == Mode.SELECT
                    and not self._is_selecting
                ):
                    hover_point = list(self._indices_view).index(self._value)
                    index = [hover_point]
                else:
                    index = []

            self._highlight_index = index
        else:
            self._highlight_index = []

        # only display dragging selection box in 2D
        if self._is_selecting:
            if self._drag_normal is None:
                pos = create_box(self._drag_box)
            else:
                pos = _create_box_from_corners_3d(
                    self._drag_box, self._drag_normal, self._drag_up
                )
            pos = pos[[*range(4), 0]]
        else:
            pos = None

        self._highlight_box = pos
        self.events.highlight()

    def _update_thumbnail(self) -> None:
        """Update thumbnail with current points and colors."""
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        view_data = self._view_data
        if len(view_data) > 0:
            # Get the zoom factor required to fit all data in the thumbnail.
            de = self._extent_data
            min_vals = [de[0, i] for i in self._slice_input.displayed]
            shape = np.ceil(
                [de[1, i] - de[0, i] + 1 for i in self._slice_input.displayed]
            ).astype(int)
            zoom_factor = np.divide(
                self._thumbnail_shape[:2], shape[-2:]
            ).min()

            # Maybe subsample the points.
            if len(view_data) > self._max_points_thumbnail:
                thumbnail_indices = np.random.randint(
                    0, len(view_data), self._max_points_thumbnail
                )
                points = view_data[thumbnail_indices]
            else:
                points = view_data
                thumbnail_indices = self._indices_view

            # Calculate the point coordinates in the thumbnail data space.
            thumbnail_shape = np.clip(
                np.ceil(zoom_factor * np.array(shape[:2])).astype(int),
                1,  # smallest side should be 1 pixel wide
                self._thumbnail_shape[:2],
            )
            coords = np.floor(
                (points[:, -2:] - min_vals[-2:] + 0.5) * zoom_factor
            ).astype(int)
            coords = np.clip(coords, 0, thumbnail_shape - 1)

            # Draw single pixel points in the colormapped thumbnail.
            colormapped = np.zeros((*thumbnail_shape, 4))
            colormapped[..., 3] = 1
            colors = self._face.colors[thumbnail_indices]
            colormapped[coords[:, 0], coords[:, 1]] = colors

        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def add(self, coords):
        """Adds points at coordinates.

        Parameters
        ----------
        coords : array
            Point or points to add to the layer data.
        """
        cur_points = len(self.data)
        self.events.data(
            value=self.data,
            action=ActionType.ADDING,
            data_indices=(-1,),
            vertex_indices=((),),
        )
        self._set_data(np.append(self.data, np.atleast_2d(coords), axis=0))
        self.events.data(
            value=self.data,
            action=ActionType.ADDED,
            data_indices=(-1,),
            vertex_indices=((),),
        )
        self.selected_data = set(np.arange(cur_points, len(self.data)))

    def remove_selected(self) -> None:
        """Removes selected points if any."""
        index = list(self.selected_data)
        index.sort()
        if len(index):
            self.events.data(
                value=self.data,
                action=ActionType.REMOVING,
                data_indices=tuple(
                    self.selected_data,
                ),
                vertex_indices=((),),
            )
            self._shown = np.delete(self._shown, index, axis=0)
            self._size = np.delete(self._size, index, axis=0)
            self._symbol = np.delete(self._symbol, index, axis=0)
            self._border_width = np.delete(self._border_width, index, axis=0)
            with self._border.events.blocker_all():
                self._border._remove(indices_to_remove=index)
            with self._face.events.blocker_all():
                self._face._remove(indices_to_remove=index)
            self._feature_table.remove(index)
            self.text.remove(index)
            if self._value in self.selected_data:
                self._value = None
            else:
                if self._value is not None:
                    # update the index of self._value to account for the
                    # data being removed
                    indices_removed = np.array(index) < self._value
                    offset = np.sum(indices_removed)
                    self._value -= offset
                    self._value_stored -= offset

            self._set_data(np.delete(self.data, index, axis=0))
            self.events.data(
                value=self.data,
                action=ActionType.REMOVED,
                data_indices=tuple(
                    self.selected_data,
                ),
                vertex_indices=((),),
            )
            self.selected_data = set()

    def _move(
        self,
        selection_indices: Sequence[int],
        position: Sequence[Union[int, float]],
    ) -> None:
        """Move points relative to drag start location.

        Parameters
        ----------
        selection_indices : Sequence[int]
            Integer indices of points to move in self.data
        position : tuple
            Position to move points to in data coordinates.
        """
        if len(selection_indices) > 0:
            selection_indices = list(selection_indices)
            disp = list(self._slice_input.displayed)
            self._set_drag_start(selection_indices, position)
            center = self.data[np.ix_(selection_indices, disp)].mean(axis=0)
            shift = np.array(position)[disp] - center - self._drag_start
            self.data[np.ix_(selection_indices, disp)] = (
                self.data[np.ix_(selection_indices, disp)] + shift
            )
            self.refresh()
            self.events.data(
                value=self.data,
                action=ActionType.CHANGED,
                data_indices=tuple(selection_indices),
                vertex_indices=((),),
            )

    def _set_drag_start(
        self,
        selection_indices: Sequence[int],
        position: Sequence[Union[int, float]],
        center_by_data: bool = True,
    ) -> None:
        """Store the initial position at the start of a drag event.

        Parameters
        ----------
        selection_indices : set of int
            integer indices of selected data used to index into self.data
        position : Sequence of numbers
            position of the drag start in data coordinates.
        center_by_data : bool
            Center the drag start based on the selected data.
            Used for modifier drag_box selection.
        """
        selection_indices = list(selection_indices)
        dims_displayed = list(self._slice_input.displayed)
        if self._drag_start is None:
            self._drag_start = np.array(position, dtype=float)[dims_displayed]
            if len(selection_indices) > 0 and center_by_data:
                center = self.data[
                    np.ix_(selection_indices, dims_displayed)
                ].mean(axis=0)
                self._drag_start -= center

    def _paste_data(self) -> None:
        """Paste any point from clipboard and select them."""
        npoints = len(self._view_data)
        totpoints = len(self.data)

        if len(self._clipboard.keys()) > 0:
            not_disp = self._slice_input.not_displayed
            data = deepcopy(self._clipboard['data'])
            offset = [
                self._data_slice[i] - self._clipboard['indices'][i]
                for i in not_disp
            ]
            data[:, not_disp] = data[:, not_disp] + np.array(offset)
            self._data = np.append(self.data, data, axis=0)
            self._shown = np.append(
                self.shown, deepcopy(self._clipboard['shown']), axis=0
            )
            self._size = np.append(
                self.size, deepcopy(self._clipboard['size']), axis=0
            )
            self._symbol = np.append(
                self.symbol, deepcopy(self._clipboard['symbol']), axis=0
            )

            self._feature_table.append(self._clipboard['features'])

            self.text._paste(**self._clipboard['text'])

            self._border_width = np.append(
                self.border_width,
                deepcopy(self._clipboard['border_width']),
                axis=0,
            )
            self._border._paste(
                colors=self._clipboard['border_color'],
                properties=_features_to_properties(
                    self._clipboard['features']
                ),
            )
            self._face._paste(
                colors=self._clipboard['face_color'],
                properties=_features_to_properties(
                    self._clipboard['features']
                ),
            )

            self._selected_view = list(
                range(npoints, npoints + len(self._clipboard['data']))
            )
            self._selected_data.update(
                set(range(totpoints, totpoints + len(self._clipboard['data'])))
            )
            self.refresh()

    def _copy_data(self) -> None:
        """Copy selected points to clipboard."""
        if len(self.selected_data) > 0:
            index = list(self.selected_data)
            self._clipboard = {
                'data': deepcopy(self.data[index]),
                'border_color': deepcopy(self.border_color[index]),
                'face_color': deepcopy(self.face_color[index]),
                'shown': deepcopy(self.shown[index]),
                'size': deepcopy(self.size[index]),
                'symbol': deepcopy(self.symbol[index]),
                'border_width': deepcopy(self.border_width[index]),
                'features': deepcopy(self.features.iloc[index]),
                'indices': self._data_slice,
                'text': self.text._copy(index),
            }
        else:
            self._clipboard = {}

    def to_mask(
        self,
        *,
        shape: tuple,
        data_to_world: Optional[Affine] = None,
        isotropic_output: bool = True,
    ) -> npt.NDArray:
        """Return a binary mask array of all the points as balls.

        Parameters
        ----------
        shape : tuple
            The shape of the mask to be generated.
        data_to_world : Optional[Affine]
            The data-to-world transform of the output mask image. This likely comes from a reference image.
            If None, then this is the same as this layer's data-to-world transform.
        isotropic_output : bool
            If True, then force the output mask to always contain isotropic balls in data/pixel coordinates.
            Otherwise, allow the anisotropy in the data-to-world transform to squash the balls in certain dimensions.
            By default this is True, but you should set it to False if you are going to create a napari image
            layer from the result with the same data-to-world transform and want the visualized balls to be
            roughly isotropic.

        Returns
        -------
        np.ndarray
            The output binary mask array of the given shape containing this layer's points as balls.
        """
        if data_to_world is None:
            data_to_world = self._data_to_world
        mask = np.zeros(shape, dtype=bool)
        mask_world_to_data = data_to_world.inverse
        points_data_to_mask_data = self._data_to_world.compose(
            mask_world_to_data
        )
        points_in_mask_data_coords = np.atleast_2d(
            points_data_to_mask_data(self.data)
        )

        # Calculating the radii of the output points in the mask is complex.
        radii = self.size / 2

        # Scale each radius by the geometric mean scale of the Points layer to
        # keep the balls isotropic when visualized in world coordinates.
        # The geometric means are used instead of the arithmetic mean
        # to maintain the volume scaling factor of the transforms.
        point_data_to_world_scale = gmean(np.abs(self._data_to_world.scale))
        mask_world_to_data_scale = (
            gmean(np.abs(mask_world_to_data.scale))
            if isotropic_output
            else np.abs(mask_world_to_data.scale)
        )
        radii_scale = point_data_to_world_scale * mask_world_to_data_scale

        output_data_radii = radii[:, np.newaxis] * np.atleast_2d(radii_scale)

        for coords, radii in zip(
            points_in_mask_data_coords, output_data_radii
        ):
            # Define a minimal set of coordinates where the mask could be present
            # by defining an inclusive lower and exclusive upper bound for each dimension.
            lower_coords = np.maximum(np.floor(coords - radii), 0).astype(int)
            upper_coords = np.minimum(
                np.ceil(coords + radii) + 1, shape
            ).astype(int)
            # Generate every possible coordinate within the bounds defined above
            # in a grid of size D1 x D2 x ... x Dd x D (e.g. for D=2, this might be 4x5x2).
            submask_coords = [
                range(lower_coords[i], upper_coords[i])
                for i in range(self.ndim)
            ]
            submask_grids = np.stack(
                np.meshgrid(*submask_coords, copy=False, indexing='ij'),
                axis=-1,
            )
            # Update the mask coordinates based on the normalized square distance
            # using a logical or to maintain any existing positive mask locations.
            normalized_square_distances = np.sum(
                ((submask_grids - coords) / radii) ** 2, axis=-1
            )
            mask[np.ix_(*submask_coords)] |= normalized_square_distances <= 1
        return mask

    def get_status(
        self,
        position: Optional[tuple] = None,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[list[int]] = None,
        world: bool = False,
    ) -> dict:
        """Status message information of the data at a coordinate position.

        # Parameters
        # ----------
        # position : tuple
        #     Position in either data or world coordinates.
        # view_direction : Optional[np.ndarray]
        #     A unit vector giving the direction of the ray in nD world coordinates.
        #     The default value is None.
        # dims_displayed : Optional[List[int]]
        #     A list of the dimensions currently being displayed in the viewer.
        #     The default value is None.
        # world : bool
        #     If True the position is taken to be in world coordinates
        #     and converted into data coordinates. False by default.

        # Returns
        # -------
        # source_info : dict
        #     Dict containing information that can be used in a status update.
        #"""
        if position is not None:
            value = self.get_value(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        else:
            value = None

        source_info = self._get_source_info()
        source_info['coordinates'] = generate_layer_coords_status(
            position[-self.ndim :], value
        )

        # if this points layer has properties
        properties = self._get_properties(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        if properties:
            source_info['coordinates'] += '; ' + ', '.join(properties)

        return source_info

    def _get_tooltip_text(
        self,
        position,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[list[int]] = None,
        world: bool = False,
    ) -> str:
        """
        tooltip message of the data at a coordinate position.

        Parameters
        ----------
        position : tuple
            Position in either data or world coordinates.
        view_direction : Optional[np.ndarray]
            A unit vector giving the direction of the ray in nD world coordinates.
            The default value is None.
        dims_displayed : Optional[List[int]]
            A list of the dimensions currently being displayed in the viewer.
            The default value is None.
        world : bool
            If True the position is taken to be in world coordinates
            and converted into data coordinates. False by default.

        Returns
        -------
        msg : string
            String containing a message that can be used as a tooltip.
        """
        return '\n'.join(
            self._get_properties(
                position,
                view_direction=view_direction,
                dims_displayed=dims_displayed,
                world=world,
            )
        )

    def _get_properties(
        self,
        position,
        *,
        view_direction: Optional[np.ndarray] = None,
        dims_displayed: Optional[list[int]] = None,
        world: bool = False,
    ) -> list:
        if self.features.shape[1] == 0:
            return []

        value = self.get_value(
            position,
            view_direction=view_direction,
            dims_displayed=dims_displayed,
            world=world,
        )
        # if the cursor is not outside the image or on the background
        if value is None or value > self.data.shape[0]:
            return []

        return [
            f'{k}: {v[value]}'
            for k, v in self.features.items()
            if k != 'index'
            and len(v) > value
            and v[value] is not None
            and not (isinstance(v[value], float) and np.isnan(v[value]))
        ]


Points._add_deprecated_properties()
