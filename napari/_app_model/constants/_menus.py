"""All Menus that are available anywhere in the napari GUI are defined here.

These might be menubar menus, context menus, or other menus.  They could
even be "toolbars", such as the set of mode buttons on the layer list.
A "menu" needn't just be a literal QMenu (though it usually is): it is better
thought of as a set of related commands.

Internally, prefer using the `MenuId` enum instead of the string literal.

SOME of these (but definitely not all) will be exposed as "contributable"
menus for plugins to contribute commands and submenu items to.
"""

from enum import Enum
from typing import Sequence, Set, Tuple

from app_model.types import SubmenuItem

from ...utils.translations import trans
from napari.utils.compat import StrEnum
from napari._app_model.context import LayerListSelectionContextKeys as LLSCK


class MenuId(StrEnum):
    """Id representing a menu somewhere in napari."""

    MENUBAR_FILE = 'napari/file'
    FILE_OPEN_WITH_PLUGIN = 'napari/file/open_with_plugin'
    FILE_SAMPLES = 'napari/file/samples'
    FILE_IO_UTILITIES = 'napari/file/io_utilities'

    MENUBAR_VIEW = 'napari/view'
    VIEW_AXES = 'napari/view/axes'
    VIEW_SCALEBAR = 'napari/view/scalebar'

    MENUBAR_PLUGINS = 'napari/plugins'

    MENUBAR_HELP = 'napari/help'

    MENUBAR_LAYERS = 'napari/layers'
    LAYERS_VISUALIZE = 'napari/layers/visualize'
    
    LAYERS_EDIT = 'napari/layers/edit'
    LAYERS_EDIT_ANNOTATE = 'napari/layers/edit/annotate'
    LAYERS_EDIT_FILTER = 'napari/layers/edit/filter'
    LAYERS_EDIT_TRANSFORM = 'napari/layers/edit/transform'

    LAYERS_MEASURE = 'napari/layers/measure'

    LAYERS_REGISTRATION = 'napari/layers/registration'
    LAYERS_PROJECTION = 'napari/layers/projection'
    LAYERS_SEGMENTATION = 'napari/layers/segmentation'
    LAYERS_TRACKS = 'napari/layers/tracks'
    LAYERS_CLASSIFICATION = 'napari/layers/classification'

    # TOOLS_CLASSIFICATION = 'napari/tools/classification'
    # TOOLS_FILTERS = 'napari/tools/filters'
    # TOOLS_MEASUREMENT = 'napari/tools/measurement'
    # TOOLS_SEGMENTATION = 'napari/tools/segmentation'
    # TOOLS_PROJECTION = 'napari/tools/projection'
    # TOOLS_TRANSFORM = 'napari/tools/transform'
    # TOOLS_UTILITIES = 'napari/tools/utilities'
    # TOOLS_VISUALIZATION = 'napari/tools/visualization'

    MENUBAR_ACQUISITION = 'napari/acquisition'

    LAYERLIST_CONTEXT = 'napari/layers/context'
    LAYERS_CONVERT_DTYPE = 'napari/layers/convert_dtype'
    LAYERS_PROJECT = 'napari/layers/project'

    def __str__(self) -> str:
        return self.value

    @classmethod
    def contributables(cls) -> Set['MenuId']:
        """Set of all menu ids that can be contributed to by plugins."""

        # TODO: add these to docs, with a lookup for what each menu is/does.
        _contributables = {
            cls.FILE_IO_UTILITIES,
            cls.LAYERLIST_CONTEXT,
            cls.LAYERS_CONVERT_DTYPE,
            cls.LAYERS_PROJECT,
            cls.MENUBAR_ACQUISITION,
            cls.MENUBAR_LAYERS,
            cls.LAYERS_VISUALIZE,
            cls.LAYERS_EDIT,
            cls.LAYERS_EDIT_ANNOTATE,
            cls.LAYERS_EDIT_FILTER,
            cls.LAYERS_EDIT_TRANSFORM,
            cls.LAYERS_MEASURE,
            cls.LAYERS_REGISTRATION,
            cls.LAYERS_PROJECTION,
            cls.LAYERS_SEGMENTATION,
            cls.LAYERS_TRACKS,
            cls.LAYERS_CLASSIFICATION
        }
        return _contributables

    @classmethod
    def sub_menus(cls) -> Sequence[Tuple['MenuId', SubmenuItem]]:
        """List of predefined submenu items to construct the default menu structure"""

        menu_id_to_sub_menus = {
            MenuId.MENUBAR_FILE: [
                {
                    'submenu': MenuId.FILE_OPEN_WITH_PLUGIN,
                    'title': trans._('Open with Plugin'),
                    'group': MenuGroup.NAVIGATION,
                    'order': 99,
                },
                {
                    'submenu': MenuId.FILE_SAMPLES,
                    'title': trans._('Open Sample'),
                    'group': MenuGroup.NAVIGATION,
                    'order': 100,
                },
                {
                    'submenu': MenuId.FILE_IO_UTILITIES,
                    'title': trans._('IO Utilities'),
                    'group': MenuGroup.NAVIGATION,
                    'order': 101,
                }
            ],
            MenuId.LAYERLIST_CONTEXT: [
                {
                    'submenu': MenuId.LAYERS_CONVERT_DTYPE,
                    'title': trans._('Convert data type'),
                    'group': MenuGroup.LAYERLIST_CONTEXT.CONVERSION,
                    'order': None,
                    'enablement': LLSCK.all_selected_layers_labels,

                },
                {
                    'submenu': MenuId.LAYERS_PROJECT,
                    'title': trans._('Projections'),
                    'group': MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
                    'order': None,
                    'enablement': LLSCK.active_layer_is_image_3d,
                },
            ],
            MenuId.MENUBAR_VIEW: [
                {
                    'submenu': MenuId.VIEW_AXES,
                    'title': trans._('Axes'),
                },
                {
                    'submenu': MenuId.VIEW_SCALEBAR,
                    'title': trans._('Scale Bar'),
                },
            ],
            MenuId.MENUBAR_LAYERS: [
                {
                    'submenu': MenuId.LAYERS_VISUALIZE,
                    'title': trans._('Visualize'),
                    'group': MenuGroup.LAYERS.EXISTING,
                },
                {
                    'submenu': MenuId.LAYERS_EDIT, 
                    'title': trans._('Edit'),
                    'group': MenuGroup.LAYERS.EXISTING
                },
                {
                    'submenu': MenuId.LAYERS_MEASURE,
                    'title': trans._('Measure'),
                    'group': MenuGroup.LAYERS.EXISTING
                },
                {
                    'submenu': MenuId.LAYERS_REGISTRATION, 
                    'title': trans._('Registration'),
                    'group': MenuGroup.LAYERS.GENERATE
                },
                {
                    'submenu': MenuId.LAYERS_PROJECTION, 
                    'title': trans._('Projection'),
                    'group': MenuGroup.LAYERS.GENERATE
                },
                {
                    'submenu': MenuId.LAYERS_SEGMENTATION, 
                    'title': trans._('Segmentation'),
                    'group': MenuGroup.LAYERS.GENERATE
                },
                {
                    'submenu': MenuId.LAYERS_TRACKS, 
                    'title': trans._('Tracks'),
                    'group': MenuGroup.LAYERS.GENERATE
                },
                {
                    'submenu': MenuId.LAYERS_CLASSIFICATION, 
                    'title': trans._('Classification'),
                    'group': MenuGroup.LAYERS.GENERATE
                },
            ],
            MenuId.LAYERS_EDIT: [
                {
                    'submenu': MenuId.LAYERS_EDIT_ANNOTATE, 
                    'title': trans._('Annotate')
                },
                {
                    'submenu': MenuId.LAYERS_EDIT_FILTER, 
                    'title': trans._('Filter')
                },
                {
                    'submenu': MenuId.LAYERS_EDIT_TRANSFORM, 
                    'title': trans._('Transform')
                },
            ],
        }

        return [
            (menu_id, SubmenuItem(**submenu))
            for menu_id, submenus in menu_id_to_sub_menus.items()
            for submenu in submenus
        ]


# XXX: the structure/usage pattern of this class may change in the future
class MenuGroup:
    NAVIGATION = 'navigation'  # always the first group in any menu
    RENDER = '1_render'
    PLUGINS = '1_plugins'
    PLUGIN_CONTRIBUTIONS = '2_plugin_contributions'
    # File menubar
    PREFERENCES = '2_preferences'
    SAVE = '3_save'
    CLOSE = '4_close'

    class LAYERLIST_CONTEXT:
        CONVERSION = '1_conversion'
        SPLIT_MERGE = '5_split_merge'
        LINK = '9_link'
        
    class LAYERS:
        NEW = '1_new'
        EXISTING = '2_existing'
        GENERATE = '3_generate'
        PLUGINS = '4_plugins'


def is_menu_contributable(menu_id: str) -> bool:
    """Return True if the given menu_id is a menu that plugins can contribute to."""
    return (
        menu_id in MenuId.contributables()
        if menu_id.startswith("napari/")
        else True
    )
