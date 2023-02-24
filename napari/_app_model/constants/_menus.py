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


class MenuId(str, Enum):
    """Id representing a menu somewhere in napari."""

    MENUBAR_VIEW = 'napari/view'
    VIEW_AXES = 'napari/view/axes'
    VIEW_SCALEBAR = 'napari/view/scalebar'

    MENUBAR_HELP = 'napari/help'

    MENUBAR_LAYERS = 'napari/layers'
    LAYERS_VISUALIZE = 'napari/layers/visualize'
    
    LAYERS_EDIT = 'napari/layers/edit'
    LAYERS_EDIT_ANNOTATE = 'napari/layers/edit/annotate'
    LAYERS_EDIT_FILTER = 'napari/layers/edit/filter'
    LAYERS_EDIT_TRANSFORM = 'napari/layers/edit/transform'

    LAYERS_MEASURE = 'napari/layers/measure'

    LAYERS_GENERATE = 'napari/layers/generate'
    LAYERS_GENERATE_REGISTRATION = 'napari/layers/generate/registration'
    LAYERS_GENERATE_PROJECTION = 'napari/layers/generate/projection'
    LAYERS_GENERATE_SEGMENTATION = 'napari/layers/generate/segmentation'
    LAYERS_GENERATE_TRACKS = 'napari/layers/generate/tracks'
    LAYERS_GENERATE_CLASSIFICATION = 'napari/layers/generate/classification'

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

    # TODO: FILE MENU!!!

    def __str__(self) -> str:
        return self.value

    @classmethod
    def contributables(cls) -> Set['MenuId']:
        """Set of all menu ids that can be contributed to by plugins."""

        # TODO: add these to docs, with a lookup for what each menu is/does.
        _contributables = {
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
            cls.LAYERS_GENERATE,
            cls.LAYERS_GENERATE_REGISTRATION,
            cls.LAYERS_GENERATE_PROJECTION,
            cls.LAYERS_GENERATE_SEGMENTATION,
            cls.LAYERS_GENERATE_TRACKS,
            cls.LAYERS_GENERATE_CLASSIFICATION
        }
        return _contributables

    @classmethod
    def sub_menus(cls) -> Sequence[Tuple['MenuId', SubmenuItem]]:
        """List of predefined submenu items to construct the default menu structure"""

        menu_id_to_sub_menus = {
            MenuId.LAYERLIST_CONTEXT: [
                {
                    'submenu': MenuId.LAYERS_CONVERT_DTYPE,
                    'title': trans._('Convert data type'),
                    'group': MenuGroup.LAYERLIST_CONTEXT.CONVERSION,
                    'order': None,
                },
                {
                    'submenu': MenuId.LAYERS_PROJECT,
                    'title': trans._('Projections'),
                    'group': MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
                    'order': None,
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
                },
                {
                    'submenu': MenuId.LAYERS_EDIT, 
                    'title': trans._('Edit')
                },
                {
                    'submenu': MenuId.LAYERS_MEASURE,
                    'title': trans._('Measure'),
                },
                {
                    'submenu': MenuId.LAYERS_GENERATE,
                    'title': trans._('Generate'),
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
            MenuId.LAYERS_GENERATE: [
                {
                    'submenu': MenuId.LAYERS_GENERATE_REGISTRATION, 
                    'title': trans._('Registration')
                },
                {
                    'submenu': MenuId.LAYERS_GENERATE_PROJECTION, 
                    'title': trans._('Projection')
                },
                {
                    'submenu': MenuId.LAYERS_GENERATE_SEGMENTATION, 
                    'title': trans._('Segmentation')
                },
                {
                    'submenu': MenuId.LAYERS_GENERATE_TRACKS, 
                    'title': trans._('Tracks')
                },
                {
                    'submenu': MenuId.LAYERS_GENERATE_CLASSIFICATION, 
                    'title': trans._('Classification')
                },
            ],
            MenuId.MENUBAR_ACQUISITION: [
                {
                    'submenu': MenuId.LAYERS_EDIT,
                    'title': trans._('Placeholder'),
                }
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

    class LAYERLIST_CONTEXT:
        CONVERSION = '1_conversion'
        SPLIT_MERGE = '5_split_merge'
        LINK = '9_link'


def is_menu_contributable(menu_id: str) -> bool:
    """Return True if the given menu_id is a menu that plugins can contribute to."""
    return (
        menu_id in MenuId.contributables()
        if menu_id.startswith("napari/")
        else True
    )
