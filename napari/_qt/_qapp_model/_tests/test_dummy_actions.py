from app_model.backends.qt import QMenuItemAction
from app_model.types import Action

from napari._app_model.constants._menus import MenuId
from napari._app_model.utils import no_op, to_action_id, to_id_key


def assert_empty_keys_in_context(viewer):
    context = viewer.window._qt_viewer._layers.model().sourceModel()._root._ctx
    for menu_id in MenuId.contributables():
        context_key = f'{to_id_key(menu_id)}_empty'
        assert context_key in context


def assert_dummy_action_registered_with_menu(menu_id):
    from napari._app_model import get_app

    app = get_app()
    assert to_action_id(to_id_key(menu_id)) in app.commands


def test_menu_viewer_relaunch(make_napari_viewer):
    viewer = make_napari_viewer()
    assert_empty_keys_in_context(viewer)
    viewer.close()

    viewer2 = make_napari_viewer()
    # prior to #7106, this would fail
    assert_empty_keys_in_context(viewer2)
    viewer2.close()

    # prior to #7106, creating this viewer would error
    make_napari_viewer()


def test_dummy_action_disappears(make_napari_viewer, qtbot):
    from napari._app_model import get_app

    viewer = make_napari_viewer(show=True)
    empty_menu = 'napari/layers/segment'
    dummy_id = to_action_id(to_id_key(empty_menu))

    # menu contains dummy action and it's visible
    assert_dummy_action_registered_with_menu(empty_menu)
    dummy_action = viewer.window.layers_menu.findChild(
        QMenuItemAction, dummy_id
    )
    assert dummy_action.isVisible()

    my_action = Action(
        id='napari.new_action',
        title='New Action',
        callback=no_op,
        menus=[{'id': empty_menu}],
    )

    app = get_app()
    deregister_action = app.register_action(my_action)
    # registering new action doesn't deregister dummy
    assert_dummy_action_registered_with_menu(empty_menu)
    viewer.window._update_layers_menu_state()
    # but it does make it invisible
    qtbot.waitUntil(lambda: not dummy_action.isVisible())

    # deregistering the new action makes the dummy action visible again
    deregister_action()
    viewer.window._update_layers_menu_state()
    qtbot.waitUntil(dummy_action.isVisible)
