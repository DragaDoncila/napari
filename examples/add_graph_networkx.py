"""
Add networkx graph
==================

Add a networkx graph directly to napari. This works as long as nodes
have a "pos" attribute with the node coordinate.

.. tags:: visualization-basic
"""

import networkx as nx
import numpy as np

import napari

hex_grid = nx.hexagonal_lattice_graph(5, 5, with_positions=True)
# make grid a bit bigger
nx.set_node_attributes(
    hex_grid,
    {
        k: (10 * v[0], 10 * v[1])
        for k, v in nx.get_node_attributes(hex_grid, 'pos').items()
    },
    'pos',
)
# below conversion not needed after napari/napari-graph#11 is released
hex_grid_ints = nx.convert_node_labels_to_integers(hex_grid)

viewer = napari.Viewer()
layer = viewer.add_graph(
    hex_grid_ints, size=1, face_color='pink', edges_visible=False
)
# layer.edges_visible = np.random.random(len(hex_grid_ints.edges)) > 0.5

if __name__ == "__main__":
    napari.run()
