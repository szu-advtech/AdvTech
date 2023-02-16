import pyvista as pv
from matplotlib.cm import get_cmap
import numpy as np

def bbox(mesh:pv.PolyData):
    min_x, max_x, min_y, max_y, min_z, max_z = mesh.bounds
    w = max_x - min_x
    l = max_y - min_y
    h = max_z - min_z

    size = np.array([w,l,h])
    
    return size

def get_pc(points):
    pc = pv.PolyData(points)
    pc['pos'] = points
    return pc

def get_axis(mat=np.eye(4)):
    axes = get_pc(np.zeros((3,3)) + mat[:3,3])
    axes['norm'] = mat[:3,:3]
    axes_arrows = axes.glyph(
        orient='norm',
        scale=False,
        factor=0.08,
    )
    return axes_arrows

def get_color(value):
    
    cmap = get_cmap("nipy_spectral")
    colors = ( np.array(cmap(value))[:3] * 255.0).astype(np.uint8)
    return colors

def show_mesh(meshes, colors=None):
    plotter = pv.Plotter()
    plotter.add_axes()

    main_axes = get_axis()
    for i, m in enumerate(meshes):
        if colors is not None:
            c = colors[i]
        else:
            c = get_color( np.random.rand() )
            
        # plotter.add_mesh(m, scalars='pos')
        plotter.add_mesh(m, color=c)
    plotter.add_mesh(main_axes, color='red')
    plotter.show()
