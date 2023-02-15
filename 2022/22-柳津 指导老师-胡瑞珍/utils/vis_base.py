
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np
import seaborn as sns
import random


class vtkTimerCallback():
    def __init__(self):
        self.timer_count = 0

    def execute(self, obj, event):
        # print(self.timer_count)
        self.actor.RotateY(self.timer_count % 360)
        # self.actor.SetPosition(self.timer_count, self.timer_count, 0)
        iren = obj
        iren.GetRenderWindow().Render()
        self.timer_count += 1
class VIS_BASE(object):
    def __init__(self):
        # self._cam_K = np.array([[1600, 0, 800], [0, 1600, 600], [0, 0, 1]])
        # self._cam_K = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1]])
        self._cam_K = np.array([[600, 0, 300], [0, 600, 200], [0, 0, 1]])

    @property
    def cam_K(self):
        return self._cam_K

    def set_axes_actor(self):
        '''
        Set camera coordinate system
        '''
        transform = vtk.vtkTransform()
        transform.Translate(0., 0., 0.)
        # self defined
        axes = vtk.vtkAxesActor()
        axes.SetUserTransform(transform)
        axes.SetTotalLength(1, 1, 1)

        axes.SetTipTypeToCone()
        axes.SetConeRadius(50e-2)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(40e-3)

        vtk_textproperty = vtk.vtkTextProperty()
        vtk_textproperty.SetFontSize(1)
        vtk_textproperty.SetBold(True)
        vtk_textproperty.SetItalic(False)
        vtk_textproperty.SetShadow(True)

        for label in [axes.GetXAxisCaptionActor2D(), axes.GetYAxisCaptionActor2D(), axes.GetZAxisCaptionActor2D()]:
            label.SetCaptionTextProperty(vtk_textproperty)

        return axes

    def set_camera(self, position, focal_point, up_vec, cam_K):
        camera = vtk.vtkCamera()
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal_point)
        camera.SetViewUp(*up_vec)
        camera.SetViewAngle((2*np.arctan(cam_K[1][2]/cam_K[0][0]))/np.pi*180)
        #camera.Elevation(90)
        #camera.SetViewAngle(180)
        return camera

    def set_actor(self, mapper):
        '''
        vtk general actor
        :param mapper: vtk shape mapper
        :return: vtk actor
        '''
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def set_mapper(self, prop, mode):

        mapper = vtk.vtkPolyDataMapper()

        if mode == 'model':
            mapper.SetInputConnection(prop.GetOutputPort())

        elif mode == 'box':
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInput(prop)
            else:
                mapper.SetInputData(prop)

        else:
            raise IOError('No Mapper mode found.')

        return mapper

    def set_points_property(self, point_clouds, point_colors):
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        x3 = point_clouds[:, 0]
        y3 = point_clouds[:, 1]
        z3 = point_clouds[:, 2]

        for x, y, z, c in zip(x3, y3, z3, point_colors):
            id = points.InsertNextPoint([x, y, z])
            colors.InsertNextTuple3(*c)
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(id)

        # Create a polydata object
        point = vtk.vtkPolyData()
        # Set the points and vertices we created as the geometry and topology of the polydata
        point.SetPoints(points)
        point.SetVerts(vertices)
        point.GetPointData().SetScalars(colors)
        point.GetPointData().SetActiveScalars('Color')

        return point

    def set_arrow_actor(self, startpoint, vector, tip_len_ratio=0.2, tip_r_ratio=0.08, shaft_r_ratio=0.02,
                        mode='vector'):
        '''
        Design an actor to draw an arrow from startpoint to startpoint + vector.
        :param startpoint: 3D point
        :param vector: 3D vector
        :return: a vtk arrow actor
        '''
        if mode == 'vector':
            vector = vector / np.linalg.norm(vector) * 0.5
            endpoint = startpoint + vector
        elif mode == 'endpoint':
            endpoint = vector
        else:
            raise NotImplementedError('Cannot recognize mode.')

        arrow_source = vtk.vtkArrowSource()

        # compute a basis
        normalisedX = [0 for i in range(3)]
        normalisedY = [0 for i in range(3)]
        normalisedZ = [0 for i in range(3)]

        # the X axis is a vector from start to end
        math = vtk.vtkMath()
        math.Subtract(endpoint, startpoint, normalisedX)
        length = math.Norm(normalisedX)
        math.Normalize(normalisedX)

        # adjust arrow length
        arrow_source.SetTipLength(tip_len_ratio/length)
        arrow_source.SetTipRadius(tip_r_ratio/length)
        arrow_source.SetShaftRadius(shaft_r_ratio/length)

        # the Z axis is an arbitrary vector cross X
        arbitrary = [0 for i in range(3)]
        arbitrary[0] = random.uniform(-10, 10)
        arbitrary[1] = random.uniform(-10, 10)
        arbitrary[2] = random.uniform(-10, 10)
        math.Cross(normalisedX, arbitrary, normalisedZ)
        math.Normalize(normalisedZ)

        # the Y axis is Z cross X
        math.Cross(normalisedZ, normalisedX, normalisedY)

        # create the direction cosine matrix
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalisedX[i])
            matrix.SetElement(i, 1, normalisedY[i])
            matrix.SetElement(i, 2, normalisedZ[i])

        # apply the transform
        transform = vtk.vtkTransform()
        transform.Translate(startpoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # create a mapper and an actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()

        mapper.SetInputConnection(arrow_source.GetOutputPort())
        actor.SetUserMatrix(transform.GetMatrix())
        actor.SetMapper(mapper)

        return actor

    def set_bbox_line_prop(self, corners, faces, color):
        edge_set1 = np.vstack([np.array(faces)[:, 0], np.array(faces)[:, 1]]).T
        edge_set2 = np.vstack([np.array(faces)[:, 1], np.array(faces)[:, 2]]).T
        edge_set3 = np.vstack([np.array(faces)[:, 2], np.array(faces)[:, 3]]).T
        edge_set4 = np.vstack([np.array(faces)[:, 3], np.array(faces)[:, 0]]).T
        edges = np.vstack([edge_set1, edge_set2, edge_set3, edge_set4])
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        pts = vtk.vtkPoints()
        for corner in corners:
            pts.InsertNextPoint(corner)

        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        for edge in edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)
            colors.InsertNextTuple3(*color)

        linesPolyData = vtk.vtkPolyData()
        linesPolyData.SetPoints(pts)
        linesPolyData.SetLines(lines)
        linesPolyData.GetCellData().SetScalars(colors)

        return linesPolyData

    def get_bbox_line_actor(self, center, vectors, color, opacity=1., width=10):
        corners, faces = self.get_box_corners(center, vectors)
        bbox_actor = self.set_actor(self.set_mapper(self.set_bbox_line_prop(corners, faces, color), 'box'))
        bbox_actor.GetProperty().SetOpacity(opacity)
        bbox_actor.GetProperty().SetLineWidth(width)
        return bbox_actor

    def get_bbox_cube_actor(self, center, vectors, color, opacity=1.):
        corners, faces = self.get_box_corners(center, vectors)
        box_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
        box_actor.GetProperty().SetOpacity(opacity)
        return box_actor

    def set_sphere_property(self, joint, radius):
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(*joint)
        sphere_source.SetRadius(radius)
        sphere_source.SetPhiResolution(10)
        sphere_source.SetThetaResolution(10)
        return sphere_source

    def set_line_property(self, p0, p1):
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoint1(p0)
        lineSource.SetPoint2(p1)
        return lineSource

    def set_ply_property(self, plyfile):

        plydata = vtk.vtkPLYReader()
        plydata.SetFileName(plyfile)
        plydata.Update()
        return plydata

    def get_box_corners(self, center, vectors):
        '''
        Convert box center and vectors to the corner-form
        :param center:
        :param vectors:
        :return: corner points and faces related to the box
        '''
        corner_pnts = [None] * 8
        corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
        corner_pnts[1] = tuple(center + vectors[0] - vectors[1] - vectors[2])
        corner_pnts[2] = tuple(center + vectors[0] + vectors[1] - vectors[2])
        corner_pnts[3] = tuple(center - vectors[0] + vectors[1] - vectors[2])

        corner_pnts[4] = tuple(center - vectors[0] - vectors[1] + vectors[2])
        corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
        corner_pnts[6] = tuple(center + vectors[0] + vectors[1] + vectors[2])
        corner_pnts[7] = tuple(center - vectors[0] + vectors[1] + vectors[2])

        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (0, 4, 7, 3)]

        return corner_pnts, faces

    def mkVtkIdList(self, it):
        vil = vtk.vtkIdList()
        for i in it:
            vil.InsertNextId(int(i))
        return vil

    def set_cube_prop(self, corners, faces, color):

        cube = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        polys = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName('Color')

        color = np.uint8(color)

        for i in range(8):
            points.InsertPoint(i, corners[i])

        for i in range(6):
            polys.InsertNextCell(self.mkVtkIdList(faces[i]))

        for i in range(8):
            colors.InsertNextTuple3(*color)

        # Assign the pieces to the vtkPolyData
        cube.SetPoints(points)
        del points
        cube.SetPolys(polys)
        del polys
        cube.GetPointData().SetScalars(colors)
        cube.GetPointData().SetActiveScalars('Color')
        del colors

        return cube

    def get_voxel_actor(self, tsdf_volume, voxel_centroids, voxel_vectors):

        voxel_actors = []
        colors = get_colors(tsdf_volume)

        for tsdf_value, centroid, color in zip(tsdf_volume, voxel_centroids, colors):
            if tsdf_value == 1:
                continue
            corners, faces = self.get_box_corners(centroid, voxel_vectors)
            voxel_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
            voxel_actor.GetProperty().SetOpacity(1)
            voxel_actors.append(voxel_actor)

        return voxel_actors

    def set_render(self, cam_loc=(0.1, 0.1, 0.1), focal_pnt=(0., 0., 0.), up_vec=None, *args, **kwargs):
        renderer = vtk.vtkRenderer()
        renderer.ResetCamera()

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        '''set camera'''
        up_vec = [-cam_loc[0], -cam_loc[1], cam_loc[0] ** 2 / cam_loc[2] + cam_loc[1] ** 2 / cam_loc[2]] if up_vec is None else up_vec
        camera = self.set_camera(cam_loc, focal_pnt, up_vec, self.cam_K)
        renderer.SetActiveCamera(camera)

        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, -10, 10), (-10, -10, 10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(1.5)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer

    def set_render_window(self, offline, *args, **kwargs):
        render_window = vtk.vtkRenderWindow()
        renderer = self.set_render(*args, **kwargs)
        # renderer.SetUseDepthPeeling(1)
        render_window.AddRenderer(renderer)
        render_window.SetSize(*np.int32((self.cam_K[:2,2]*2)))
        render_window.SetOffScreenRendering(offline)

        return render_window

    def camRT2vtk_cam(self, camRT):
        cam_position = camRT[:3, -1]
        focal_point = np.array([0., 0., 1.]).dot(camRT[:3, :3].T) + camRT[:3, -1]
        up = np.array([0., -1., 0.]).dot(camRT[:3, :3].T)
        return cam_position, focal_point, up


    def visualize(self, save_path = None, offline=False, *args, **kwargs):
        '''
               Visualize a 3D scene.
        '''
        render_window = self.set_render_window(offline, *args, **kwargs)
        render_window.Render()

        if save_path is not None:
            windowToImageFilter = vtk.vtkWindowToImageFilter()
            windowToImageFilter.SetInput(render_window)
            windowToImageFilter.Update()

            writer = vtk.vtkJPEGWriter()
            writer.SetFileName(save_path)
            writer.SetInputConnection(windowToImageFilter.GetOutputPort())
            writer.Write()

        #if not offline:
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)
        render_window_interactor.Start()
        return render_window_interactor
    # '''
    # Visualize a 3D scene.
    # '''
    # render_window = self.set_render_window(offline, *args, **kwargs)
    #
    # if save_path is not None:
    #     windowToImageFilter = vtk.vtkWindowToImageFilter()
    #     windowToImageFilter.SetInput(render_window)
    #     windowToImageFilter.Update()
    #
    #     writer = vtk.vtkJPEGWriter()
    #     writer.SetFileName(save_path)
    #     writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    #     writer.Write()
    #
    # if not offline:
    #
    #     render_window_interactor = vtk.vtkRenderWindowInteractor()
    #     render_window_interactor.SetRenderWindow(render_window)
    #     render_window.Render()
    #     # Sign up to receive TimerEvent
    #     cubeSource = vtk.vtkCubeSource()
    #     cubeSource.SetCenter(2.0, 2.0, 2.0)
    #     mapper = vtk.vtkPolyDataMapper()
    #     mapper.SetInputConnection(cubeSource.GetOutputPort())
    #     actor = vtk.vtkActor()
    #     actor.SetMapper(mapper)
    #     actor.GetProperty().SetColor(0, 0, 0)
    #     renderer = self.set_render(*args, **kwargs)
    #     renderer.AddActor(actor)
    #
    #
    #     cb = vtkTimerCallback()
    #     cb.actor = actor
    #     render_window_interactor.AddObserver('TimerEvent', cb.execute)
    #     render_window_interactor.CreateRepeatingTimer(100)
    #     render_window_interactor.Start()


def get_colors(values, palatte_name = 'RdBu', color_depth = 256):
    '''
    Return color values given scalars.
    :param values: N values
    :param palatte_name: check seaborn
    :param color_depth:
    :return: Nx3 colors
    '''
    palatte = np.array(sns.color_palette(palatte_name, color_depth))
    scaled_values = np.int32((values-min(values))/(max(values) - min(values)) * (color_depth-1))
    return palatte[scaled_values]

if __name__ == '__main__':
    viser = VIS_BASE()
    viser.visualize(offline=False)