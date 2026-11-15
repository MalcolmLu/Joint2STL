from __future__ import annotations

import math
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget

from .model import Session


def orientation_letter(vector):
    i = int(np.argmax(np.abs(vector)))
    return ('L', 'P', 'S')[i] if vector[i] > 0 else ('R', 'A', 'I')[i]


def plane_display_title(volume,plane):
    fixed={'axial':0,'coronal':1,'sagittal':2}[plane]
    normal=volume.affine_lps[:3,2-fixed]
    direction=np.abs(normal)/np.linalg.norm(normal)
    name=('矢状位','冠状位','轴位')[int(direction.argmax())]
    if direction.max()<.999:name='斜'+name
    return name+(' · 采集平面' if fixed==0 else ' · 重排')


class SliceView(QWidget):
    interaction_started = Signal()
    navigate = Signal(object)
    roi_changed = Signal(object)
    stroke_start = Signal(str, int)
    stroke_point = Signal(str, object)
    stroke_end = Signal()
    point_action = Signal(str, object, str)
    slice_scrolled = Signal(str, int)
    model_box = Signal(str,object,object)

    def __init__(self, plane):
        super().__init__()
        self.plane = plane
        self.session: Session | None = None
        self.cross = [0, 0, 0]
        self.tool = 'browse'
        self.window = 1600
        self.level = 450
        self.opacity = 0.4
        self.active = 0
        self.brush_mm = 3
        self.zoom = 1.0
        self.pan = QPointF()
        self._drag = None
        self._start = None
        self._last_mouse = None
        self._last_voxel = None
        self._cursor = None
        self._cache_key = None
        self._cache = None
        self.setMinimumSize(200, 160)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def fixed_axis(self):
        return {'axial': 0, 'coronal': 1, 'sagittal': 2}[self.plane]

    @property
    def flip_z(self):
        return self.session is not None and self.session.volume.affine_lps[2, 2] > 0

    def invalidate(self):
        self._cache_key = None
        self.update()

    def extract(self, data):
        z, y, x = self.cross
        if self.plane == 'axial':
            return data[z]
        a = data[:, y, :] if self.plane == 'coronal' else data[:, :, x]
        return a[::-1] if self.flip_z else a

    def dimensions(self):
        if self.session is None:
            return 1, 1, 1, 1
        shape = self.session.volume.hu.shape
        sz, sy, sx = self.session.volume.spacing_zyx
        if self.plane == 'axial':
            return shape[2], shape[1], sx, sy
        if self.plane == 'coronal':
            return shape[2], shape[0], sx, sz
        return shape[1], shape[0], sy, sz

    def image_rect(self):
        nx, ny, sx, sy = self.dimensions()
        available = self.rect().adjusted(30, 24, -30, -24)
        scale = min(available.width() / (nx * sx), available.height() / (ny * sy)) * self.zoom
        width, height = nx * sx * scale, ny * sy * scale
        return QRectF(self.width()/2 - width/2 + self.pan.x(), self.height()/2 - height/2 + self.pan.y(), width, height)

    def to_pixel(self, voxel):
        z, y, x = voxel
        z = self.session.volume.hu.shape[0] - 1 - z if self.flip_z else z
        return (x, y) if self.plane == 'axial' else ((x, z) if self.plane == 'coronal' else (y, z))

    def from_position(self, point):
        if self.session is None:
            return None
        rect = self.image_rect()
        if not rect.contains(point):
            return None
        nx, ny, *_ = self.dimensions()
        u = min(nx - 1, max(0, int((point.x() - rect.left()) / rect.width() * nx)))
        v = min(ny - 1, max(0, int((point.y() - rect.top()) / rect.height() * ny)))
        z, y, x = self.cross
        if self.plane == 'axial':
            x, y = u, v
        else:
            z = self.session.volume.hu.shape[0] - 1 - v if self.flip_z else v
            if self.plane == 'coronal': x = u
            else: y = u
        return z, y, x

    def screen_point(self, voxel):
        u, v = self.to_pixel(voxel)
        nx, ny, *_ = self.dimensions(); rect = self.image_rect()
        return QPointF(rect.left() + (u + 0.5) / nx * rect.width(), rect.top() + (v + 0.5) / ny * rect.height())

    def _image(self):
        s = self.session
        key = (id(s), s.revision, tuple(self.cross), self.window, self.level, self.opacity,
               tuple((t.id, t.visible, t.color) for t in s.segments))
        if key == self._cache_key:
            return self._cache
        data = self.extract(s.volume.hu).astype(np.float32)
        gray = np.clip((data - (self.level - self.window/2)) * (255 / max(1, self.window)), 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        lab = self.extract(s.labels)
        maximum = max((t.id for t in s.segments), default=0)
        if maximum:
            colors = np.zeros((maximum + 1, 3), dtype=np.uint8)
            visible = np.zeros(maximum + 1, dtype=bool)
            for t in s.segments:
                color = QColor(t.color)
                colors[t.id] = [color.red(), color.green(), color.blue()]
                visible[t.id] = t.visible
            draw = visible[lab]
            rgb[draw] = (rgb[draw] * (1-self.opacity) + colors[lab[draw]] * self.opacity).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        self._cache_key, self._cache = key, image
        return image

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#0a101b'))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.session is None:
            painter.setPen(QColor('#718198'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '导入 CT / MRI 后显示切片')
            return
        rect = self.image_rect()
        painter.drawImage(rect, self._image())
        s = self.session
        roi = s.roi
        lo = [b.start for b in roi]; hi = [b.stop - 1 for b in roi]
        a, b = self.screen_point(lo), self.screen_point(hi)
        r = QRectF(a, b).normalized()
        painter.setPen(QPen(QColor('#f1c46e'), 1.3, Qt.PenStyle.DashLine))
        painter.drawRect(r)
        if not roi[self.fixed_axis].start <= self.cross[self.fixed_axis] < roi[self.fixed_axis].stop:
            painter.fillRect(rect, QColor(10, 16, 27, 115))
            painter.setPen(QColor('#f1c46e'))
            painter.drawText(12, 20, '当前层在分割范围外')
        cross = self.screen_point(self.cross)
        painter.setPen(QPen(QColor(98, 202, 215, 130), 0.8))
        painter.drawLine(QPointF(rect.left(), cross.y()), QPointF(rect.right(), cross.y()))
        painter.drawLine(QPointF(cross.x(), rect.top()), QPointF(cross.x(), rect.bottom()))
        # Orientation follows the actual DICOM direction, including reversed row/column axes.
        affine = s.volume.affine_lps[:3, :3]
        horizontal = affine[:, 0] if self.plane != 'sagittal' else affine[:, 1]
        vertical = affine[:, 1] if self.plane == 'axial' else affine[:, 2] * (-1 if self.flip_z else 1)
        painter.setPen(QColor('#90a2b9'))
        painter.drawText(8, self.height()//2, orientation_letter(-horizontal))
        painter.drawText(self.width()-22, self.height()//2, orientation_letter(horizontal))
        painter.drawText(self.width()//2, 17, orientation_letter(-vertical))
        painter.drawText(self.width()//2, self.height()-7, orientation_letter(vertical))
        for seed in s.seeds:
            if seed.zyx[self.fixed_axis] != self.cross[self.fixed_axis]:
                continue
            seg = s.segment(seed.label)
            if seg:
                p = self.screen_point(seed.zyx)
                painter.setPen(QPen(QColor(seg.color), 2.5))
                painter.drawEllipse(p, 5, 5)
                painter.drawLine(p + QPointF(-9,0), p+QPointF(9,0))
                painter.drawLine(p+QPointF(0,-9),p+QPointF(0,9))
        if s.volume.is_mr and self.fixed_axis==0:
            from .mr_correction import stored_points
            for prompt in stored_points(s,self.active,self.cross[0]):
                p=self.screen_point(prompt['zyx']);positive=prompt['positive']
                painter.setPen(QPen(QColor('#70edac' if positive else '#ff8c83'),2))
                painter.drawEllipse(p,6,6);painter.drawLine(p+QPointF(-4,0),p+QPointF(4,0))
                if positive:painter.drawLine(p+QPointF(0,-4),p+QPointF(0,4))
        if self._drag in ('roi','model_box') and self._start is not None and self._cursor is not None:
            painter.setPen(QPen(QColor('#66ddd1' if self._drag=='model_box' else '#f1c46e'), 2))
            painter.fillRect(QRectF(self._start, self._cursor).normalized(), QColor(100,220,210,28) if self._drag=='model_box' else QColor(241,196,110,28))
            painter.drawRect(QRectF(self._start, self._cursor).normalized())
        if self._cursor and self.tool in ('paint','erase') and rect.contains(self._cursor):
            nx, ny, sx, sy = self.dimensions()
            rx = self.brush_mm / (nx*sx) * rect.width()
            ry = self.brush_mm / (ny*sy) * rect.height()
            painter.setPen(QPen(QColor('#ffafa4' if self.tool=='erase' else '#ffffff'),1.4))
            painter.drawEllipse(self._cursor,rx,ry)

    def mousePressEvent(self, event):
        if self.session is None: return
        # Cocoa accessibility can update a checkable toolbar button before its
        # Qt notification reaches the view. Resolve the selected tool before
        # deciding whether this drag paints or moves the crosshair.
        self.interaction_started.emit()
        self.setFocus()
        point = event.position(); voxel = self.from_position(point)
        self._last_mouse, self._cursor = point, point
        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.RightButton:
            self._drag = 'pan'; return
        if event.button() != Qt.MouseButton.LeftButton or voxel is None: return
        if self.tool == 'roi':
            self._drag = 'roi'; self._start = point; self._start_voxel = voxel
        elif self.tool=='model_box':
            self._drag='model_box';self._start=point;self._start_voxel=voxel
        elif self.tool in ('paint','erase'):
            self._drag = 'stroke'; self._last_voxel = voxel
            self.stroke_start.emit(self.plane, self.cross[self.fixed_axis])
            self.stroke_point.emit(self.plane,voxel)
        elif self.tool == 'browse':
            self._drag = 'navigate'; self.navigate.emit(voxel)
        else:
            self.point_action.emit(self.tool,voxel,self.plane)
        self.update()

    def mouseMoveEvent(self,event):
        point=event.position(); self._cursor=point
        if self._drag=='pan':
            self.pan += point-self._last_mouse
        voxel=self.from_position(point)
        if voxel is not None:
            if self._drag=='stroke':
                last=np.asarray(self._last_voxel); new=np.asarray(voxel)
                distance=np.linalg.norm((new-last)*self.session.volume.spacing_zyx)
                steps=max(1,int(math.ceil(distance/max(self.brush_mm/2,0.3))))
                for t in np.linspace(0,1,steps+1)[1:]:
                    self.stroke_point.emit(self.plane,tuple(np.rint(last+(new-last)*t).astype(int)))
                self._last_voxel=voxel
            elif self._drag=='navigate': self.navigate.emit(voxel)
        self._last_mouse=point
        self.update()

    def mouseReleaseEvent(self,event):
        if self._drag=='stroke': self.stroke_end.emit()
        elif self._drag=='roi':
            end=self.from_position(event.position())
            if end is not None:
                box=list(self.session.roi)
                for axis in range(3):
                    if axis!=self.fixed_axis:
                        a,b=self._start_voxel[axis],end[axis]
                        box[axis]=slice(min(a,b),max(a,b)+1)
                self.roi_changed.emit(tuple(box))
        elif self._drag=='model_box':
            end=self.from_position(event.position())
            if end is not None:self.model_box.emit(self.plane,self._start_voxel,end)
        self._drag=None; self._start=None; self.update()

    def leaveEvent(self,event):
        self._cursor=None; self.update()

    def wheelEvent(self,event):
        if self._drag == 'stroke':
            # A stroke and its undo patch belong to one fixed slice.
            event.accept()
            return
        amount=1 if event.angleDelta().y()>0 else -1
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom=min(12,max(0.2,self.zoom*(1.15**amount)));self.update()
        else: self.slice_scrolled.emit(self.plane,amount)
        event.accept()


class SlicePanel(QFrame):
    def __init__(self,plane,title,color):
        super().__init__();self.setObjectName('viewPanel')
        layout=QVBoxLayout(self);layout.setContentsMargins(1,1,1,5);layout.setSpacing(3)
        header=QHBoxLayout();header.setContentsMargins(12,5,8,0)
        self.title=QLabel(title);self.title.setStyleSheet(f'color: {color}; font-weight: 600;')
        self.counter=QLabel('—');self.counter.setObjectName('muted')
        fit=QPushButton('适配');fit.setObjectName('compact');fit.clicked.connect(self.fit)
        header.addWidget(self.title);header.addStretch();header.addWidget(self.counter);header.addWidget(fit)
        layout.addLayout(header)
        self.view=SliceView(plane);layout.addWidget(self.view,1)
        self.slider=QSlider(Qt.Orientation.Horizontal);self.slider.setEnabled(False)
        self.slider.setObjectName(f'{plane}Slider');layout.addWidget(self.slider)

    def fit(self):
        self.view.zoom=1;self.view.pan=QPointF();self.view.update()


class SurfacePanel(QFrame):
    refresh_requested=Signal()

    def __init__(self):
        super().__init__();self.setObjectName('viewPanel')
        import vtkmodules.qt
        vtkmodules.qt.PyQtImpl='PySide6'
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
        from vtkmodules.vtkRenderingCore import vtkRenderer
        from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
        import vtkmodules.vtkRenderingOpenGL2  # register rendering backend
        layout=QVBoxLayout(self);layout.setContentsMargins(1,1,1,1);layout.setSpacing(0)
        header=QHBoxLayout();header.setContentsMargins(12,5,8,5)
        self.caption=QLabel('三维骨模型');self.caption.setStyleSheet('color:#b7a5e3;font-weight:600')
        self.state=QLabel('LPS · mm');self.state.setObjectName('muted')
        self.refresh=QPushButton('更新 3D');self.refresh.setObjectName('compact');self.refresh.clicked.connect(self.refresh_requested.emit)
        self.front=QPushButton('前视');self.front.setObjectName('compact');self.front.clicked.connect(lambda:self.camera('front'))
        self.side=QPushButton('侧视');self.side.setObjectName('compact');self.side.clicked.connect(lambda:self.camera('side'))
        header.addWidget(self.caption);header.addStretch();header.addWidget(self.state);header.addWidget(self.front);header.addWidget(self.side);header.addWidget(self.refresh)
        layout.addLayout(header)
        self.vtk=QVTKRenderWindowInteractor(self);layout.addWidget(self.vtk,1)
        self.renderer=vtkRenderer();self.renderer.SetBackground(0.04,0.065,0.11)
        self.renderer.SetBackground2(0.085,0.13,0.19);self.renderer.GradientBackgroundOn()
        self.vtk.GetRenderWindow().AddRenderer(self.renderer)
        self.vtk.GetRenderWindow().SetMultiSamples(4)
        self.vtk.GetRenderWindow().GetInteractor().SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        self.vtk.Initialize()
        self.actors={};self.bounds=None

    def set_meshes(self,meshes,reset=True):
        from vtkmodules.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkPolyData,vtkCellArray
        from vtkmodules.vtkRenderingCore import vtkActor,vtkPolyDataMapper
        from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
        self.renderer.RemoveAllViewProps();self.actors={}
        self.bounds=None
        bounds=[]
        for segment,mesh,_ in meshes:
            points=vtkPoints();points.SetData(numpy_to_vtk(np.asarray(mesh.vertices,dtype=np.float32),deep=True))
            cells=vtkCellArray()
            indexed=np.column_stack((np.full(len(mesh.faces),3,dtype=np.int64),mesh.faces.astype(np.int64))).ravel()
            cells.SetCells(len(mesh.faces),numpy_to_vtkIdTypeArray(indexed,deep=True))
            poly=vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
            normals=vtkPolyDataNormals();normals.SetInputData(poly);normals.SplittingOff();normals.ConsistencyOn();normals.Update()
            mapper=vtkPolyDataMapper();mapper.SetInputConnection(normals.GetOutputPort());mapper.ScalarVisibilityOff()
            actor=vtkActor();actor.SetMapper(mapper)
            c=QColor(segment.color);actor.GetProperty().SetColor(c.redF(),c.greenF(),c.blueF())
            actor.GetProperty().SetAmbient(0.18);actor.GetProperty().SetDiffuse(0.75);actor.GetProperty().SetSpecular(0.22);actor.GetProperty().SetSpecularPower(25)
            self.renderer.AddActor(actor);self.actors[segment.id]=actor;bounds.append(mesh.bounds)
        if bounds:
            self.bounds=np.array([np.min(np.array(bounds)[:,0],axis=0),np.max(np.array(bounds)[:,1],axis=0)])
            if reset:self.camera('front')
        self.state.setText('LPS · mm · 原始分辨率')
        self.vtk.GetRenderWindow().Render()

    def camera(self,mode):
        if self.bounds is None:return
        center=self.bounds.mean(axis=0);d=max(np.ptp(self.bounds,axis=0))*3
        c=self.renderer.GetActiveCamera();c.SetFocalPoint(*center)
        c.SetPosition(*(center+np.array([0,-d,0] if mode=='front' else [-d,0,0])))
        c.SetViewUp(0,0,1);c.ParallelProjectionOn();self.renderer.ResetCamera();self.renderer.ResetCameraClippingRange()
        self.vtk.GetRenderWindow().Render()

    def visibility(self,segments):
        visible = {s.id: s.visible for s in segments}
        for ident, actor in self.actors.items():
            actor.SetVisibility(visible.get(ident, False))
        self.vtk.GetRenderWindow().Render()
