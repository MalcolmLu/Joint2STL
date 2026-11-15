from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QSignalBlocker, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QRadioButton,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSlider, QSpinBox, QSplitter, QStyle,
    QVBoxLayout, QWidget)

from .dicom_io import load_series, scan_directory, recommended_series
from .meshing import export_stl, segment_meshes
from .model import BONES, Seed, Session, count_labels, load_project, save_project
from .segmentation import (assign_component, clean_segment, initial_segmentation,
    inside, install_initial, paint, plane_box, split_from_seeds)
from .views import SlicePanel, SurfacePanel,plane_display_title
from .controls import MenuSelector,SegmentItem,SegmentList,PrecisionSpinBox,PrecisionDoubleSpinBox
from .workflow import prepare_initial, recognize_existing, install_recognition
from .preprocessing import remove_segmented_table
from .recovery import recover_ct_edges
from .resources import detect_memory, memory_policy
from .mr_policy import require_mri_modeling_source,native_acquisition_plane,sagittal_only
from . import __version__


STYLE = '''
QMainWindow, QDialog { background: #101824; }
QWidget { color: #dce5ef; font-size: 12px; }
QLabel#muted { color: #8a9bb1; font-size: 11px; }
QLabel#step { color: #68d5c4; font-size: 12px; font-weight: 600; padding-top: 6px; }
QFrame#sidebar { background: #151f2e; border: 1px solid #263449; border-radius: 10px; }
QWidget#sidebarBody { background: #151f2e; }
QFrame#viewPanel { background: #111b2a; border: 1px solid #29394d; border-radius: 8px; }
QPushButton { background: #223148; border: 1px solid #34465f; border-radius: 6px; padding: 7px 10px; }
QPushButton:hover { background: #2c405b; border-color: #6a93b3; }
QPushButton:pressed, QPushButton:checked { background: #174d51; border-color: #54c9b9; color: #a8f3e5; }
QPushButton:disabled { background: #1c2635; color: #596b82; border-color: #273449; }
QPushButton#primary { background: #237f74; color: white; border: 1px solid #3ba495; font-weight: 600; padding: 9px; }
QPushButton#primary:hover { background: #2b9587; }
QPushButton#primary:disabled { background: #20443f; color: #688880; border-color: #2e5d53; }
QPushButton#compact { padding: 4px 7px; font-size: 11px; }
QSpinBox, QDoubleSpinBox, QLineEdit { background: #101a28; border: 1px solid #35465e; border-radius: 5px; padding: 5px; min-height: 18px; }
QMenu { background:#192738; border:1px solid #456079; padding:4px; }
QMenu::item { padding:6px 22px; }
QMenu::item:selected { background:#286d68; }
QFrame#segmentList { background: #101a28; border: 1px solid #304159; border-radius: 5px; }
QWidget#boneRows { background: #101a28; }
QLineEdit#boneName { background: transparent; border: none; padding: 2px; }
QRadioButton::indicator { width: 13px; height: 13px; border: 1px solid #72889c; border-radius: 7px; background:#152132; }
QRadioButton::indicator:checked { background: #6cd6bd; border-color:#92f0da; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #101824; width: 7px; }
QScrollBar::handle:vertical { background: #40536b; min-height: 25px; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSlider::groove:horizontal { height: 4px; background: #2b3c53; border-radius: 2px; }
QSlider::handle:horizontal { background: #71cbbb; width: 11px; margin: -4px 0; border-radius: 5px; }
QProgressBar { border: none; background: #233247; border-radius: 3px; height: 6px; }
QProgressBar::chunk { background: #5acdb6; border-radius: 3px; }
QCheckBox { spacing: 6px; }
QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #6f849d; border-radius: 3px; background: #152132; }
QCheckBox::indicator:checked { background: #4bb59f; border-color: #82dfc9; }
QToolTip { color: #e5edf6; background: #24364c; border: 1px solid #5f819e; }
'''


class Cancelled(Exception):
    pass


class Task(QThread):
    progress = Signal(int, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, parent):
        super().__init__(parent); self.fn = fn

    def report(self, value, text):
        if self.isInterruptionRequested():
            raise Cancelled('已取消。')
        self.progress.emit(value, text)

    def run(self):
        try:
            result = self.fn(self.report)
            # Once a save/export commits, deliver its result even if cancellation was requested.
            self.succeeded.emit(result)
        except Cancelled as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc) or type(exc).__name__)


class SeriesDialog(QDialog):
    def __init__(self, series, unreadable, parent):
        super().__init__(parent); self.setWindowTitle('选择重建序列'); self.resize(1030, 460)
        self.series = series; self.selected = None;self.selected_index=-1
        layout = QVBoxLayout(self)
        from .dicom_io import study_recommendations
        preferred = recommended_series(series)
        per_study = study_recommendations(series)
        usable_studies = {s.study_key for s in series if s.usable}
        multiple = len(usable_studies) > 1
        title = QLabel('选择 CT / MRI 原始序列'); title.setStyleSheet('font-size:18px;font-weight:600;'); layout.addWidget(title)
        if multiple:
            advice = f'发现 {len(usable_studies)} 项可重建检查。请先核对所属检查及 CT / MRI 类型，再手动选用序列。各检查分别给出序列提示。'
        else:
            advice = ('MRI 优先长 TR 的矢状采集，核对 TE / TR、层厚与间距；也可手动选择 T1 或其他方向。'
                      if any(getattr(s,'modality','CT')=='MR' and s.usable for s in series) else
                      '优先软组织薄层（层厚和层间距均 ≤ 1.5 mm）；Bone / 锐利核不作自动推荐。' if preferred else '未找到明确的软组织薄层序列，请核对重建核和层厚后手动选择。')
        help_text = QLabel(advice + ' 定位片、报告和不规则序列不可重建。' + (f' 另有 {unreadable} 个文件未识别。' if unreadable else ''))
        help_text.setWordWrap(True)
        help_text.setObjectName('muted');layout.addWidget(help_text)
        scroll=QScrollArea();scroll.setWidgetResizable(True)
        body=QWidget();body.setObjectName('sidebarBody');scroll.setWidget(body)
        rows=QVBoxLayout(body);self.choices=QButtonGroup(self);self.choice_buttons=[]
        for row,s in enumerate(series):
            card=QFrame();card.setObjectName('viewPanel');line=QHBoxLayout(card)
            radio=QRadioButton('选用');radio.setEnabled(s.usable);self.choices.addButton(radio,row)
            group=f'{s.study_label} · ' if multiple else ''
            radio.setAccessibleName(f'{group}选择序列 {s.number}：{s.description}，{len(s.files)} 层')
            radio.toggled.connect(lambda checked,i=row:self.select_row(i) if checked else None)
            self.choice_buttons.append(radio);line.addWidget(radio)
            info=QVBoxLayout();name=QLabel(f'{group}{s.source_label} · 序列 {s.number} · {s.description}');name.setWordWrap(True);info.addWidget(name)
            if s.usable:
                spacing=' × '.join(f'{v:.3f}' for v in s.spacing)
                thickness=f'{s.thickness:.3f}' if s.thickness is not None else '—'
                mr=getattr(s,'modality','CT')=='MR'
                if mr:
                    plane={'sagittal':'矢状','coronal':'冠状','axial':'轴位'}.get(s.acquisition_plane,s.acquisition_plane)
                    te='—' if s.echo_time_ms is None else f'{s.echo_time_ms:g}'
                    tr='—' if s.repetition_time_ms is None else f'{s.repetition_time_ms:g}'
                    detail=f'MRI {plane} · {len(s.files)} 层 | 像素与层间距 {spacing} mm | 层厚 {thickness} mm\nTE {te} ms · TR {tr} ms | {s.recommendation_note}'
                    radio.setAccessibleName(f'{group}MRI {plane}，{len(s.files)} 层，TE {te}，TR {tr}')
                else:detail=f'CT · {len(s.files)} 层  |  像素与层间距 {spacing} mm  |  标称层厚 {thickness} mm\n重建核：{s.kernel or "未提供"}  |  {s.recommendation_note}'
                if s is per_study.get(s.study_key):detail+=('  | 本检查 MRI 推荐' if mr else '  |  本检查推荐：软组织薄层')
                if s is preferred and not multiple:radio.setChecked(True)
            else:detail=s.reason
            subtitle=QLabel(detail);subtitle.setObjectName('muted');subtitle.setWordWrap(True);info.addWidget(subtitle)
            line.addLayout(info,1);rows.addWidget(card)
        rows.addStretch();layout.addWidget(scroll,1)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Open|QDialogButtonBox.StandardButton.Cancel)
        self.open_button=buttons.button(QDialogButtonBox.StandardButton.Open)
        self.open_button.setText('载入选中序列');self.open_button.setEnabled(self.selected_index>=0)
        buttons.accepted.connect(self.accept_selected);buttons.rejected.connect(self.reject);layout.addWidget(buttons)

    def select_row(self,index):
        self.selected_index=index
        if hasattr(self,'open_button'):self.open_button.setEnabled(True)

    def accept_selected(self):
        row=self.selected_index
        if row<0 or not self.series[row].usable:
            QMessageBox.information(self,'无法使用','请选择状态为“可重建”的原始轴位序列。');return
        self.selected=self.series[row];self.accept()


class StlExportDialog(QDialog):
    def __init__(self,session,current_id,parent=None):
        super().__init__(parent);self.setWindowTitle('STL 导出范围与方式');self.resize(560,460)
        counts=session.counts()
        self.mri=session.volume.is_mr
        self.checked_ids=[s.id for s in session.segments if s.visible and counts[s.id]>0]
        self.all_ids=[s.id for s in session.segments if counts[s.id]>0]
        self.current_ids=[current_id] if current_id in self.all_ids else []
        layout=QVBoxLayout(self);heading=QLabel('选择骨骼范围与文件组合方式');heading.setStyleSheet('font-size:17px;font-weight:600');layout.addWidget(heading)
        scope_box=QFrame();scope_box.setObjectName('viewPanel');scope_layout=QVBoxLayout(scope_box);scope_layout.addWidget(QLabel('骨骼范围'))
        self.scope=QButtonGroup(self)
        current=session.segment(current_id)
        noun='结构' if self.mri else '骨'
        scopes=[f'列表中勾选的{noun}（{len(self.checked_ids)}）',f'所有非空{noun}（{len(self.all_ids)}）',f'当前{noun}：'+(current.name if current else '未选中')]
        for i,text in enumerate(scopes):
            radio=QRadioButton(text);self.scope.addButton(radio,i);scope_layout.addWidget(radio)
            if i==0:radio.setChecked(True)
            radio.toggled.connect(self.update_action)
        layout.addWidget(scope_box)
        mode_box=QFrame();mode_box.setObjectName('viewPanel');mode_layout=QVBoxLayout(mode_box);mode_layout.addWidget(QLabel('导出方式'));self.mode=QButtonGroup(self)
        for i,text in enumerate([('每个结构独立一个 STL' if self.mri else '每块骨独立一个 STL'),('所选结构合并为一个 STL' if self.mri else '所选骨合并为一个 STL'),'同时导出独立文件和组合文件']):
            radio=QRadioButton(text);self.mode.addButton(radio,i);mode_layout.addWidget(radio)
            if i==0:radio.setChecked(True)
        layout.addWidget(mode_box)
        note=QLabel('列表方框用于多选，圆钮表示当前编辑骨。组合文件保留各骨相对位置和独立骨壳，导入单位选择 mm。');note.setWordWrap(True);note.setObjectName('muted');layout.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Save|QDialogButtonBox.StandardButton.Cancel)
        self.action=buttons.button(QDialogButtonBox.StandardButton.Save);self.action.setText('下一步：选择输出位置')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons);self.update_action()

    def selected_ids(self):
        return [self.checked_ids,self.all_ids,self.current_ids][max(0,self.scope.checkedId())]

    def update_action(self,*_):
        if hasattr(self,'action'):self.action.setEnabled(bool(self.selected_ids()))

    def export_options(self):
        scope=['checked','all','current'][max(0,self.scope.checkedId())]
        noun='结构' if self.mri else '骨骼'
        return {'ids':self.selected_ids(),'include_hidden':True,'mode':['separate','combined','both'][max(0,self.mode.checkedId())],
                'selection_scope':scope,'combined_name':('全部' if scope=='all' else '选定')+noun+'_组合.stl'}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'骨模 · Joint2STL {__version__}')
        self.settings=QSettings('Joint2STL','Workbench')
        self.session: Session | None=None
        self.job=None;self._on_result=None;self.pending_close=False;self._apply_on_close=False
        self.active=0;self._updating=False;self._stroke=None;self.last_export=''
        self.current_tool='browse'
        self._processing_counts_key=None;self._processing_counts=None
        self.startup_memory=detect_memory()
        self._sagittal_layout=False
        self.setMinimumSize(1210,780)
        available=QApplication.primaryScreen().availableGeometry()
        self.resize(min(1600,available.width()-40),min(1020,available.height()-50))
        self._build()
        self.refresh_memory_status(self.startup_memory)
        self.setStyleSheet(STYLE)
        self._set_enabled(False)
        self._shortcuts()
        self.set_tool('browse')
        self._menus()

    def _menus(self):
        files=self.menuBar().addMenu('文件')
        for text,fn in [('导入 DICOM…',self.open_dicom),('打开项目…',self.open_project),('保存项目',self.save),('项目另存为…',lambda:self.save(True)),('导出 STL（范围 / 组合）…',self.export)]:
            action=files.addAction(text);action.triggered.connect(fn)
        tools=self.menuBar().addMenu('工具')
        tools.addAction('硬件与计算后端…').triggered.connect(self.show_hardware)
        help_menu=self.menuBar().addMenu('帮助')
        help_menu.addAction('快速使用说明').triggered.connect(self.show_help)

    def show_hardware(self):
        if self.job:return
        from .diagnostics import hardware_report
        self.run_job('检测本机 GPU 与模型环境',lambda p:hardware_report(),self.hardware_ready)

    def hardware_ready(self,report):
        import json
        box=QMessageBox(self);box.setWindowTitle('硬件与计算后端')
        backends='、'.join(report['surface_backends'].values())
        text=f'表面滤波：{backends}\n医学模型：{str(report["inference_device"] or "未安装").upper()}\n本地模型：'+('已就绪' if report.get('model_ready') else '待安装')
        if report.get('mri_bone_model'):text+='\nMRI 骨模型：'+('已就绪' if report['mri_bone_model']['ready'] else '待安装')+' · SegmentAnyBone'
        if report.get('mri_correction_model'):text+='\nMRI 局部纠错：'+('已就绪' if report['mri_correction_model']['ready'] else '待安装')+' · ScribblePrompt'
        policy=report['memory_policy'];operations=policy['operations']
        text+=f'\n系统内存：{policy["memory_total_bytes"]/1024**3:.1f} GB'
        if policy['memory_available_bytes'] is not None:text+=f' · 当前可用 {policy["memory_available_bytes"]/1024**3:.1f} GB'
        text+=f'\n当前体素上限：初分割 {operations["initial"]["limit_voxels"]/1e6:.1f} 百万 / 模型 {operations["model"]["limit_voxels"]/1e6:.1f} 百万'
        text+='\n三维视图使用 OpenGL GPU；连通域与网格校验使用 CPU。'
        box.setText(text);box.setDetailedText(json.dumps(report,ensure_ascii=False,indent=2));box.exec()
        self.status.setText('硬件检测完成')

    def show_help(self):
        if self.session and self.session.volume.is_mr:
            QMessageBox.information(self,'MRI 快速使用说明',
                '1. 选择原始（斜）矢状采集。斜轴位、斜冠状位不参与建模。\n'
                '2. 生成 MRI 骨候选，核对骨名、侧别及关节间隙；相连骨可用种子分离。\n'
                '3. 选 ACL / PCL 等结构后按 M 框选，默认仅当前采集层；空掩膜不代表缺失。\n'
                '4. 用“保留点 +”指出目标内部，用“排除点 −”指出误标处，再运行“局部模型纠错”。分离的应保留骨区可各放一个保留点。\n'
                '5. 运行“检查逐层轮廓与覆盖”，跳转核对提示层；提示不等于准确率。\n'
                '6. 逐层检查并保存项目，按非空结构导出 STL。单层韧带候选不能视为完整韧带；标注覆盖与完整性需分别核对。')
            return
        QMessageBox.information(self,'快速使用说明',
            '1. 导入 DICOM：CT 选择软组织薄层轴位，MRI 选择原始（斜）矢状采集。\n'
            '2. 选择膝、肩或全腿和侧别，用框选工具调整黄色范围。全腿模式按层分段初分割，跨段合并骨块；需对照原片命名。\n'
            '3. 初分割默认清理床板、石膏、夹板和支具，并生成骨名候选；已有项目也可单独处理。\n'
            '4. 骨列表圆钮单选编辑目标，标有“选”的方框可多选；“全选”可勾选所有骨。开启批量后精修所有勾选骨，按钮显示处理数量。骨名菜单仅用于命名当前骨。\n'
            '5. 画笔、橡皮、整块归属均可撤销。修改后点击“更新 3D”。\n'
            '6. 选择 STL 表面平滑程度并更新 3D。导出时选择勾选骨、所有骨或当前骨，可分别导出、组合为一个 STL，或同时导出。导入单位选 mm；保存项目可继续编辑。\n\n'
            '滚轮：切层；Ctrl+滚轮：切片缩放；右键拖动：切片平移。\n'
            'V 浏览 / R 框选 / B 画笔 / E 橡皮 / I 拾取 / A 归属 / S 种子。\n'
            '模型输出是轮廓建议；扫描范围端部会封口，精度需对照原 CT 检查。')

    def button(self,text,fn,primary=False):
        b=QPushButton(text);b.clicked.connect(fn)
        if primary:b.setObjectName('primary')
        return b

    def spin(self,lo,hi,value,decimals=0,suffix=''):
        s=PrecisionDoubleSpinBox() if decimals else PrecisionSpinBox();s.setRange(lo,hi);s.setValue(value);s.setSuffix(suffix)
        if decimals:s.setDecimals(decimals)
        return s

    def step(self,layout,text):
        label=QLabel(text);label.setObjectName('step');layout.addWidget(label)

    def _build(self):
        root=QWidget();self.setCentralWidget(root)
        outer=QVBoxLayout(root);outer.setContentsMargins(14,10,14,8);outer.setSpacing(9)
        header=QHBoxLayout()
        brand=QLabel('骨模  <span style="color:#77d7c2">Joint2STL</span>');brand.setStyleSheet('font-size:23px;font-weight:700;')
        header.addWidget(brand)
        subtitle=QLabel('CT / MRI 骨与韧带建模工作台');subtitle.setObjectName('muted');header.addWidget(subtitle);header.addStretch()
        self.import_button=self.button('导入 DICOM',self.open_dicom)
        self.open_button=self.button('打开项目',self.open_project)
        self.save_button=self.button('保存项目',self.save)
        self.export_button=self.button('导出 STL',self.export,True)
        for b in [self.import_button,self.open_button,self.save_button,self.export_button]:header.addWidget(b)
        outer.addLayout(header)
        splitter=QSplitter();outer.addWidget(splitter,1)
        self.sidebar=QFrame();self.sidebar.setObjectName('sidebar');self.sidebar.setMinimumWidth(300);self.sidebar.setMaximumWidth(480)
        side_outer=QVBoxLayout(self.sidebar);side_outer.setContentsMargins(5,5,5,5)
        scroll=QScrollArea();scroll.setWidgetResizable(True);side_outer.addWidget(scroll);self.sidebar_scroll=scroll
        inner=QWidget();inner.setObjectName('sidebarBody');scroll.setWidget(inner)
        left=QVBoxLayout(inner);left.setContentsMargins(10,5,10,10);left.setSpacing(8)
        self.step(left,'01   数据与范围')
        self.dataset=QLabel('尚未导入影像\n请选择 DICOM CT / MR 文件夹');self.dataset.setWordWrap(True);self.dataset.setObjectName('muted');left.addWidget(self.dataset)
        f=QFormLayout();f.setHorizontalSpacing(10)
        self.joint=MenuSelector();self.joint.addItem('膝关节','knee');self.joint.addItem('肩关节','shoulder');self.joint.addItem('全腿（分段）','leg');self.joint.currentIndexChanged.connect(self.joint_changed)
        self.side=MenuSelector();self.side.addItems(['双侧','左侧','右侧','未指定']);self.side.currentTextChanged.connect(self.side_changed)
        f.addRow('建模部位',self.joint);f.addRow('侧别',self.side);left.addLayout(f)
        row=QHBoxLayout();self.side_roi=self.button('侧别建议范围',self.suggest_roi);self.full_roi=self.button('完整视野',self.reset_roi)
        self.side_roi.setToolTip('按 LPS 方向建议半幅视野，请在切片确认实际中线。单侧扫描可选完整视野。')
        row.addWidget(self.side_roi);row.addWidget(self.full_roi);left.addLayout(row)
        g=QGridLayout();g.addWidget(QLabel('范围（含首末层）'),0,0);g.addWidget(QLabel('起始'),0,1);g.addWidget(QLabel('终止'),0,2)
        self.roi_spins=[]
        for axis,title in enumerate(['层 Z','行 Y','列 X']):
            a=self.spin(1,9999,1);b=self.spin(1,9999,1)
            a.setObjectName(f'roi{axis}Start');b.setObjectName(f'roi{axis}End')
            a.valueChanged.connect(self.roi_from_spins);b.valueChanged.connect(self.roi_from_spins)
            g.addWidget(QLabel(title),axis+1,0);g.addWidget(a,axis+1,1);g.addWidget(b,axis+1,2);self.roi_spins.append((a,b))
        left.addLayout(g)
        help_label=QLabel('用“框选”在任一切片拖出范围，滚轮切层。\n双侧按 LPS 方向识别左右；单侧扫描请选侧别。');help_label.setObjectName('muted');help_label.setWordWrap(True);left.addWidget(help_label)
        self.step(left,'02   自动初分割')
        self.memory_status=QLabel();self.memory_status.setObjectName('muted');self.memory_status.setWordWrap(True);left.addWidget(self.memory_status)
        form=QFormLayout()
        self.ct_threshold_form=form
        self.threshold=self.spin(-500,5000,200,suffix=' HU');self.threshold.setObjectName('threshold')
        self.upper=self.spin(-100,32000,4000,suffix=' HU')
        self.min_size=self.spin(1,100000,200,suffix=' mm³')
        form.addRow('骨阈值下限',self.threshold);form.addRow('上限',self.upper);form.addRow('最小骨块体积',self.min_size);left.addLayout(form)
        self.fill=QCheckBox('填充骨内部（生成实体外形）');self.fill.setChecked(True);left.addWidget(self.fill)
        self.auto_table=QCheckBox('自动去床板、石膏和支具');self.auto_table.setChecked(True);left.addWidget(self.auto_table)
        self.auto_names=QCheckBox('初分割后自动识别骨名');self.auto_names.setChecked(True);left.addWidget(self.auto_names)
        self.auto_recover=QCheckBox('自动补全 CT 支持的骨缘');self.auto_recover.setChecked(True);left.addWidget(self.auto_recover)
        self.auto_button=self.button('自动初分割',self.auto_segment,True);self.auto_button.setObjectName('primary');left.addWidget(self.auto_button)
        self.mri_panel=QFrame();self.mri_panel.setObjectName('viewPanel');mr=QVBoxLayout(self.mri_panel)
        mr.addWidget(QLabel('MRI 骨结构与交叉韧带'))
        self.mri_info=QLabel('保留原始 MRI 信号与采集网格。框选或画笔标注结构；韧带存在状态单独记录。');self.mri_info.setWordWrap(True);self.mri_info.setObjectName('muted');mr.addWidget(self.mri_info)
        self.mri_bone_button=self.button('MRI 骨模型初分割',self.mri_bone_initial,True);mr.addWidget(self.mri_bone_button)
        self.mri_labels_button=self.button('补齐骨与 ACL / PCL 标签',self.prepare_mri_labels);mr.addWidget(self.mri_labels_button)
        self.mri_acl_absent=QCheckBox('本病例 ACL 已缺失（用户确认）');self.mri_acl_absent.toggled.connect(self.acl_absence_changed);mr.addWidget(self.mri_acl_absent)
        form_mr=QFormLayout();self.mri_presence=MenuSelector()
        for text,value in [('未评估','unknown'),('确认可见','present'),('确认缺失','absent')]:self.mri_presence.addItem(text,value)
        self.mri_presence.currentIndexChanged.connect(self.mri_presence_changed)
        form_mr.addRow('当前韧带状态',self.mri_presence)
        self.mri_span=PrecisionSpinBox();self.mri_span.setRange(0,4);self.mri_span.setValue(0);self.mri_span.setSuffix(' 层')
        form_mr.addRow('框选时向两侧传播',self.mri_span);mr.addLayout(form_mr)
        self.mri_prompt_button=self.button('框选标注当前结构 M',lambda:self.set_tool('model_box'));mr.addWidget(self.mri_prompt_button)
        self.mri_refiner=MenuSelector();self.mri_refiner.addItem('原模型 · LiteMedSAM','litemedsam');self.mri_refiner.addItem('局部纠错 · ScribblePrompt','scribbleprompt')
        self.mri_refiner.currentIndexChanged.connect(self.mri_refiner_changed)
        refine_form=QFormLayout();refine_form.addRow('批量精修模型',self.mri_refiner);mr.addLayout(refine_form)
        row=QHBoxLayout();self.mri_correct_button=self.button('局部模型纠错',self.mri_correct_current,True)
        self.mri_clear_points=self.button('清除本层提示',self.clear_mri_correction_points)
        row.addWidget(self.mri_correct_button);row.addWidget(self.mri_clear_points);mr.addLayout(row)
        self.mri_correction_status=QLabel('保留点 + 放在目标内部；排除点 − 放在误标处。局部纠错仅处理当前矢状采集层。')
        self.mri_correction_status.setWordWrap(True);self.mri_correction_status.setObjectName('muted');mr.addWidget(self.mri_correction_status)
        self.mri_review_button=self.button('检查逐层轮廓与覆盖',self.review_mri_masks);mr.addWidget(self.mri_review_button)
        self.mri_review_flags=MenuSelector();self.mri_review_flags.addItem('轮廓检查：尚未运行',None)
        self.mri_review_flags.currentIndexChanged.connect(self.jump_mri_review);mr.addWidget(self.mri_review_flags)
        self.mri_state=QLabel();self.mri_state.setObjectName('muted');self.mri_state.setWordWrap(True);mr.addWidget(self.mri_state)
        self.mri_panel.hide();left.addWidget(self.mri_panel)
        self.table_button=self.button('去床板 / 石膏 / 支具',self.remove_table);self.table_button.setToolTip('从当前范围的骨分割中排除床板、石膏、夹板和外部支具；与骨相连的结构会提示核对。保留原始 CT，可撤销。');left.addWidget(self.table_button)
        self.cleanup_status=QLabel('清理分割中的外部附着物；原始 CT 保留。');self.cleanup_status.setObjectName('muted');self.cleanup_status.setWordWrap(True);left.addWidget(self.cleanup_status)
        note=QLabel('按密度与连通域生成骨块；切片内填孔可能\n改变真实孔隙，请对照切片修正。');note.setObjectName('muted');self.ct_density_note=note;left.addWidget(note)
        self.leg_note=QLabel('全腿模式分段初分割，跨段合并骨块；整腿不自动精修、补全或命名。可缩小黄色范围后精修局部。')
        self.leg_note.setWordWrap(True);self.leg_note.setObjectName('muted');self.leg_note.hide();left.addWidget(self.leg_note)
        self.step(left,'03   骨标签与修正')
        self.recognize_button=self.button('自动识别并命名',self.recognize_bones);left.addWidget(self.recognize_button)
        self.recognition_status=QLabel('按解剖位置与形态识别；手工确认的名称会保留。');self.recognition_status.setObjectName('muted');self.recognition_status.setWordWrap(True);left.addWidget(self.recognition_status)
        self.recover_button=self.button('CT 骨缘补全',self.recover_edges);self.recover_button.setToolTip('沿 CT 皮质补回漏分割；保留现有标签。范围由“处理所有勾选骨”控制。');left.addWidget(self.recover_button)
        self.repair_status=QLabel('CT 骨缘补全保留现有分割，可撤销。');self.repair_status.setObjectName('muted');self.repair_status.setWordWrap(True);left.addWidget(self.repair_status)
        row=QHBoxLayout();row.addWidget(QLabel('骨骼多选'))
        self.check_all_button=self.button('全选',lambda:self.set_bone_checks(True));self.check_none_button=self.button('全不选',lambda:self.set_bone_checks(False))
        for button in [self.check_all_button,self.check_none_button]:button.setObjectName('compact');row.addWidget(button)
        left.addLayout(row)
        self.segment_list=SegmentList();self.segment_list.setObjectName('segmentList');self.segment_list.setMinimumHeight(155);self.segment_list.setMaximumHeight(230)
        self.segment_list.currentItemChanged.connect(self.selected_segment)
        self.segment_list.itemChanged.connect(self.segment_item_changed)
        left.addWidget(self.segment_list)
        row=QHBoxLayout()
        self.edit_target=QLabel('当前编辑：未选择');self.edit_target.setWordWrap(True);row.addWidget(self.edit_target,1)
        self.delete_segment_button=self.button('删除当前标签',self.delete_current_segment)
        self.delete_segment_button.setObjectName('compact')
        self.delete_segment_button.setToolTip('删除当前标签及全部层面的对应标注、种子和提示点；其他标签保留，可撤销恢复。')
        row.addWidget(self.delete_segment_button);left.addLayout(row)
        hint=QLabel('圆钮：单选当前编辑骨。带“选”的方框：多选显示、批量精修 / 补全及导出。');hint.setObjectName('muted');hint.setWordWrap(True);left.addWidget(hint)
        self.assign_2d=QCheckBox('归属仅当前切片（不勾选为三维归属）');left.addWidget(self.assign_2d)
        name_hint=QLabel('骨名预设（仅用于命名当前骨）');name_hint.setObjectName('muted');self.preset_hint=name_hint;left.addWidget(name_hint)
        row=QHBoxLayout();self.preset=MenuSelector();self.update_presets();row.addWidget(self.preset,1)
        self.preset.setAccessibleDescription('骨名预设，仅用于命名当前骨；精修范围由骨列表方框决定')
        self.rename_button=self.button('命名',self.rename_current);self.add_button=self.button('+ 新骨',self.add_bone)
        row.addWidget(self.rename_button);row.addWidget(self.add_button);left.addLayout(row)
        row=QHBoxLayout();self.split_button=self.button('运行种子分离',self.split_seeds);self.clear_seeds_button=self.button('清除种子',self.clear_seeds)
        row.addWidget(self.split_button);row.addWidget(self.clear_seeds_button);left.addLayout(row)
        self.ai_all=QCheckBox('批量精修 / 补全所有勾选骨');self.ai_all.setChecked(True);left.addWidget(self.ai_all)
        self.ai_all.setToolTip('开启时处理列表方框勾选的全部非空骨；关闭时仅处理圆钮选中的当前骨')
        self.ai_all.toggled.connect(self.refresh_processing_scope)
        self.ai_scope_status=QLabel('未载入骨标签');self.ai_scope_status.setObjectName('muted');self.ai_scope_status.setWordWrap(True);left.addWidget(self.ai_scope_status)
        self.ai_button=self.button('本地模型精修',self.ai_refine,True);left.addWidget(self.ai_button)
        self.ai_status=QLabel('LiteMedSAM · 自动选择 MPS / CUDA / CPU');self.ai_status.setObjectName('muted');self.ai_status.setWordWrap(True);left.addWidget(self.ai_status)
        self.ai_flags=MenuSelector();self.ai_flags.addItem('异常层：尚无记录',None);self.ai_flags.currentIndexChanged.connect(self.jump_ai_flag);left.addWidget(self.ai_flags)
        row=QHBoxLayout();self.fill_button=self.button('当前骨髓腔填充',lambda:self.cleanup('fill'));self.fill_button.setToolTip('沿三平面填充骨内空隙，可处理扫描端部敞开的髓腔；不覆盖其他骨标签。');self.largest_button=self.button('保留最大块',lambda:self.cleanup('largest'))
        row.addWidget(self.fill_button);row.addWidget(self.largest_button);left.addLayout(row)
        row=QHBoxLayout();self.undo_button=self.button('撤销',lambda:self.undo(False));self.redo_button=self.button('重做',lambda:self.undo(True));self.clear_button=self.button('清空当前骨',lambda:self.cleanup('clear'))
        for b in [self.undo_button,self.redo_button,self.clear_button]:row.addWidget(b)
        left.addLayout(row)
        self.step(left,'04   表面与导出')
        form=QFormLayout();self.smoothing=self.spin(0,2,0.35,2,' mm');form.addRow('体素滤波 σ',self.smoothing);left.addLayout(form)
        self.smoothing.valueChanged.connect(self.surface_parameter_changed)
        self.detail_guard=QCheckBox('保护薄骨缘（限制体素滤波）');self.detail_guard.setChecked(True);left.addWidget(self.detail_guard)
        self.detail_guard.stateChanged.connect(self.surface_parameter_changed)
        self.filter_status=QLabel('载入 CT 后显示实际滤波强度。');self.filter_status.setObjectName('muted');self.filter_status.setWordWrap(True);left.addWidget(self.filter_status)
        self.mesh_smoothing=MenuSelector()
        for text,key in [('关闭','off'),('轻度 · 位移上限 0.15 mm','light'),('标准 · 位移上限 0.30 mm','standard'),('加强 · 位移上限 0.50 mm','strong')]:self.mesh_smoothing.addItem(text,key)
        self.mesh_smoothing.setCurrentIndex(2);form.addRow('STL 表面平滑',self.mesh_smoothing)
        self.mesh_smoothing.currentIndexChanged.connect(self.surface_parameter_changed)
        self.backend=MenuSelector()
        for text,key in [('自动选择','auto'),('CPU','cpu'),('Apple GPU · Metal','metal'),('NVIDIA GPU · CUDA','cuda')]:self.backend.addItem(text,key)
        form.addRow('表面计算后端',self.backend)
        self.backend.currentIndexChanged.connect(self.backend_changed)
        tip=QLabel('预览与导出共享原始分辨率表面。\nSTL 按 mm 导入；扫描边缘会自动封口。');tip.setObjectName('muted');tip.setWordWrap(True);left.addWidget(tip)
        self.export_side_button=self.button('导出 STL（范围 / 组合）',self.export,True);left.addWidget(self.export_side_button)
        left.addStretch()
        splitter.addWidget(self.sidebar)
        self.work=QWidget();wl=QVBoxLayout(self.work);wl.setContentsMargins(10,0,0,0);wl.setSpacing(8)
        toolrow=QHBoxLayout();toolrow.setSpacing(4)
        self.tool_group=QButtonGroup(self);self.tool_buttons={}
        for key,text in [('browse','浏览 V'),('roi','框选 R'),('paint','画笔 B'),('erase','橡皮 E'),('pick','拾取 I'),('assign','归属 A'),('seed','种子 S'),('model_box','MRI 框选 M'),('mr_positive','保留点 +'),('mr_negative','排除点 −')]:
            b=QPushButton(text);b.setCheckable(True);b.setObjectName('compact');self.tool_group.addButton(b);toolrow.addWidget(b);self.tool_buttons[key]=b
            b.toggled.connect(lambda checked,k=key:self.set_tool(k) if checked else None)
        self.tool_buttons['browse'].blockSignals(True)
        self.tool_buttons['browse'].setChecked(True)
        self.tool_buttons['browse'].blockSignals(False)
        toolrow.addStretch();toolrow.addWidget(QLabel('笔刷'))
        self.brush=self.spin(0.5,30,3,1,' mm');self.brush.setMaximumWidth(92);self.brush.valueChanged.connect(self.display_settings);toolrow.addWidget(self.brush)
        wl.addLayout(toolrow)
        displayrow=QHBoxLayout();self.context=QLabel('导入 CT，选择范围后开始初分割。');self.context.setObjectName('muted');displayrow.addWidget(self.context,1)
        displayrow.addWidget(QLabel('窗宽'));self.ww=self.spin(1,20000,1600);self.ww.setMaximumWidth(76)
        displayrow.addWidget(self.ww);displayrow.addWidget(QLabel('窗位'));self.wc=self.spin(-2000,10000,450);self.wc.setMaximumWidth(76);displayrow.addWidget(self.wc)
        displayrow.addWidget(QLabel('叠加'));self.opacity=QSlider(Qt.Orientation.Horizontal);self.opacity.setRange(0,85);self.opacity.setValue(40);self.opacity.setMaximumWidth(76);displayrow.addWidget(self.opacity)
        for w in [self.ww,self.wc,self.opacity]:w.valueChanged.connect(self.display_settings)
        wl.addLayout(displayrow)
        grid=QGridLayout();grid.setSpacing(8);self.panels={};self.view_grid=grid
        for plane,title,color,pos in [('axial','轴位 · 采集平面','#77d4c3',(0,0)),('coronal','冠状重排','#8fb9ee',(0,1)),('sagittal','矢状重排','#f0bf7e',(1,0))]:
            panel=SlicePanel(plane,title,color);self.panels[plane]=panel;grid.addWidget(panel,*pos)
            panel.slider.valueChanged.connect(lambda value,p=plane:self.slide_changed(p,value))
            v=panel.view;v.navigate.connect(self.navigate);v.roi_changed.connect(self.roi_from_view)
            v.interaction_started.connect(self.sync_tool_from_buttons)
            v.model_box.connect(self.mri_prompt)
            v.slice_scrolled.connect(self.scroll_slice);v.stroke_start.connect(self.begin_stroke);v.stroke_point.connect(self.stroke_point);v.stroke_end.connect(self.end_stroke);v.point_action.connect(self.point_action)
        self.surface=SurfacePanel();grid.addWidget(self.surface,1,1);self.surface.refresh_requested.connect(self.refresh_surface)
        grid.setRowStretch(0,1);grid.setRowStretch(1,1);grid.setColumnStretch(0,1);grid.setColumnStretch(1,1);wl.addLayout(grid,1)
        splitter.addWidget(self.work);splitter.setSizes([330,1100]);splitter.setCollapsible(0,False)
        bottom=QHBoxLayout();self.status=QLabel('本地处理 · 分割结果请对照原始切片核对');self.status.setObjectName('muted');bottom.addWidget(self.status,1)
        self.progress=QProgressBar();self.progress.setRange(0,100);self.progress.setTextVisible(False);self.progress.setMaximumWidth(200);self.progress.hide();bottom.addWidget(self.progress)
        self.cancel_button=self.button('取消当前步骤',self.cancel_job);self.cancel_button.setObjectName('compact');self.cancel_button.hide();bottom.addWidget(self.cancel_button)
        self.project_status=QLabel(f'v{__version__} · {self.startup_memory.total_bytes/1024**3:.0f}GB 内存');self.project_status.setObjectName('muted');bottom.addWidget(self.project_status);outer.addLayout(bottom)

    def _shortcuts(self):
        for key,tool in [('V','browse'),('R','roi'),('B','paint'),('E','erase'),('I','pick'),('A','assign'),('S','seed'),('M','model_box')]:
            action=QAction(self);action.setShortcut(key);action.triggered.connect(lambda checked=False,t=tool:self.set_tool(t));self.addAction(action)
        for shortcut,fn in [(QKeySequence.StandardKey.Save,self.save),(QKeySequence.StandardKey.SaveAs,lambda:self.save(True)),(QKeySequence.StandardKey.Undo,lambda:self.undo(False)),(QKeySequence.StandardKey.Redo,lambda:self.undo(True))]:
            action=QAction(self);action.setShortcut(shortcut);action.triggered.connect(fn);self.addAction(action)

    def _set_enabled(self,loaded):
        busy=self.job is not None
        self.sidebar.setEnabled(loaded and not busy);self.work.setEnabled(not busy)
        for b in [self.save_button,self.export_button]:b.setEnabled(loaded and not busy)
        for b in [self.import_button,self.open_button]:b.setEnabled(not busy)
        self.sync_joint_controls()

    def sync_joint_controls(self):
        if not hasattr(self,'leg_note'):return
        leg=bool(self.session and self.session.joint=='leg')
        self.leg_note.setVisible(leg)
        mr=bool(self.session and self.session.volume.is_mr)
        self.joint.setEnabled(not mr)
        self.auto_names.setEnabled(not leg and not mr)
        self.auto_recover.setEnabled(not leg and not mr)
        self.recognize_button.setEnabled(not leg and not mr)
        self.auto_button.setText('全腿分段初分割' if leg else '自动初分割')
        if hasattr(self,'mri_panel'):
            self.mri_panel.setVisible(mr)
            for key in ('model_box','mr_positive','mr_negative'):self.tool_buttons[key].setVisible(mr)
            for widget in [self.threshold,self.upper]:
                widget.setVisible(not mr);self.ct_threshold_form.labelForField(widget).setVisible(not mr)
            for widget in [self.fill,self.auto_table,self.auto_names,self.auto_recover,self.auto_button,self.table_button,self.cleanup_status,self.ct_density_note,self.recognize_button,self.recover_button,self.repair_status]:widget.setVisible(not mr)
            self.sync_mri_controls()
        self.fit_sidebar_width()

    def fit_sidebar_width(self):
        # MRI controls have a wider minimum than the original CT sidebar.
        # Reserve the native scrollbar and frame so the label actions stay in view.
        scroll=self.sidebar_scroll;body=scroll.widget()
        body.layout().activate()
        margins=self.sidebar.layout().contentsMargins()
        bar=scroll.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent,None,scroll)
        frames=2*(self.sidebar.frameWidth()+scroll.frameWidth())
        width=max(300,body.minimumSizeHint().width()+margins.left()+margins.right()+bar+frames)
        self.sidebar.setMaximumWidth(max(480,width));self.sidebar.setMinimumWidth(width)

    def run_job(self,title,fn,on_result,apply_on_close=False):
        if self.job:return
        self.status.setText(title);self.progress.setValue(0);self.progress.show();self.cancel_button.show()
        self._on_result=on_result
        self._apply_on_close=apply_on_close
        task=Task(fn,self);self.job=task
        task.progress.connect(self.job_progress);task.succeeded.connect(self.job_success);task.failed.connect(self.job_failed)
        task.finished.connect(task.deleteLater);self._set_enabled(self.session is not None);task.start()

    def job_progress(self,value,text):
        self.progress.setValue(value);self.status.setText(text)

    def finish_job(self):
        self.job=None;self.progress.hide();self.cancel_button.hide();self._set_enabled(self.session is not None)

    def job_success(self,result):
        fn=self._on_result;self._on_result=None;self.finish_job()
        if self.pending_close:
            if self._apply_on_close:fn(result)
            QTimer.singleShot(100,self.close);return
        try:fn(result)
        except Exception as exc:self.show_error(str(exc))

    def job_failed(self,text):
        self._on_result=None;self.finish_job()
        if self.pending_close:
            if self._apply_on_close:
                self.pending_close=False;self.show_error(text)
            else:QTimer.singleShot(100,self.close)
            return
        if text.startswith('已取消'):self.status.setText(text)
        else:self.show_error(text)

    def cancel_job(self):
        if self.job:self.job.requestInterruption();self.status.setText('正在取消，等待当前计算步骤结束…')

    def show_error(self,text):
        self.status.setText(text);QMessageBox.warning(self,'操作未完成',text)

    def allow_replace(self):
        if self.job:return False
        if self.session and self.session.dirty:
            answer=QMessageBox.question(self,'尚有未保存的修改','当前项目有未保存的修改。放弃这些修改并继续？',QMessageBox.StandardButton.Discard|QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.Cancel)
            return answer==QMessageBox.StandardButton.Discard
        return True

    def open_dicom(self,directory=None):
        if isinstance(directory,bool):directory=None
        if not self.allow_replace():return
        directory=directory or QFileDialog.getExistingDirectory(self,'选择 DICOM 文件夹',self.settings.value('last_dicom',str(Path.home()/'Downloads')))
        if not directory:return
        self.settings.setValue('last_dicom',directory)
        self.run_job('扫描 DICOM 序列',lambda p:scan_directory(directory,p),self.scanned)

    def scanned(self,result):
        series,unreadable=result
        if not any(s.usable for s in series):self.show_error('没有可用的原始 CT / MRI 体序列。请检查卡片中的不可用原因。');return
        dialog=SeriesDialog(series,unreadable,self)
        if dialog.exec()==QDialog.DialogCode.Accepted:
            selected=dialog.selected
            self.run_job('载入影像',lambda p:load_series(selected,p),self.imported_volume)

    def imported_volume(self,volume):
        session=Session(volume)
        if volume.is_mr:
            session.side={'L':'左侧','R':'右侧'}.get(volume.reconstruction.get('laterality'),'未指定')
            session.parameters.update(fill=False,recover_edges=False,remove_table=False,min_mm3=100)
            from .mr_tools import ensure_mr_segments
            ensure_mr_segments(session)
        self.set_session(session)

    def open_project(self,path=None):
        if isinstance(path,bool):path=None
        if not self.allow_replace():return
        path=path or QFileDialog.getOpenFileName(self,'打开项目',self.settings.value('last_project_dir',str(Path.home()/'Documents')),'Joint2STL 项目 (*.j2stl)')[0]
        if not path:return
        self.run_job('打开本地项目',lambda p:load_project(path),self.set_session)

    def set_session(self,session):
        if session.volume.is_mr:session.parameters['mr_sagittal_only']=True
        self.session=session;self.active=session.segments[0].id if session.segments else 0;self._stroke=None
        self.mri_review_flags.blockSignals(True);self.mri_review_flags.clear()
        self.mri_review_flags.addItem('本会话轮廓检查：尚未运行',None);self.mri_review_flags.blockSignals(False)
        if session.project_path:self.settings.setValue('last_project_dir',str(Path(session.project_path).parent))
        self._updating=True
        self.joint.setCurrentIndex(self.joint.findData(session.joint));self.side.setCurrentText(session.side)
        for widget,key in [(self.threshold,'threshold'),(self.upper,'upper'),(self.min_size,'min_mm3'),(self.smoothing,'surface_mm')]:widget.setValue(session.parameters[key])
        self.fill.setChecked(session.parameters['fill'])
        self.auto_table.setChecked(session.parameters.get('remove_table',True))
        self.auto_names.setChecked(session.parameters.get('auto_name',True))
        self.auto_recover.setChecked(session.parameters.get('recover_edges',True))
        self.detail_guard.setChecked(session.parameters.get('preserve_detail',True))
        self.mesh_smoothing.setCurrentIndex(self.mesh_smoothing.findData(session.parameters.get('mesh_smoothing','standard')))
        self.backend.setCurrentIndex(self.backend.findData(session.parameters.get('compute_backend','auto')))
        self._updating=False;self.update_presets();self.sync_roi_spins()
        shape=session.volume.hu.shape;spacing=session.volume.spacing_zyx[::-1]
        self.dataset.setText(f'{session.volume.description or "原始轴位 CT"}\n{shape[2]} × {shape[1]} × {shape[0]}   |   {" × ".join(f"{n:.3f}" for n in spacing)} mm')
        if session.volume.reconstruction.get('kernel'):
            kind={'soft':'软组织','sharp':'骨 / 锐利','unknown':'待确认'}.get(session.volume.reconstruction.get('kind'),'待确认')
            self.dataset.setText(self.dataset.text()+f'\n重建核：{session.volume.reconstruction["kernel"]} · {kind}')
        if session.volume.is_mr:
            meta=session.volume.reconstruction
            self.dataset.setText(self.dataset.text()+f'\nMRI 原始信号 · TE {meta.get("echo_time_ms","—")} ms · TR {meta.get("repetition_time_ms","—")} ms')
            values=session.volume.values
            sample=values[::max(1,values.shape[0]//20)]
            nonzero=sample[sample>0]
            lo,hi=np.percentile(nonzero,[.5,99.5]) if len(nonzero) else (float(values.min()),float(values.max()))
            self.ww.setMaximum(max(20000,int(hi-lo)+10));self.wc.setRange(min(-2000,int(lo)-10),max(10000,int(hi)+10))
            self.ww.setValue(max(1,int(hi-lo)));self.wc.setValue(int((hi+lo)/2))
        else:self.ww.setValue(1600);self.wc.setValue(450)
        self.cross=[(s.start+s.stop-1)//2 for s in session.roi]
        if session.segments:
            from scipy import ndimage as ndi
            boxes=ndi.find_objects(session.labels)
            active_box=boxes[self.active-1] if 0<self.active<=len(boxes) else None
            if active_box:
                self.cross=[(b.start+b.stop-1)//2 for b in active_box]
        for plane,panel in self.panels.items():
            panel.title.setText(plane_display_title(session.volume,plane))
            panel.view.session=session;panel.view.cross=list(self.cross);panel.fit()
            panel.slider.blockSignals(True);panel.slider.setRange(0,shape[panel.view.fixed_axis]-1);panel.slider.setValue(self.cross[panel.view.fixed_axis]);panel.slider.setEnabled(True);panel.slider.blockSignals(False)
        self.surface.set_meshes([]);self.surface.bounds=None
        self._set_enabled(True);self.refresh_list();self.display_settings();self.navigate(self.cross);self.refresh_memory_status()
        self.refresh_cleanup_status()
        self.refresh_repair_status();self.refresh_filter_status()
        self.status.setText('MRI 已载入。选择骨或韧带标签，使用模型框选或画笔；缺失状态单独记录。' if session.volume.is_mr else 'CT 已载入。长扫描可选“全腿（分段）”，关节建模请先框选目标范围。')
        self.project_status.setText('项目已载入' if session.project_path else '新项目')
        if session.segments and np.any(session.labels) and (not session.volume.is_mr or native_acquisition_plane(session.volume)['sagittal']):QTimer.singleShot(100,self.refresh_surface)

    def save(self,force_dialog=False):
        if self.job or self.session is None:return
        path=self.session.project_path if not force_dialog else ''
        if not path:path=QFileDialog.getSaveFileName(self,'保存可继续编辑的项目',str(Path(self.settings.value('last_project_dir',str(Path.home()/'Documents')))/'关节建模.j2stl'),'Joint2STL 项目 (*.j2stl)')[0]
        if not path:return
        if not path.lower().endswith('.j2stl'):path+='.j2stl'
        self.sync_parameters()
        self.run_job('保存影像、结构标签和编辑结果',lambda p:save_project(self.session,path),self.saved,apply_on_close=True)

    def saved(self,path):
        self.session.project_path=path;self.session.dirty=False
        self.settings.setValue('last_project_dir',str(Path(path).parent));self.project_status.setText('已保存');self.status.setText(f'项目已保存：{Path(path).name}')

    def sync_parameters(self):
        if self.session:self.session.parameters.update(threshold=self.threshold.value(),upper=self.upper.value(),min_mm3=self.min_size.value(),fill=self.fill.isChecked(),surface_mm=self.smoothing.value(),compute_backend=self.backend.currentData(),remove_table=self.auto_table.isChecked(),auto_name=self.auto_names.isChecked(),mesh_smoothing=self.mesh_smoothing.currentData(),preserve_detail=self.detail_guard.isChecked(),recover_edges=self.auto_recover.isChecked())

    def joint_changed(self):
        self.update_presets()
        if self.session and not self._updating:
            self.session.joint=self.joint.currentData();self.session.dirty=True
            self.sync_joint_controls();self.refresh_memory_status()

    def side_changed(self):
        if self.session and not self._updating:self.session.side=self.side.currentText();self.session.dirty=True

    def update_presets(self):
        if not hasattr(self,'preset'):return
        self.preset.clear();self.preset.addItems(BONES[self.joint.currentData() or 'knee'])
        if self.session and self.session.volume.is_mr:self.preset.addItems(['前交叉韧带 ACL','后交叉韧带 PCL'])

    def desired_name(self):
        side=self.side.currentText()
        return (side+'_' if side in ('左侧','右侧') else '')+self.preset.currentText()

    def rename_current(self):
        if not self.session or not self.active:return
        if self.session.volume.is_mr:self.add_bone();return
        self.session.update_bone_names([{'id':self.active,'name':self.desired_name(),'reviewed':True}], '手工命名',preserve_reviewed=False)
        self.refresh_list();self.project_status.setText('有未保存修改')

    def add_bone(self):
        if not self.session:return
        if self.session.volume.is_mr:
            from .mr_tools import MR_STRUCTURES,ensure_mr_segments
            ids=ensure_mr_segments(self.session)
            expected=next((key for key,name,_ in MR_STRUCTURES if name==self.preset.currentText()),None)
            if expected:
                self.active=ids[expected];self.refresh_list();self.status.setText('已选择对应 MRI 结构标签；存在状态不会因命名预设而改变。');return
        seg=self.session.add_segment(self.desired_name());seg.reviewed=True;self.active=seg.id;self.refresh_list();self.status.setText('已建立空骨标签；可用画笔、整块归属或种子填入。')

    def refresh_list(self):
        if not self.session:return
        if self.session.segment(self.active) is None:
            self.active=self.session.segments[0].id if self.session.segments else 0
        counts=self.session.counts();self._updating=True
        existing={self.segment_list.item(i).data(Qt.ItemDataRole.UserRole):self.segment_list.item(i) for i in range(self.segment_list.count())}
        ids={s.id for s in self.session.segments}
        for i in range(self.segment_list.count()-1,-1,-1):
            if self.segment_list.item(i).data(Qt.ItemDataRole.UserRole) not in ids:
                self.segment_list.takeItem(i)
        for seg in self.session.segments:
            item=existing.get(seg.id)
            if item is None:
                item=SegmentItem(seg.name)
                item.setData(Qt.ItemDataRole.UserRole,seg.id)
                item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable|Qt.ItemFlag.ItemIsEditable)
                pix=QPixmap(12,12);pix.fill(QColor(seg.color));item.setIcon(QIcon(pix))
                self.segment_list.addItem(item)
            if item.text()!=seg.name:item.setText(seg.name)
            state=Qt.CheckState.Checked if seg.visible else Qt.CheckState.Unchecked
            if item.checkState()!=state:item.setCheckState(state)
            count=int(counts[seg.id]) if seg.id<len(counts) else 0
            item.row.setVisible(count>0 or seg.id==self.active or seg.reviewed or (self.session.volume.is_mr and seg.structure in ('femur','tibia','fibula','patella','acl','pcl')))
            state={'unknown':'待评估','present':'已确认可见','absent':'已标记缺失'}.get(seg.presence,'待评估')
            item.annotation_state.setText(state if seg.kind=='ligament' else '')
            item.annotation_state.setVisible(seg.kind=='ligament')
            item.setToolTip(f'{seg.name}\n体积：{count*self.session.volume.voxel_mm3/1000:.2f} cm³\n'+('人工命名' if seg.reviewed else '自动骨块，待人工命名'))
            if seg.recognition and not seg.reviewed:
                state='自动识别，待核对' if seg.recognition.get('status')=='identified' else '自动识别信息不足，待核对'
                item.setToolTip(f'{seg.name}\n体积：{count*self.session.volume.voxel_mm3/1000:.2f} cm³\n{state}\n'+'；'.join(seg.recognition.get('evidence',[])))
            item.setForeground(QColor('#8192a8' if count==0 else '#dce5ef'))
            if seg.id==self.active:self.segment_list.setCurrentItem(item)
        self._updating=False
        self.undo_button.setEnabled(bool(self.session.history.undo_stack));self.redo_button.setEnabled(bool(self.session.history.redo_stack))
        nonempty=[s for s in self.session.segments if s.id<len(counts) and counts[s.id]>0]
        automatic=sum(bool(s.recognition.get('name')) and s.recognition.get('status')!='check_connection' and not s.reviewed for s in nonempty)
        connected=sum(s.recognition.get('status')=='check_connection' and not s.reviewed for s in nonempty)
        manual=sum(s.reviewed for s in nonempty)
        self.recognition_status.setText(f'自动命名 {automatic} 块 / 疑似相连 {connected} 块 / 手工确认 {manual} 块 / 待识别 {len(nonempty)-automatic-connected-manual} 块；自动结果请核对。')
        self.refresh_ai_status()
        self.refresh_processing_scope()
        self.sync_mri_controls()
        self.display_settings()
        self.update_tool_context()
        self.delete_segment_button.setEnabled(bool(self.active) and not self.job)

    def selected_segment(self,item,previous):
        if self._updating or item is None:return
        self.active=item.data(Qt.ItemDataRole.UserRole);self.display_settings();self.refresh_ai_status();self.refresh_processing_scope();self.sync_mri_controls()
        self.update_tool_context()

    def delete_current_segment(self):
        if not self.session or self.job or not self.session.segment(self.active):return
        if self._stroke:self.end_stroke()
        ident=self.active;name=self.session.segment(ident).name
        index=next(i for i,s in enumerate(self.session.segments) if s.id==ident)
        self.session.remove_segment(ident)
        self.active=(self.session.segments[min(index,len(self.session.segments)-1)].id
                     if self.session.segments else 0)
        self.after_edit()
        self.status.setText(f'已删除“{name}”标签及全部层面的标注；可撤销恢复。')

    def sync_mri_controls(self):
        if not hasattr(self,'mri_panel'):return
        mr=bool(self.session and self.session.volume.is_mr)
        if mr!=self._sagittal_layout:
            grid=self.view_grid
            for widget in [*self.panels.values(),self.surface]:grid.removeWidget(widget)
            if mr:
                self.panels['coronal'].hide();self.panels['sagittal'].hide()
                grid.addWidget(self.panels['axial'],0,0,2,1);grid.addWidget(self.surface,0,1,2,1)
            else:
                for key,pos in [('axial',(0,0)),('coronal',(0,1)),('sagittal',(1,0))]:
                    self.panels[key].show();grid.addWidget(self.panels[key],*pos)
                grid.addWidget(self.surface,1,1)
            self._sagittal_layout=mr
        if not mr:
            if self.panels['axial'].view.tool in ('model_box','mr_positive','mr_negative'):self.set_tool('browse')
            self.fill_button.setEnabled(True)
            self.fill_button.setToolTip('沿三平面填充骨内空隙，可处理扫描端部敞开的髓腔；不覆盖其他骨标签。')
            self.ai_button.setEnabled(not self.job);self.export_side_button.setEnabled(bool(self.session) and not self.job)
            if hasattr(self.surface,'caption'):self.surface.caption.setText('三维骨模型')
            self.preset_hint.setText('骨名预设（仅用于命名当前骨）');self.rename_button.setText('命名');self.add_button.setText('+ 新骨')
            return
        self.preset_hint.setText('MRI 结构预设（选择对应标签）');self.rename_button.setText('选择结构');self.add_button.setText('补齐标签')
        bone=self.session.segment(self.active)
        source_ok=native_acquisition_plane(self.session.volume)['sagittal']
        meta=self.session.volume.reconstruction
        self.mri_info.setText(('仅使用原始（斜）矢状位建模，斜轴位和斜冠状位不参与。' if source_ok else '当前采集不是矢状位，仅供浏览；请选择原始（斜）矢状序列后建模。')+f'层厚 {meta.get("thickness_mm","—")} mm，采集层中心间距 {self.session.volume.spacing_zyx[0]:.3f} mm。')
        self.mri_bone_button.setEnabled(source_ok)
        self.ai_button.setEnabled(source_ok)
        self.export_button.setEnabled(source_ok and not self.job);self.export_side_button.setEnabled(source_ok and not self.job)
        acl=next((s for s in self.session.segments if s.structure=='acl'),None)
        self.mri_acl_absent.blockSignals(True);self.mri_acl_absent.setChecked(bool(acl and acl.presence=='absent'));self.mri_acl_absent.blockSignals(False)
        self.mri_presence.blockSignals(True)
        self.mri_presence.setCurrentIndex(self.mri_presence.findData(bone.presence if bone and bone.kind=='ligament' else 'unknown'))
        self.mri_presence.blockSignals(False);self.mri_presence.setEnabled(bool(bone and bone.kind=='ligament'))
        self.mri_prompt_button.setEnabled(bool(source_ok and bone and bone.presence!='absent'))
        from .mr_correction import stored_points,correction_model_ready
        points=stored_points(self.session,self.active,self.cross[0])
        positive=sum(p['positive'] for p in points);negative=len(points)-positive
        self.mri_correct_button.setEnabled(bool(source_ok and bone and bone.presence!='absent' and correction_model_ready()))
        self.mri_review_button.setEnabled(source_ok)
        self.mri_clear_points.setEnabled(bool(points))
        self.mri_correction_status.setText(f'本层提示：保留 {positive} 点 / 排除 {negative} 点。'+('局部模型已就绪。' if correction_model_ready() else '请先安装局部纠错模型。')+' 使用上方“保留点 + / 排除点 −”标记，再运行局部纠错。')
        self.mri_refiner.blockSignals(True)
        self.mri_refiner.setCurrentIndex(self.mri_refiner.findData(self.session.parameters.get('mr_refinement_model','litemedsam')))
        self.mri_refiner.blockSignals(False)
        self.fill_button.setEnabled(bool(source_ok and bone and bone.kind=='bone'))
        self.fill_button.setToolTip('仅填充原始矢状采集层的封闭骨内空白；不以其他方向补洞，不覆盖其他结构。')
        state='先选择骨或韧带标签。'
        if bone:
            state=bone.name+'：'+({'unknown':'待评估，可框选或手工标注。','present':'用户已确认可见。','absent':'用户已标记缺失，模型和画笔不会生成该结构。'}[bone.presence])
            if bone.kind=='ligament' and bone.presence!='absent':
                layers=int(np.any(self.session.labels==bone.id,axis=(1,2)).sum())
                state+=f' 当前标注覆盖 {layers} 个原始采集层。'
                if layers<=1:state+=' 当前为未完成的局部标注，不能视为完整韧带。'
                else:state+=' 覆盖层数不代表标注已完整，请核对边界和邻层。'
        self.mri_state.setText(state)
        self.recognition_status.setText('MRI 骨结构与 ACL / PCL 分别标注；模型候选须对照原片复核。')
        if hasattr(self.surface,'caption'):self.surface.caption.setText('三维结构模型')

    def prepare_mri_labels(self):
        if not self.session or not self.session.volume.is_mr or self.job:return
        from .mr_tools import ensure_mr_segments
        ids=ensure_mr_segments(self.session)
        if not self.active:self.active=ids['femur']
        self.after_edit();self.status.setText('MRI 结构标签已就绪；选择标签后在原片框选或用画笔标注。')

    def acl_absence_changed(self,checked):
        if self._updating or not self.session or not self.session.volume.is_mr or self.job:return
        from .mr_tools import ensure_mr_segments
        ids=ensure_mr_segments(self.session)
        self.session.update_presence(ids['acl'],'absent' if checked else 'unknown')
        self.after_edit();self.refresh_surface_if_any()

    def mri_presence_changed(self,*_):
        if self._updating or not self.session or not self.session.volume.is_mr or self.job:return
        label=self.session.segment(self.active)
        if label and label.kind=='ligament':
            self.session.update_presence(label.id,self.mri_presence.currentData())
            self.after_edit();self.refresh_surface_if_any()

    def refresh_surface_if_any(self,reset=False):
        if self.session and any(s.visible and self.session.counts()[s.id]>0 for s in self.session.segments):self.refresh_surface(reset_camera=reset)
        else:self.surface.set_meshes([])

    def mri_prompt(self,plane,start,end):
        if not self.session or not self.session.volume.is_mr or self.job:return
        try:require_mri_modeling_source(self.session,plane)
        except ValueError as exc:self.status.setText(str(exc));return
        label=self.session.segment(self.active)
        if not label or label.presence=='absent':
            self.status.setText('请先选择待标注结构；已缺失标签不能生成掩膜。');return
        if not self.memory_check('model'):return
        from .mr_tools import predict_mr_prompt
        ident=self.active;span=self.mri_span.value()
        self.run_job('MRI 模型框选标注',lambda p:predict_mr_prompt(self.session,ident,plane,start,end,p,span=span),self.mri_prompt_finished)

    def mri_prompt_finished(self,result):
        local,report=result
        report['prompt_source']='interactive_box'
        self.session.apply(self.session.roi,local,'MRI 模型框选标注')
        self.session.ai_reports.append(report);self.session.dirty=True
        self.after_edit();self.refresh_surface_if_any()
        self.status.setText(f'MRI 框选候选已处理：采用 {report.get("accepted_slices",0)} 层 / 未采用 {report.get("flagged_slices",0)} 层。请对照原片核对；可撤销。')

    def mri_refiner_changed(self,*_):
        if self.session and self.session.volume.is_mr and not self._updating:
            self.session.parameters['mr_refinement_model']=self.mri_refiner.currentData();self.session.dirty=True

    def review_mri_masks(self):
        if not self.session or not self.session.volume.is_mr or self.job:return
        from .mr_review import review_native_masks
        self.run_job('检查原始矢状层轮廓与覆盖',lambda p:review_native_masks(self.session),self.mri_review_finished)

    def mri_review_finished(self,report):
        self.mri_review_flags.blockSignals(True);self.mri_review_flags.clear()
        self.mri_review_flags.addItem(f'待核对 {len(report["flags"])} 项（轮廓检查，不代表准确率）',None)
        for flag in report['flags']:
            self.mri_review_flags.addItem(f'{flag["label_name"]} · 第 {flag["slice_1based"]} 层 · {flag["message"]}',flag)
        self.mri_review_flags.blockSignals(False)
        self.session.repair_reports.append(report);self.session.dirty=True
        self.status.setText('轮廓与覆盖检查已记录。未提示异常也需要对照原片核对；空标注不代表结构缺失。')

    def jump_mri_review(self,*_):
        if self._updating or self.job or not self.session or not self.session.volume.is_mr:return
        flag=self.mri_review_flags.currentData()
        if not flag:return
        self.active=flag['label_id'];point=list(self.cross);point[0]=flag['slice_1based']-1
        self.navigate(point);self.refresh_list()

    def clear_mri_correction_points(self):
        if not self.session or not self.session.volume.is_mr or self.job:return
        from .mr_correction import stored_points
        self.session.parameters['mr_correction_points']=[p for p in stored_points(self.session) if not(p['label_id']==self.active and p['zyx'][0]==self.cross[0])]
        self.session.dirty=True;self.sync_mri_controls();self.panels['axial'].view.update()
        self.project_status.setText('提示已修改，尚未保存')

    def mri_correct_current(self):
        if not self.session or not self.session.volume.is_mr or self.job:return
        from .mr_correction import propose_correction,stored_points
        try:require_mri_modeling_source(self.session,'axial')
        except ValueError as exc:self.status.setText(str(exc));return
        if not self.memory_check('model'):return
        points=stored_points(self.session,self.active,self.cross[0]);ident=self.active;index=self.cross[0]
        positive=[p['zyx'] for p in points if p['positive']];negative=[p['zyx'] for p in points if not p['positive']]
        self.run_job('MRI 当前矢状层局部纠错',lambda p:propose_correction(self.session,ident,index,positive,negative),self.mri_correction_finished)

    def mri_correction_finished(self,result):
        local,report=result
        self.session.apply(self.session.roi,local,'MRI 正负提示局部纠错');self.session.ai_reports.append(report);self.session.dirty=True
        self.after_edit();self.refresh_surface_if_any()
        self.status.setText('局部纠错候选已应用；请对照原片核对，可撤销。' if report['accepted_slices'] else '本层未获得支持提示的候选，保留原标注；可调整保留点和排除点后重试。')

    def mri_bone_initial(self):
        if not self.session or not self.session.volume.is_mr or self.job or not self.memory_check('model'):return
        self.prepare_mri_labels()
        bone_ids=[s.id for s in self.session.segments if s.kind=='bone']
        if np.any(np.isin(self.session.labels[self.session.roi],bone_ids)):
            answer=QMessageBox.question(self,'MRI 骨初分割','重新生成当前范围的骨候选？韧带标签和范围外分割保留，可撤销。',QMessageBox.StandardButton.Ok|QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.Cancel)
            if answer!=QMessageBox.StandardButton.Ok:return
        from .mr_bone import prepare_mri_bones
        self.sync_parameters()
        self.run_job('MRI 骨结构模型',lambda p:prepare_mri_bones(self.session,p),self.mri_bones_finished)

    def mri_bones_finished(self,result):
        from .model import Segment
        local,report,segments=result
        for entry in segments:
            if self.session.segment(entry['id']) is None:self.session.segments.append(Segment(**entry))
            else:
                target=self.session.segment(entry['id'])
                target.recognition=entry.get('recognition',target.recognition)
        self.session.apply(self.session.roi,local,'MRI 骨结构模型')
        self.session.repair_reports.append(report);self.session.dirty=True
        self.after_edit();self.refresh_surface_if_any(reset=True)
        self.status.setText('MRI 骨结构候选已生成；相连骨、小骨块与韧带仍需逐层复核。')

    def processing_ids(self):
        if not self.session:return []
        key=(id(self.session),self.session.revision,self.session.history.version,
             tuple((s.start,s.stop) for s in self.session.roi))
        if key!=self._processing_counts_key:
            self._processing_counts=(self.session.counts() if self.session.roi==self.session.volume.full_box else
                count_labels(self.session.labels[self.session.roi],max((s.id for s in self.session.segments),default=0)+1))
            self._processing_counts_key=key
        counts=self._processing_counts
        if len(counts)<=max((s.id for s in self.session.segments),default=0):
            counts=np.pad(counts,(0,max(s.id for s in self.session.segments)+1-len(counts)))
        return [s.id for s in self.session.segments if counts[s.id]>0 and
                (s.visible if self.ai_all.isChecked() else s.id==self.active)]

    def refresh_processing_scope(self,*_):
        if not hasattr(self,'ai_button'):return
        ids=self.processing_ids()
        mr=bool(self.session and self.session.volume.is_mr)
        unit='个结构' if mr else '块骨'
        if self.ai_all.isChecked():text=f'本次处理：{len(ids)} '+('个勾选结构' if mr else '块勾选骨')+'（方框可多选）'
        else:
            bone=self.session.segment(self.active) if self.session else None
            text='本次仅处理当前骨：'+(bone.name if bone and ids else '未选中非空骨')
        self.ai_scope_status.setText(text)
        self.ai_button.setText(f'本地模型精修（{len(ids)} {unit}）')

    def set_bone_checks(self,checked):
        if not self.session or self.job:return
        for bone in self.session.segments:bone.visible=bool(checked)
        self.session.dirty=True;self.refresh_list()
        self.surface.visibility(self.session.segments)
        for panel in self.panels.values():panel.view.invalidate()
        self.project_status.setText('有未保存修改')

    def refresh_ai_status(self):
        if not self.session:return
        report=next((r for r in reversed(self.session.ai_reports) if r.get('label_id')==self.active),None)
        self.ai_flags.blockSignals(True);self.ai_flags.clear()
        self.ai_flags.addItem('异常层：模型未采用，可跳转核对',None)
        if report:
            self.ai_status.setText(f'{report["device"].upper()} · 模型通过 {report["accepted_slices"]} 层 / 未采用 {report["flagged_slices"]} 层')
            for entry in report['slices']:
                if not entry['accepted']:
                    axis=entry.get('fixed_axis',report.get('fixed_axis',0))
                    value={'axis':axis,'index':entry['slice_1based']} if self.session.volume.is_mr else entry['slice_1based']
                    translations={'pcl_signal_or_shape_uncertain':'PCL 信号或形态支持不足','touches_bbox':'候选触及框边缘',
                        'no_component_overlaps_prior_or_prompt_centre':'未找到目标候选','overlarge_prediction':'候选范围过大',
                        'low_overlap_with_prior':'与原掩膜重叠不足','area_ratio_outside_0.5_to_2':'轮廓变化过大',
                        'foreground_prompt_not_supported':'未找到支持保留点的候选',
                        'some_foreground_prompts_not_supported':'部分保留点未得到支持',
                        'large_model_change_needs_review':'变化过大，需人工核对',
                        'unprompted_existing_fragment_needs_review':'已有分离骨块可能遗漏，请放保留或排除点核对',
                        'no_valid_signal_for_model':'局部没有有效信号'}
                    flags=[translations.get(f,f) for f in entry['flags']]
                    self.ai_flags.addItem(f'{"采集 / 重排" if self.session.volume.is_mr else "轴位"} {entry["slice_1based"]} · {"、".join(flags)}',value)
        else:self.ai_status.setText('LiteMedSAM · 当前骨尚无精修记录')
        self.ai_flags.blockSignals(False)

    def jump_ai_flag(self):
        if not self.session:return
        index=self.ai_flags.currentData()
        if index is None:return
        point=list(self.cross)
        if isinstance(index,dict):point[index['axis']]=int(index['index'])-1
        else:
            z=int(index)-1;point[0]=z
            ys,xs=np.where(self.session.labels[z]==self.active)
            if len(ys):point[1]=int((ys.min()+ys.max())//2);point[2]=int((xs.min()+xs.max())//2)
        self.navigate(point)

    def segment_item_changed(self,item):
        if self._updating or not self.session:return
        seg=self.session.segment(item.data(Qt.ItemDataRole.UserRole));name=item.text().strip()
        if not name:name=f'骨块 {seg.id:02d}'
        if name!=seg.name:
            self.session.update_bone_names([{'id':seg.id,'name':name,'reviewed':True}], '手工命名',preserve_reviewed=False)
            self.undo_button.setEnabled(bool(self.session.history.undo_stack));self.redo_button.setEnabled(bool(self.session.history.redo_stack))
        seg.visible=item.checkState()==Qt.CheckState.Checked;self.session.dirty=True
        self.project_status.setText('有未保存修改')
        self.surface.visibility(self.session.segments)
        for panel in self.panels.values():panel.view.invalidate()
        self.refresh_processing_scope()

    def roi_from_spins(self):
        if self._updating or not self.session:return
        box=tuple(slice(a.value()-1,b.value()) for a,b in self.roi_spins)
        try:self.session.set_roi(box)
        except ValueError:
            self.sync_roi_spins();self.status.setText('范围起始不能大于终止，已恢复原范围。');return
        self.after_roi()

    def sync_roi_spins(self):
        self._updating=True
        for (a,b),s,n in zip(self.roi_spins,self.session.roi,self.session.volume.hu.shape):
            a.setMaximum(n);b.setMaximum(n);a.setValue(s.start+1);b.setValue(s.stop)
        self._updating=False

    def roi_from_view(self,box):
        if not self.session or self.job:return
        self.session.set_roi(box);self.sync_roi_spins();self.after_roi()

    def after_roi(self):
        voxels=int(np.prod([s.stop-s.start for s in self.session.roi]))
        self.status.setText(f'已选范围：{voxels/1e6:.1f} 百万体素 · 可在三平面继续调整')
        if self.session.joint=='leg':self.refresh_memory_status()
        self.refresh_processing_scope()
        for p in self.panels.values():p.view.update()

    def reset_roi(self):
        if self.session:self.roi_from_view(self.session.volume.full_box)

    def suggest_roi(self):
        if not self.session:return
        box=list(self.session.volume.full_box);side=self.side.currentText()
        if side!='双侧':
            # This is explicitly a half-FOV suggestion, not an inferred patient midline.
            affine=self.session.volume.affine_lps
            directions=affine[:3,:3]/np.linalg.norm(affine[:3,:3],axis=0)
            xyz_axis=int(np.argmax(np.abs(directions[0,:2])));axis=2-xyz_axis
            n=self.session.volume.hu.shape[axis];middle=n//2
            positive=affine[0,xyz_axis]>0
            upper=(side=='左侧')==positive
            box[axis]=slice(middle,n) if upper else slice(0,middle)
        self.roi_from_view(tuple(box));self.status.setText('已按侧别建议范围。请核对实际中线，单侧扫描可选“完整视野”。')

    def navigate(self,point):
        if not self.session:return
        self.cross=[int(np.clip(i,0,n-1)) for i,n in zip(point,self.session.volume.hu.shape)]
        for p in self.panels.values():
            p.view.cross=list(self.cross);p.view.invalidate()
            axis=p.view.fixed_axis;p.slider.blockSignals(True);p.slider.setValue(self.cross[axis]);p.slider.blockSignals(False)
            p.counter.setText(f'{self.cross[axis]+1} / {self.session.volume.hu.shape[axis]}')
        if self.session.volume.is_mr:self.sync_mri_controls()

    def slide_changed(self,plane,value):
        if not self.session:return
        point=list(self.cross);point[self.panels[plane].view.fixed_axis]=value;self.navigate(point)

    def scroll_slice(self,plane,amount):
        if self.session:self.slide_changed(plane,self.cross[self.panels[plane].view.fixed_axis]+amount)

    def sync_tool_from_buttons(self):
        tool=next((key for key,button in self.tool_buttons.items() if button.isChecked()),self.current_tool)
        self.set_tool(tool)

    def check_tool_button(self,tool):
        # Suppress recursive group notifications while changing the selected
        # button; all three views and the context are updated together below.
        blockers=[QSignalBlocker(button) for button in self.tool_buttons.values()]
        self.tool_buttons[tool].setChecked(True)
        del blockers

    def set_tool(self,tool):
        if self.job:
            self.check_tool_button(self.current_tool);return
        if self.session and self.session.volume.is_mr and tool in ('paint','erase','assign','seed','model_box','mr_positive','mr_negative'):
            try:require_mri_modeling_source(self.session)
            except ValueError as exc:
                self.check_tool_button(self.current_tool);self.status.setText(str(exc));return
        if tool in ('model_box','mr_positive','mr_negative') and not (self.session and self.session.volume.is_mr):
            self.check_tool_button(self.current_tool);self.status.setText('MRI 结构框选工具需先导入 MRI。');return
        if self._stroke:self.end_stroke()
        self.check_tool_button(tool);self.current_tool=tool
        for p in self.panels.values():p.view.tool=tool;p.view.update()
        self.update_tool_context()

    def update_tool_context(self):
        bone=self.session.segment(self.active) if self.session else None
        name=bone.name if bone else '未选择标签'
        self.edit_target.setText('当前编辑：'+name)
        tool=self.current_tool
        messages={'browse':'单击联动切片；滚轮切层，Ctrl+滚轮缩放，右键拖动平移。',
                  'roi':'在任一切片拖出黄色范围；另两个方向同步更新。',
                  'paint':'按住左键补画当前层；松开后滚轮切层。仅在黄色范围内修改。',
                  'erase':'按住左键擦除当前标签；松开后滚轮切层。其他标签保留。',
                  'pick':'点击已有骨块，选中它的标签。',
                  'assign':'先选目标骨标签，再点击骨块；整块归入目标骨。',
                  'seed':'先选目标骨，在骨内部点击放种子；不同骨分别放置后运行种子分离。',
                  'model_box':'先选结构标签，在原始（斜）矢状采集切片拖框；模型仅处理该方向的当前层及明确设置的邻层，青框不改变 ROI。',
                  'mr_positive':'在目标骨或韧带内部放保留点，再点击“MRI 局部模型纠错”；提示仅属于当前结构和原始矢状层。',
                  'mr_negative':'在当前结构误标的软组织上放排除点，再运行局部模型纠错；其他结构不会被擦除。'}
        text=self.tool_buttons[tool].text()+' · '+name+'｜'+messages[tool]
        self.context.setText(text);self.context.setToolTip(text)

    def display_settings(self):
        for p in self.panels.values():
            p.view.window=self.ww.value();p.view.level=self.wc.value();p.view.opacity=self.opacity.value()/100
            p.view.brush_mm=self.brush.value();p.view.active=self.active;p.view.invalidate()

    def begin_stroke(self,plane,index):
        if self.job or not self.session or not self.active:return
        try:require_mri_modeling_source(self.session,plane)
        except ValueError as exc:self.status.setText(str(exc));return
        label=self.session.segment(self.active)
        if label and label.presence=='absent':self.status.setText('该结构已标记缺失，请先修改存在状态。');return
        box=plane_box(self.session.labels.shape,plane,index)
        self._stroke=(box,self.session.labels[box].copy())

    def stroke_point(self,plane,point):
        if self._stroke is None:return
        paint(self.session,plane,point,self.brush.value(),self.active,self.panels[plane].view.tool=='erase')
        for p in self.panels.values():p.view.invalidate()

    def end_stroke(self):
        if self._stroke is None:return
        box,before=self._stroke;self._stroke=None
        if self.session.record_preapplied(box,before,'画笔 / 橡皮'):
            self.after_edit()

    def point_action(self,tool,point,plane='axial'):
        if not self.session or self.job:return
        if tool in ('mr_positive','mr_negative'):
            from .mr_correction import stored_points
            try:
                if not self.session.volume.is_mr:raise ValueError('MRI 纠错提示仅支持 MR。')
                require_mri_modeling_source(self.session,plane)
                label=self.session.segment(self.active)
                if not label or label.presence=='absent':raise ValueError('请先选择可标注的结构；缺失标签不能放置保留点。')
                if not inside(point,self.session.roi):raise ValueError('请在当前黄色范围内放置提示。')
                if tool=='mr_positive' and self.session.labels[tuple(point)] not in (0,self.active):raise ValueError('该位置属于其他标签，请先核对归属。')
            except ValueError as exc:self.status.setText(str(exc));return
            entries=[p for p in stored_points(self.session) if not(p['label_id']==self.active and p['zyx']==list(point))]
            if len(entries)>=1024:self.status.setText('提示点过多，请先清除不需要的提示。');return
            entries.append({'label_id':int(self.active),'zyx':list(map(int,point)),'positive':tool=='mr_positive'})
            self.session.parameters['mr_correction_points']=entries;self.session.dirty=True
            self.project_status.setText('提示已修改，尚未保存');self.sync_mri_controls();self.panels['axial'].view.update();return
        if tool in ('seed','assign'):
            try:require_mri_modeling_source(self.session,plane)
            except ValueError as exc:self.status.setText(str(exc));return
        ident=int(self.session.labels[point])
        if tool=='pick':
            if ident:self.active=ident;self.refresh_list()
            return
        if not self.active:self.status.setText('先选中或新建一个目标骨标签。');return
        if self.session.segment(self.active).presence=='absent':self.status.setText('该结构已标记缺失，请先修改存在状态。');return
        if tool=='seed':
            if not inside(point,self.session.roi) or ident==0:self.status.setText('种子需落在黄色范围内的已分割骨块中。');return
            self.session.seeds=[s for s in self.session.seeds if tuple(s.zyx)!=tuple(point)]
            self.session.seeds.append(Seed(tuple(map(int,point)),self.active));self.session.dirty=True
            self.status.setText(f'已放置 {len(self.session.seeds)} 个种子。不同骨需选择不同标签。')
            for p in self.panels.values():p.view.update()
        elif tool=='assign':
            scope=plane if self.assign_2d.isChecked() else None
            title='二维归属' if scope else '三维归属'
            self.run_job(title,lambda p:assign_component(self.session,point,self.active,scope),lambda result:self.apply_local(result,title))

    def clear_seeds(self):
        if self.session:
            self.session.seeds.clear();self.session.dirty=True
            for p in self.panels.values():p.view.update()
            self.status.setText('种子已清除，骨分割未改变。')

    def refresh_memory_status(self,memory=None):
        memory=memory if memory is not None else detect_memory()
        policy=memory_policy(memory);self.current_memory_policy=policy
        available=policy['memory_available_bytes'];ops=policy['operations']
        text=f'内存 {policy["memory_total_bytes"]/1024**3:.0f} GB'
        if available is not None:text+=f' · 可用 {available/1024**3:.1f} GB'
        if not policy['memory_detected']:text+='（保守配置）'
        if self.session and self.session.volume.is_mr:
            text+=f'\nMRI 模型上限 {ops["model"]["limit_voxels"]/1e6:.1f} 百万体素 · 保留原采集网格'
        elif self.session and self.session.joint=='leg':
            from .long_scan import whole_leg_plan
            try:
                depth=whole_leg_plan(self.session.roi,policy)['slab_depth']
                text+=f'\n全腿初分割：每段最多 {depth} 层'
            except ValueError:
                text+='\n全腿分段：当前内存不足'
            text+=f'；局部精修上限 {ops["model"]["limit_voxels"]/1e6:.1f} 百万体素'
        else:
            text+=f'\n当前上限：初分割 {ops["initial"]["limit_voxels"]/1e6:.1f} / 精修 {ops["model"]["limit_voxels"]/1e6:.1f} 百万体素'
        self.memory_status.setText(text)
        self.memory_status.setToolTip('按总内存和当前可用内存估算；每次运行前重新检查。种子分离、CT 补全分别计算预算。关闭其他程序后重试可重新评估上限；内存估算不保证峰值。')
        return policy

    def memory_check(self,operation):
        policy=self.refresh_memory_status();limits=policy['operations'][operation]
        if operation=='initial' and self.session.joint=='leg':
            from .long_scan import whole_leg_plan
            try:whole_leg_plan(self.session.roi,policy)
            except ValueError as exc:self.show_error(str(exc));return False
            return True
        limit=limits['limit_voxels']
        n=int(np.prod([s.stop-s.start for s in self.session.roi]))
        if n>limit:
            title={'initial':'初分割','cleanup':'外部结构清理','model':'模型精修','split':'种子分离','recovery':'CT 骨缘补全'}[operation]
            text=f'当前范围 {n/1e6:.1f} 百万体素；{title}本次上限 {limit/1e6:.1f} 百万。\n系统内存 {policy["memory_total_bytes"]/1024**3:.1f} GB'
            if policy['memory_available_bytes'] is not None:text+=f'，当前可用 {policy["memory_available_bytes"]/1024**3:.1f} GB'
            text+=f'；按总内存计算的上限为 {limits["physical_limit_voxels"]/1e6:.1f} 百万体素。'
            text+='\n请缩小黄色范围，或释放内存后重试；程序会重新检测可用内存。'
            self.show_error(text);return False
        return True

    def auto_segment(self):
        if self.session and self.session.volume.is_mr:self.mri_bone_initial();return
        if not self.session or self.job or not self.memory_check('initial'):return
        self.sync_parameters()
        if np.any(self.session.labels[self.session.roi]):
            answer=QMessageBox.question(self,'重新初分割','替换黄色范围内的分割结果？范围外保留，可撤销此操作。',QMessageBox.StandardButton.Ok|QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.Cancel)
            if answer!=QMessageBox.StandardButton.Ok:return
        p=self.session.parameters.copy()
        self.run_job('自动初分割',lambda cb:prepare_initial(self.session,p,cb),self.auto_finished)

    def auto_finished(self,result):
        ids=install_initial(self.session,result['segmentation'],result['recognition'])
        if result['preprocessing']:self.session.preprocess_reports.append(result['preprocessing'])
        if result.get('repair'):
            report=result['repair']
            for row in report['bones']:
                row['id']=ids[row['id']-1];row['name']=self.session.segment(row['id']).name
            self.session.repair_reports.append(report);self.refresh_repair_status()
        self.refresh_cleanup_status()
        self.active=ids[0]
        named=sum(bool(p['name']) and p['status']!='check_connection' for p in result['recognition'])
        connected=sum(p['status']=='check_connection' for p in result['recognition'])
        self.after_edit()
        if self.session.joint=='leg':
            self.status.setText(f'全腿分段初分割完成：{len(ids)} 个候选骨块。请对照原 CT 复核和命名。')
        else:
            self.status.setText(f'已生成 {len(ids)} 个骨块，自动命名 {named} 块、疑似相连 {connected} 块。请核对侧别、关节面和骨名。')
        self.refresh_surface()

    def recognize_bones(self):
        if not self.session or self.job:return
        self.run_job('自动识别骨标签',lambda cb:recognize_existing(self.session,cb),self.recognition_finished)

    def recognition_finished(self,proposals):
        summary=install_recognition(self.session,proposals)
        self.refresh_list()
        text=f'自动命名 {summary["identified"]} 块 / 疑似相连 {summary["connection"]} 块 / 待识别 {summary["uncertain"]} 块 / 保留手工名称 {summary["manual_preserved"]} 块'
        self.recognition_status.setText(text);self.status.setText(text+'；自动名称可修改、可撤销。')
        if summary['changed']:self.project_status.setText('有未保存修改')

    def remove_table(self):
        if not self.session or self.job or not self.memory_check('cleanup'):return
        self.run_job('去床板、石膏和外部支具',lambda cb:remove_segmented_table(self.session,cb),self.table_removed)

    def table_removed(self,result):
        local,report=result
        changed=self.session.apply(self.session.roi,local,'去床板与外部附着物')
        if changed:
            counts=self.session.counts()
            if self.active>=len(counts) or not counts[self.active]:
                self.active=next((s.id for s in self.session.segments if s.id<len(counts) and counts[s.id]),0)
        self.session.preprocess_reports.append(report);self.session.dirty=True
        self.refresh_cleanup_status()
        self.after_edit()
        if changed:
            self.session.seeds=[seed for seed in self.session.seeds if self.session.labels[seed.zyx]>0]
            from scipy import ndimage as ndi
            boxes=ndi.find_objects(self.session.labels)
            if 0<self.active<=len(boxes) and boxes[self.active-1]:
                self.navigate([(b.start+b.stop-1)//2 for b in boxes[self.active-1]])
            self.status.setText(f'已排除床板 / 外部附着物候选 {report["removed_labeled_voxels"]} 个体素；可撤销。')
            self.refresh_surface(reset_camera=True)
        else:self.status.setText('当前范围没有可可靠去除的外部结构；原分割已保留。')
        self.project_status.setText('有未保存修改')

    def refresh_cleanup_status(self):
        reports=self.session.preprocess_reports if self.session else []
        if not reports:
            self.cleanup_status.setText('清理分割中的外部附着物；原始 CT 保留。');return
        report=reports[-1];ambiguous=report.get('ambiguous_attachment_components',0)
        count=report.get('removed_labeled_voxels',report.get('excluded_dense_voxels',0))
        text=f'上次清理：排除 {count} 个体素。'
        if ambiguous:text+=f' {ambiguous} 处结构贴近或连接骨，保留内部区域，请核对。'
        if report.get('preserved_uncertain_slices_1based'):text+=' 部分层轮廓不明确，已保留。'
        self.cleanup_status.setText(text)

    def split_seeds(self):
        if not self.session or self.job or not self.memory_check('split'):return
        self.run_job('种子分离',lambda p:split_from_seeds(self.session,p),lambda result:self.apply_local(result,'种子分离'))

    def ai_refine(self):
        if not self.session or self.job:return
        ids=self.processing_ids()
        if not ids:self.show_error('请在骨列表标有“选”的方框勾选非空骨，或关闭批量选项后选择当前骨。');return
        if not self.memory_check('model'):return
        from .ai import refine_many
        if self.session.volume.is_mr and self.mri_refiner.currentData()=='scribbleprompt':
            from .mr_correction import refine_with_correction as refine_many
        self.sync_parameters()
        self.run_job('本地医学模型精修',lambda p:refine_many(self.session,ids,p),self.ai_finished)

    def ai_finished(self,result):
        local,reports=result
        if not reports:self.status.setText('目标为空或已标记缺失，没有应用模型结果。');return
        self.session.apply(self.session.roi,local,'LiteMedSAM 模型精修')
        self.session.ai_reports.extend(reports);self.session.dirty=True
        if reports and reports[0].get('batch_ct_recovery'):
            self.session.repair_reports.append(reports[0]['batch_ct_recovery']);self.refresh_repair_status()
        self.after_edit()
        unit='个结构' if self.session.volume.is_mr else '块骨'
        text=f'{reports[0]["device"].upper()} · {len(reports)} {unit}，模型通过 {sum(r["accepted_slices"] for r in reports)} 层 / 未采用 {sum(r["flagged_slices"] for r in reports)} 层'
        self.ai_status.setText(text);self.status.setText(text+'；可撤销，保存项目会保留逐层检查记录。')
        self.refresh_surface()

    def recover_edges(self):
        if not self.session or self.job:return
        ids=self.processing_ids()
        if not ids:self.show_error('请在骨列表方框勾选非空骨，或关闭批量选项后选择当前骨。');return
        if not self.memory_check('recovery'):return
        self.sync_parameters()
        self.run_job('CT 骨缘补全',lambda cb:recover_ct_edges(self.session,ids,cb),self.edges_recovered)

    def edges_recovered(self,result):
        local,report=result
        changed=self.session.apply(self.session.roi,local,'CT 骨缘补全')
        self.session.repair_reports.append(report);self.session.dirty=True
        self.after_edit();self.refresh_repair_status()
        if changed:self.refresh_surface()
        else:self.status.setText('本次没有可可靠补全的骨缘；现有分割已保留。')

    def refresh_repair_status(self):
        if self.session and self.session.volume.is_mr:
            self.repair_status.setText('MRI 使用结构模型或框选；不执行 CT HU 补全。');return
        reports=self.session.repair_reports if self.session else []
        if not reports:self.repair_status.setText('CT 骨缘补全保留现有分割，可撤销。');return
        report=reports[-1];text=f'上次 CT 骨缘补全：增加 {report["added_voxels"]} 个体素。'
        flagged=sum(bool(row.get('flags')) for row in report.get('bones',[]))
        if flagged or report.get('flags'):text+=' 部分区域需对照 CT 核对。'
        self.repair_status.setText(text)

    def refresh_filter_status(self):
        if not self.session:return
        requested=self.smoothing.value();limit=float(min(self.session.volume.spacing_zyx)*.5)
        actual=min(requested,limit) if self.detail_guard.isChecked() else requested
        self.filter_status.setText(f'实际体素滤波 σ：{actual:.3f} mm'+(' · 骨缘保护' if self.detail_guard.isChecked() else ''))

    def backend_changed(self):
        if self.session and not self._updating:
            self.session.parameters['compute_backend']=self.backend.currentData();self.session.dirty=True
            self.surface.state.setText('后端已修改 · 更新 3D')

    def cleanup(self,operation):
        if not self.session or not self.active or self.job:return
        titles={'fill':'当前骨髓腔填充','largest':'保留最大块','clear':'清空当前骨'}
        def finished(result):
            changed=self.apply_local(result,titles[operation])
            if changed and operation=='fill':self.refresh_surface()
        self.run_job(titles[operation],lambda p:clean_segment(self.session,self.active,operation,p),finished)

    def apply_local(self,result,title):
        if result is not None and self.session.apply(self.session.roi,result,title):
            self.after_edit();return True
        return False

    def after_edit(self):
        self.refresh_list();self.surface.state.setText('分割已修改 · 更新 3D');self.project_status.setText('有未保存修改')
        self.surface.visibility(self.session.segments)
        for p in self.panels.values():p.view.invalidate()

    def undo(self,redo=False):
        if not self.session or self.job:return
        if self._stroke:self.end_stroke()
        stack=self.session.history.redo_stack if redo else self.session.history.undo_stack
        removed=stack[-1].removed_segment if stack else None
        title=self.session.travel_history(redo)
        if title and removed and not redo:self.active=removed['segment']['id']
        if title:self.after_edit();self.status.setText(('已重做：' if redo else '已撤销：')+title)

    def surface_parameter_changed(self):
        if self.session and not self._updating:
            self.session.parameters['surface_mm']=self.smoothing.value();self.session.dirty=True
            self.session.parameters['mesh_smoothing']=self.mesh_smoothing.currentData()
            self.session.parameters['preserve_detail']=self.detail_guard.isChecked()
            self.surface.state.setText('平滑已修改 · 更新 3D')
        self.refresh_filter_status()

    def refresh_surface(self,reset_camera=False):
        if not self.session or self.job or not self.session.segments:return
        self.sync_parameters()
        reset=reset_camera or not bool(self.surface.actors)
        self.run_job('更新三维预览',lambda p:segment_meshes(self.session,preview=True,progress=p),lambda meshes:self.surface_ready(meshes,reset))

    def surface_ready(self,meshes,reset):
        self.surface.set_meshes(meshes,reset);self.status.setText('精细三维表面已更新；导出可复用相同网格。左键旋转，右键 / 滚轮缩放。')

    def export(self):
        if not self.session or self.job:return
        options_dialog=StlExportDialog(self.session,self.active,self)
        if options_dialog.exec()!=QDialog.DialogCode.Accepted:return
        options=options_dialog.export_options()
        directory=QFileDialog.getExistingDirectory(self,'选择 STL 输出位置（自动建立新批次文件夹）',self.settings.value('last_export',str(Path.home()/'Documents')))
        if not directory:return
        self.settings.setValue('last_export',directory);self.sync_parameters()
        self.run_job('导出 STL',lambda p:export_stl(self.session,directory,p,**options),self.export_finished)

    def export_finished(self,result):
        directory,manifest=result;self.last_export=directory
        files=manifest.get('files',manifest['segments'])
        bad=[s['filename'] for s in files if not(s['watertight'] and s['winding_consistent'] and s['positive_volume'])]
        noun='结构标签' if self.session and self.session.volume.is_mr else '骨标签'
        self.status.setText(f'已导出 {len(files)} 个 STL，包含 {len(manifest["segments"])} 个{noun}：{directory}')
        message=f'已导出 {len(files)} 个 STL，包含 {len(manifest["segments"])} 个{noun}，导入单位选择 mm。\n{directory}'
        if bad:message+='\n以下网格需要复核：'+', '.join(bad)
        else:message+='\n导出回读、网格闭合与绕向检查通过。'
        box=QMessageBox(self);box.setWindowTitle('导出完成');box.setText(message)
        open_button=box.addButton('打开文件夹',QMessageBox.ButtonRole.ActionRole);box.addButton('完成',QMessageBox.ButtonRole.AcceptRole);box.exec()
        if box.clickedButton()==open_button:QDesktopServices.openUrl(QUrl.fromLocalFile(directory))

    def closeEvent(self,event):
        if self.job:
            self.pending_close=True;self.cancel_job();event.ignore();return
        if self.session and self.session.dirty:
            answer=QMessageBox.question(self,'退出前保存','当前项目有未保存修改。是否先保存？',QMessageBox.StandardButton.Save|QMessageBox.StandardButton.Discard|QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.Save)
            if answer==QMessageBox.StandardButton.Save:
                self.pending_close=True;self.save()
                if self.job is None:self.pending_close=False
                event.ignore();return
            if answer!=QMessageBox.StandardButton.Discard:event.ignore();return
        self.surface.vtk.Finalize();event.accept()
