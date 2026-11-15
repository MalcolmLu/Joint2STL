"""Bone rows backed by real widgets, avoiding Cocoa's synthesized item-view rows."""
from __future__ import annotations

from PySide6.QtCore import Qt,Signal
from PySide6.QtGui import QColor,QIcon
from PySide6.QtWidgets import (QButtonGroup,QCheckBox,QDialog,QDialogButtonBox,QFrame,QGridLayout,QHBoxLayout,QLabel,QLayout,
                              QLineEdit,QPushButton,QRadioButton,QScrollArea,QSpinBox,QDoubleSpinBox,
                              QSizePolicy,QVBoxLayout,QWidget)


class PrecisionSpinBox(QSpinBox):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self,event):
        event.ignore()


class PrecisionDoubleSpinBox(QDoubleSpinBox):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self,event):
        event.ignore()


class MenuSelector(QPushButton):
    """Accessible choices backed by actual buttons rather than virtual rows."""
    currentIndexChanged=Signal(int)
    currentTextChanged=Signal(str)

    def __init__(self):
        super().__init__();self._items=[];self._index=-1
        self._dialog=None
        self.clicked.connect(self._open_menu)
        self.setAccessibleDescription('打开选项列表')
        self.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(90)
    def addItem(self,text,userData=None):
        self._items.append((text,userData))
        if self._index<0:self.setCurrentIndex(0)
    def addItems(self,texts):
        for text in texts:self.addItem(text)
    def _open_menu(self):
        dialog=QDialog(self.window());dialog.setWindowTitle('选择选项');dialog.resize(430,min(520,120+len(self._items)*42))
        layout=QVBoxLayout(dialog);layout.addWidget(QLabel('当前：'+self.currentText()))
        scroll=QScrollArea();scroll.setWidgetResizable(True);body=QWidget();body.setObjectName('boneRows');scroll.setWidget(body)
        rows=QVBoxLayout(body)
        def choose(index):
            self.setCurrentIndex(index);dialog.accept()
        for i,(text,_) in enumerate(self._items):
            option=QPushButton(text);option.setAccessibleName(text)
            option.clicked.connect(lambda checked=False,index=i:choose(index));rows.addWidget(option)
        rows.addStretch();layout.addWidget(scroll)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
        self._dialog=dialog
        dialog.exec();self._dialog=None;dialog.deleteLater()
    def clear(self):
        old=self._index;self._items=[];self._index=-1;self._sync_label()
        if old!=-1:self.currentIndexChanged.emit(-1);self.currentTextChanged.emit('')
    def count(self):return len(self._items)
    def currentIndex(self):return self._index
    def currentText(self):return self._items[self._index][0] if self._index>=0 else ''
    def currentData(self):return self._items[self._index][1] if self._index>=0 else None
    def findData(self,value):return next((i for i,(_,data) in enumerate(self._items) if data==value),-1)
    def findText(self,text):return next((i for i,(name,_) in enumerate(self._items) if name==text),-1)
    def setCurrentText(self,text):
        index=self.findText(text)
        if index>=0:self.setCurrentIndex(index)
    def setCurrentIndex(self,index):
        if not -1<=index<len(self._items) or self._index==index:return
        self._index=index
        self._sync_label();self.currentIndexChanged.emit(index);self.currentTextChanged.emit(self.currentText())
    def _sync_label(self):
        text=self.currentText();self.setAccessibleName(text);self.setToolTip(text)
        self.setText(self.fontMetrics().elidedText(text,Qt.TextElideMode.ElideRight,max(30,self.width()-30))+' ▾')
    def resizeEvent(self,event):
        super().resizeEvent(event);self._sync_label()


class SegmentItem:
    """A persistent row with separate active, visibility, color and name controls."""
    def __init__(self,text):
        self.owner=None;self._data={};self._icon=QIcon()
        self._flags=Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable
        self.row=QFrame();self.row.setObjectName('boneRow')
        self.row.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        layout=QGridLayout(self.row);layout.setContentsMargins(6,6,6,6)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        layout.setHorizontalSpacing(8);layout.setVerticalSpacing(1);layout.setColumnStretch(3,1)
        self.active=QRadioButton();self.active.setToolTip('单选当前编辑骨；批量精修由方框勾选控制')
        self.visible=QCheckBox('选');self.visible.setToolTip('可同时勾选多块骨，用于显示、批量精修 / 补全和勾选骨导出')
        for control in (self.active,self.visible):
            # QMacStyle otherwise subtracts native layout margins from these
            # custom indicators, letting the checkbox text overlap the swatch.
            control.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect,True)
            control.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Fixed)
        self.swatch=QLabel();self.swatch.setFixedSize(12,12)
        self.name=QLineEdit(text);self.name.setObjectName('boneName')
        self.name.setMinimumWidth(0);self.name.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        self.name.setCursorPosition(0)
        self.annotation_state=QLabel();self.annotation_state.setObjectName('muted');self.annotation_state.hide()
        for column,control in enumerate((self.active,self.visible,self.swatch,self.name)):
            layout.addWidget(control,0,column,Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.annotation_state,1,3)
        self.active.toggled.connect(self._activated)
        self.visible.stateChanged.connect(self._notify)
        self.name.editingFinished.connect(self._notify)
        self._accessible_names()
        self._activated(False)

    def _accessible_names(self):
        self.active.setAccessibleName('编辑骨：'+self.text())
        self.visible.setAccessibleName('多选骨（显示 / 精修 / 导出）：'+self.text())
        self.name.setAccessibleName('骨名称：'+self.text())
        self.name.setToolTip(self.text())

    def _activated(self,checked):
        self.row.setStyleSheet('QFrame#boneRow { background: '+('#244b58' if checked else '#101a28')+'; border-radius:4px; }')
        if checked and self.owner:self.owner._activate(self)

    def _notify(self,*_):
        self._accessible_names()
        if self.owner:self.owner.itemChanged.emit(self)

    def data(self,role):return self._data.get(role)
    def setData(self,role,value):self._data[role]=value
    def flags(self):return self._flags
    def setFlags(self,flags):self._flags=flags
    def text(self):return self.name.text()
    def setText(self,text):self.name.setText(text);self.name.setCursorPosition(0);self._accessible_names()
    def checkState(self):return self.visible.checkState()
    def setCheckState(self,state):self.visible.setCheckState(state)
    def icon(self):return self._icon
    def setIcon(self,icon):self._icon=icon;self.swatch.setPixmap(icon.pixmap(12,12))
    def setToolTip(self,text):self.row.setToolTip(text)
    def setForeground(self,color):self.name.setStyleSheet('color:'+QColor(color).name()+';')


class SegmentList(QFrame):
    currentItemChanged=Signal(object,object)
    itemChanged=Signal(object)

    def __init__(self):
        super().__init__();self._items=[];self._current=None
        self.group=QButtonGroup(self);self.group.setExclusive(True)
        layout=QVBoxLayout(self);layout.setContentsMargins(1,1,1,1)
        self.scroll=QScrollArea();self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff);layout.addWidget(self.scroll)
        body=QWidget();body.setObjectName('boneRows');self.scroll.setWidget(body)
        self.rows=QVBoxLayout(body);self.rows.setContentsMargins(1,1,1,1);self.rows.setSpacing(2)
        self.rows.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize);self.rows.addStretch()

    def count(self):return len(self._items)
    def item(self,index):return self._items[index]
    def currentItem(self):return self._current
    def addItem(self,item):
        item.owner=self;self._items.append(item);self.group.addButton(item.active)
        self.rows.insertWidget(self.rows.count()-1,item.row)
    def takeItem(self,index):
        item=self._items.pop(index);self.group.removeButton(item.active);self.rows.removeWidget(item.row)
        if self._current is item:self._current=None
        item.owner=None;item.row.deleteLater()
        return item
    def clear(self):
        while self._items:self.takeItem(len(self._items)-1)
    def setCurrentItem(self,item):item.active.setChecked(True)
    def _activate(self,item):
        previous=self._current;self._current=item
        if previous is not item:self.currentItemChanged.emit(item,previous)
