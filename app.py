# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import pyedflib
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QFileDialog, QMainWindow, QWidget
from random import randint
from scipy import signal
import matplotlib.pyplot as plt

class Ui_MainWindow(QMainWindow):
    def openSecondDialog(self,arr,no,title):
        mydialog = QtWidgets.QMdiSubWindow(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("sig.png"),
                       QtGui.QIcon.Normal, QtGui.QIcon.Off)
        mydialog.setWindowIcon(icon)
        mydialog.setWindowTitle( str(no+1)+'#' + title)
        mydialog.graphWidget = pg.PlotWidget()
        mydialog.setWidget(mydialog.graphWidget)
        mydialog.graphWidget.setBackground('w')
        mydialog.graphWidget.plot(arr, pen='b')
        mydialog.graphWidget.showGrid(x=True, y=True)
        mydialog.graphWidget.setXRange(0, 1000, padding=0)
        mydialog.graphWidget.setYRange(-150, 150, padding=0)
        self.mdi.addSubWindow(mydialog)
        mydialog.show()

    def read_file(self, filename):
        file_name = pyedflib.data.get_generator_filename()
        f = pyedflib.EdfReader(filename)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i)
        f, t, Sxx = signal.spectrogram(sigbufs[1], fs=200)
        plt.pcolormesh(t, f, 10*np.log10(Sxx))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()
        plt.show()
        for i in range(0,5):
            print(i)
            self.graphWidget = pg.PlotWidget()
            self.openSecondDialog(sigbufs[i],i,signal_labels[i])

        self.mdi.tileSubWindows()
    def browsefiles(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '../')
        self.read_file(fname[0])

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("sig.png"),
                       QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # self.mdiArea = QtWidgets.QMdiArea(self.centralwidget)
        # # sizePolicy = QtWidgets.QSizePolicy(
        # #     QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        # # sizePolicy.setHorizontalStretch(1000)
        # # sizePolicy.setVerticalStretch(1000)
        # # sizePolicy.setHeightForWidth(
        # #     self.mdiArea.sizePolicy().hasHeightForWidth())
        # # self.mdiArea.setSizePolicy(sizePolicy)
        # self.mdiArea.setStyleSheet("background-color: rgb(255, 255, 255);")
        # self.mdiArea.setFrameShape(QtWidgets.QFrame.NoFrame)
        # # self.mdiArea.setSizeAdjustPolicy(
        # #     QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        # # self.mdiArea.setActivationOrder(
        # #     QtWidgets.QMdiArea.ActivationHistoryOrder)
        # self.mdiArea.setObjectName("mdiArea")
        self.mdi = QtWidgets.QMdiArea()
        MainWindow.setCentralWidget(self.mdi)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 579, 21))

        self.menubar.setStyleSheet("QMenulBar{\n"

                                   "border-bottom: 1px solid #888888;\n"
                                   "}")
        self.menubar.setInputMethodHints(
            QtCore.Qt.ImhEmailCharactersOnly | QtCore.Qt.ImhFormattedNumbersOnly | QtCore.Qt.ImhUrlCharactersOnly)
        self.menubar.setObjectName("menubar")
        self.menus = QtWidgets.QMenu(self.menubar)
        self.menus.setEnabled(True)
        self.menus.setObjectName("menus")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuPlay_navigate = QtWidgets.QMenu(self.menubar)
        self.menuPlay_navigate.setObjectName("menuPlay_navigate")
        self.menuInstruments_markers = QtWidgets.QMenu(self.menubar)
        self.menuInstruments_markers.setObjectName("menuInstruments_markers")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setEnabled(True)
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        font.setKerning(True)
        self.toolBar.setFont(font)
        self.toolBar.setAutoFillBackground(False)
        self.toolBar.setStyleSheet("QToolBar{\n"
                                   "background-color: rgb(255, 255, 255);\n"
                                   "padding: 0px;\n"
                                   "}\n"
                                   "\n"
                                   "")
        self.toolBar.setMovable(False)
        self.toolBar.setIconSize(QtCore.QSize(30, 30))
        self.toolBar.setFloatable(True)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("open.png"),
                        QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen.setIcon(icon1)
        self.actionOpen.setObjectName("actionOpen")
        self.actionPlay = QtWidgets.QAction(MainWindow)
        self.actionPlay.setCheckable(True)
        self.actionPlay.setEnabled(False)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("play.png"),
                        QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPlay.setIcon(icon2)
        self.actionPlay.setObjectName("actionPlay")
        self.actionPause = QtWidgets.QAction(MainWindow)
        self.actionPause.setEnabled(False)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("stop.png"),
                        QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPause.setIcon(icon3)
        self.actionPause.setObjectName("actionPause")
        self.actionBack = QtWidgets.QAction(MainWindow)
        self.actionBack.setEnabled(False)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("back.png"),
                        QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionBack.setIcon(icon4)
        self.actionBack.setObjectName("actionBack")
        self.actionNext = QtWidgets.QAction(MainWindow)
        self.actionNext.setEnabled(False)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("next.png"),
                        QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNext.setIcon(icon5)
        self.actionNext.setObjectName("actionNext")
        self.actionZoomIn = QtWidgets.QAction(MainWindow)
        self.actionZoomIn.setEnabled(False)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap("zoom in.png"),
                        QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionZoomIn.setIcon(icon6)
        self.actionZoomIn.setObjectName("actionZoomIn")
        self.actionZoomOut = QtWidgets.QAction(MainWindow)
        self.actionZoomOut.setEnabled(False)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap("zoom out.png"),
                        QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionZoomOut.setIcon(icon7)
        self.actionZoomOut.setObjectName("actionZoomOut")
        self.actionSpectrogram = QtWidgets.QAction(MainWindow)
        self.actionSpectrogram.setEnabled(False)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap("spectr.png"),
                        QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSpectrogram.setIcon(icon8)
        self.actionSpectrogram.setObjectName("actionSpectrogram")
        self.actionSave_as = QtWidgets.QAction(MainWindow)
        self.actionSave_as.setEnabled(False)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("save.png"),
                        QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave_as.setIcon(icon9)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionPlay_as_fast_as_possible_2 = QtWidgets.QAction(MainWindow)
        self.actionPlay_as_fast_as_possible_2.setObjectName(
            "actionPlay_as_fast_as_possible_2")
        self.actionRepeat_forever_play_in_loop = QtWidgets.QAction(MainWindow)
        self.actionRepeat_forever_play_in_loop.setObjectName(
            "actionRepeat_forever_play_in_loop")
        self.actionPlay_as_fast_as_possible = QtWidgets.QAction(MainWindow)
        self.actionPlay_as_fast_as_possible.setObjectName(
            "actionPlay_as_fast_as_possible")
        self.menus.addAction(self.actionOpen)
        self.menus.addSeparator()
        self.menus.addAction(self.actionSave_as)
        self.menus.addSeparator()
        self.menus.addAction(self.actionExit)
        self.menuEdit.addAction(self.actionZoomIn)
        self.menuEdit.addAction(self.actionZoomOut)
        self.menuPlay_navigate.addAction(self.actionBack)
        self.menuPlay_navigate.addAction(self.actionNext)
        self.menuPlay_navigate.addSeparator()
        self.menuPlay_navigate.addAction(self.actionPlay)
        self.menuPlay_navigate.addSeparator()
        self.menuPlay_navigate.addAction(self.actionPause)
        self.menuInstruments_markers.addAction(self.actionSpectrogram)
        self.menubar.addAction(self.menus.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuPlay_navigate.menuAction())
        self.menubar.addAction(self.menuInstruments_markers.menuAction())
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionSave_as)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionZoomIn)
        self.toolBar.addAction(self.actionZoomOut)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionBack)
        self.toolBar.addAction(self.actionPlay)
        self.toolBar.addAction(self.actionNext)
        self.toolBar.addAction(self.actionPause)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSpectrogram)

        self.retranslateUi(MainWindow)
        self.actionExit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.actionOpen.triggered.connect(lambda: self.browsefiles())

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SIGVIEW"))
        self.menus.setStatusTip(_translate(
            "MainWindow", "Creates a new document"))
        self.menus.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuPlay_navigate.setTitle(
            _translate("MainWindow", "Play && navigate"))
        self.menuInstruments_markers.setTitle(
            _translate("MainWindow", "3D tools"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionOpen.setText(_translate("MainWindow", "Open signal..."))
        self.actionOpen.setStatusTip(
            _translate("MainWindow", "Opens new signal"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionPlay.setText(_translate(
            "MainWindow", "Play signal (no sound)"))
        self.actionPlay.setShortcut(_translate("MainWindow", "F5"))
        self.actionPause.setText(_translate("MainWindow", "Stop playing"))
        self.actionPause.setStatusTip(
            _translate("MainWindow", "Stops acqusition"))
        self.actionPause.setShortcut(_translate("MainWindow", "F7"))
        self.actionBack.setText(_translate(
            "MainWindow", "<< Signal beginning"))
        self.actionBack.setShortcut(_translate("MainWindow", "Home"))
        self.actionNext.setText(_translate("MainWindow", "Signal end >>"))
        self.actionNext.setShortcut(_translate("MainWindow", "End"))
        self.actionZoomIn.setText(_translate("MainWindow", "Zoom In"))
        self.actionZoomIn.setStatusTip(
            _translate("MainWindow", "Zoom selected part"))
        self.actionZoomIn.setShortcut(_translate("MainWindow", "Ctrl+Up"))
        self.actionZoomOut.setText(_translate("MainWindow", "Zoom Out"))
        self.actionZoomOut.setStatusTip(
            _translate("MainWindow", "Show previous zoom"))
        self.actionZoomOut.setShortcut(_translate("MainWindow", "Ctrl+Down"))
        self.actionSpectrogram.setText(_translate(
            "MainWindow", "Spectrogram"))
        self.actionSpectrogram.setStatusTip(_translate(
            "MainWindow", "Spectrum of the visible part of the signal"))
        self.actionSpectrogram.setShortcut(_translate("MainWindow", "Ctrl+G"))
        self.actionSave_as.setText(_translate(
            "MainWindow", "Save signal as..."))
        self.actionSave_as.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Alt+F4"))
        self.actionPlay_as_fast_as_possible_2.setText(
            _translate("MainWindow", "Play as fast as possible"))
        self.actionRepeat_forever_play_in_loop.setText(
            _translate("MainWindow", "Repeat forever (play in loop)"))
        self.actionRepeat_forever_play_in_loop.setStatusTip(_translate(
            "MainWindow", "Start playing signal from the beginning each time its end has been reached"))
        self.actionPlay_as_fast_as_possible.setText(
            _translate("MainWindow", "Play as fast as possible"))
# import app_rc


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())