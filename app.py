# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

import pandas as pd
import pyedflib
import os
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore, QtGui, QtWidgets, QtPrintSupport
from PyQt5.QtWidgets import QDialog, QFileDialog, QMainWindow, QWidget
from random import randint
from scipy import signal
from PIL import Image
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


class MdiWind(QtWidgets.QMdiSubWindow):
    def closeEvent(self, event):
        ui.openedWinds -= 1
        if ui.openedWinds == 0:
            ui.hideIcons()  


class MainWind(QtWidgets.QMainWindow):
    def closeEvent(self, event):
        if ui.closeWindow:
            reply = QtWidgets.QMessageBox()
            reply.setText("Do you really want to close Sigview?")
            reply.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            reply.setWindowTitle("Close")
            reply.setWindowIcon(QtGui.QIcon("blank.png"))
            reply = reply.exec()

            if reply == QtWidgets.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()


class Ui_MainWindow(QMainWindow):
    signals = []
    signalGraph = []
    zoomRange = []
    count = 0
    j = 0
    openedWinds = 0
    stop = False
    closeWindow = False
    zoomCnt = 0

    def print_widget(self, widget_list, filename):
        imagelist = []
        for widget in widget_list:
            # start scale
            indx,flag=self.titleIndex(widget)
            if flag:
                graphWidget = self.graphWidg(self.signals[indx-1])
                # graphWidget.setXRange(0, len(signal[indx-1]), padding=0)
                exporter = pg.exporters.ImageExporter(graphWidget.plotItem)
                exporter.params['width'] = 500
                exporter.params['height'] = 500
                exporter.export(f'fileName{str(indx)}.png')
            else:
                fig,_ = self.spectroDraw(self.signals[indx-1])
                fig.savefig(f'fileName{str(indx)}.png')
            img = Image.open(f'fileName{str(indx)}.png')
            imgConverted = img.convert('RGB')
            imagelist.append(imgConverted)
            os.remove(f'fileName{str(indx)}.png')
        imagelist[0].save(filename,save_all=True, append_images=imagelist[1:])
        # printer = QtPrintSupport.QPrinter(
        #     QtPrintSupport.QPrinter.HighResolution)
        # printer.setOutputFormat(QtPrintSupport.QPrinter.PdfFormat)
        # printer.setOutputFileName(filename)
        # painter = QtGui.QPainter(printer)
        # 
        # for widget in widget_list:
        #     xscale = printer.pageRect().width() * 1.0 / widget.width()
        #     yscale = printer.pageRect().height() * 1.0 / widget.height()
        #     scale = min(xscale, yscale)
        #     painter.translate(
        #         printer.paperRect().x() + printer.pageRect().width() / 2,
        #         printer.paperRect().y() + printer.pageRect().height() / 2,
        #     )
        #     painter.scale(scale, scale)
        #     painter.translate(-widget.width() / 2, -widget.height() / 2)
        #     # end scale
        #     # widget.graphWidget.setXRange(0, 300000)
        #     widget.render(painter)
        #     painter.resetTransform()
        #     printer.newPage()
        # painter.end()

    def printPDF(self, widget_list):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PDF", None, "PDF files (.pdf);;All Files()"
        )
        if fn:
            if QtCore.QFileInfo(fn).suffix() == "":
                fn += ".pdf"
                self.print_widget(widget_list, fn)

    def hideIcons(self):
        self.actionZoomIn.setEnabled(False)
        self.actionZoomOut.setEnabled(False)
        self.actionPlay.setEnabled(False)
        self.actionSpectrogram.setEnabled(False)
        self.actionSave_as.setEnabled(False)
        self.actionNext.setEnabled(False)
        self.actionBack.setEnabled(False)
        self.actionPause.setEnabled(False)

    def showIcons(self):
        self.actionZoomIn.setEnabled(True)
        self.actionZoomOut.setEnabled(True)
        self.actionPlay.setEnabled(True)
        self.actionPause.setEnabled(True)
        self.actionSpectrogram.setEnabled(True)
        self.actionSave_as.setEnabled(True)
        self.actionNext.setEnabled(True)
        self.actionBack.setEnabled(True)

    def titleIndex(self, subWindow):
        subWindowTitle = subWindow.windowTitle()
        if subWindowTitle.find("Time-FFT") == -1:
            if subWindowTitle[1] != "#":
                subWindowIndex = int(
                    subWindowTitle[0]) * 10 + int(subWindowTitle[1])
            else:
                subWindowIndex = int(subWindowTitle[0])
            return (subWindowIndex, True)
        else:
            if subWindowTitle[1] != "#":
                subWindowIndex = int(
                    subWindowTitle[12]) * 10 + int(subWindowTitle[13])
            else:
                subWindowIndex = int(subWindowTitle[12])
            return (subWindowIndex, False)

    def scrollRight(self, subWindow):
        subWindowIndex, flag = self.titleIndex(subWindow)
        if flag:
            subWindow.graphWidget.plotItem.getViewBox().translateBy(x=100, y=0)

    def scrollLeft(self, subWindow):
        subWindowIndex, flag = self.titleIndex(subWindow)
        if flag:
            subWindow.graphWidget.plotItem.getViewBox().translateBy(x=-100, y=0)

    def zoomIn(self, subWindow):
        subWindowIndex, flag = self.titleIndex(subWindow)
        self.zoomRange[subWindowIndex-1] = subWindow.graphWidget.viewRange(
        )[0][1] - subWindow.graphWidget.viewRange()[0][0]
        if flag and self.zoomRange[subWindowIndex-1] > 50:
            subWindow.graphWidget.plotItem.getViewBox().scaleBy(x=0.5, y=1)
            self.zoomRange[subWindowIndex-1] *= 0.5
            if (self.zoomRange[subWindowIndex-1] <= 50):
                self.actionZoomIn.setEnabled(False)
            if self.zoomRange[subWindowIndex-1] < len(self.signals[subWindowIndex-1]):
                self.actionZoomOut.setEnabled(True)

    def zoomOut(self, subWindow):
        subWindowIndex, flag = self.titleIndex(subWindow)
        self.zoomRange[subWindowIndex-1] = subWindow.graphWidget.viewRange(
        )[0][1] - subWindow.graphWidget.viewRange()[0][0]
        if flag and self.zoomRange[subWindowIndex-1] < len(self.signals[subWindowIndex-1]):
            subWindow.graphWidget.plotItem.getViewBox().scaleBy(x=2, y=1)
            self.zoomRange[subWindowIndex-1] *= 2
            if (self.zoomRange[subWindowIndex-1] >= len(self.signals[subWindowIndex-1])):
                self.actionZoomOut.setEnabled(False)
            if (self.zoomRange[subWindowIndex-1] > 50):
                self.actionZoomIn.setEnabled(True)

    def play(self, subWindow):
        subWindowIndex, flag = self.titleIndex(subWindow)
        self.zoomRange[subWindowIndex-1] = subWindow.graphWidget.viewRange(
        )[0][1] - subWindow.graphWidget.viewRange()[0][0]
        cursor = 0
        if flag:
            while cursor + 40 + self.signalGraph[subWindowIndex - 1] < len(
                self.signals[subWindowIndex - 1]
            ):
                if self.stop:
                    self.stop = False
                    self.signalGraph[subWindowIndex - 1] += cursor
                    break
                cursor += 40
                self.playClicked(subWindow, subWindowIndex,
                                 cursor)

    def playClicked(self, subWindow, subWindowIndex, cursor):
        subWindow.graphWidget.setXRange(
            self.signalGraph[subWindowIndex - 1] + cursor,
            self.zoomRange[subWindowIndex-1] + cursor +
            self.signalGraph[subWindowIndex - 1],
        )
        QtWidgets.QApplication.processEvents()

    def stopClicked(self):
        self.stop = True

    def spectroDraw(self,arr):
        figure = plt.figure()
        canvas = FigureCanvas(figure)
        figure.clear()
        f, t, Sxx = signal.spectrogram(arr, fs=200)
        ax = figure.add_subplot()
        ax.pcolormesh(t, f, 10 * np.log10(Sxx))
        canvas.draw()
        return(figure,canvas)

    def Spectrogram(self, arr, title):
        mydialog = QtWidgets.QMdiSubWindow(self)
        mydialog.figure, mydialog.canvas = self.spectroDraw(arr)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("sig.png"),
                       QtGui.QIcon.Normal, QtGui.QIcon.Off)
        mydialog.setWindowIcon(icon)
        mydialog.setWindowTitle(str(self.count) + "#Time-FFT: " + title)
        mydialog.setWidget(mydialog.canvas)
        self.mdi.addSubWindow(mydialog)
        mydialog.show()

    def openSpectro(self, subWindow):
        subWindowIndex, flag = self.titleIndex(subWindow)
        if flag:
            self.count = self.count + 1
            self.signalGraph.append(0)
            self.signals.append(0)
            self.zoomRange.append(0)
            self.Spectrogram(
                self.signals[subWindowIndex - 1], subWindow.windowTitle())

    def graphWidg(self,arr):
        graphWidget = pg.PlotWidget()
        graphWidget.setBackground("w")
        graphWidget.plot(arr, pen="b")
        graphWidget.showGrid(x=True, y=True)
        return(graphWidget)

    def openSecondDialog(self, arr, title):
        self.count = self.count + 1
        self.openedWinds += 1
        mydialog = MdiWind(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("sig.png"),
                       QtGui.QIcon.Normal, QtGui.QIcon.Off)
        mydialog.setWindowIcon(icon)
        mydialog.setWindowTitle(str(self.count) + "#" + title)
        mydialog.graphWidget = self.graphWidg(arr)
        mydialog.graphWidget.setXRange(0,400,padding=0)
        mydialog.setWidget(mydialog.graphWidget)
        x = mydialog.graphWidget.viewRange()
        self.mdi.addSubWindow(mydialog)
        mydialog.show()
        self.showIcons()

    def read_edf(self, filename):
        f = pyedflib.EdfReader(filename)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i)
        for i in range(0, n):
            self.signals.append(sigbufs[i])
            self.signalGraph.append(0)
            self.zoomRange.append(400)
            self.graphWidget = pg.PlotWidget()
            self.openSecondDialog(sigbufs[i], signal_labels[i])
        self.mdi.cascadeSubWindows()

    def read_txt(self, filename):
        with open(filename) as fp:
            signal_label = os.path.basename(filename)
            arr = []
            for line in fp:
                arr.append((line.rstrip().split(" ")[1]))
            data = np.array(arr).astype(np.float)
            self.signals.append(data)
            self.signalGraph.append(0)
            self.zoomRange.append(400)
            self.graphWidget = pg.PlotWidget()
            self.openSecondDialog(data, signal_label[0:-4])

    # def read_mat(self, filename):
    #     mat = loadmat(filename)
    #     signal_label = os.path.basename(filename)
    #     mat_file = pd.DataFrame(mat["F"]).iloc[:, 1]
    #     self.signals.append(mat_file)
    #     self.signalGraph.append(0)
    #     self.graphWidget = pg.PlotWidget()
    #     self.openSecondDialog(mat_file, signal_label[0:-4])

    def read_csv(self, filename):
        data = pd.read_csv(filename).iloc[:, 1]
        signal_label = os.path.basename(filename)
        array = data.to_numpy()
        self.signals.append(array)
        self.signalGraph.append(0)
        self.zoomRange.append(400)
        self.graphWidget = pg.PlotWidget()
        self.openSecondDialog(array, signal_label[0:-4])

    def browsefiles(self):
        self.closeWindow = True
        fname = QFileDialog.getOpenFileName(
            self, "Open file", "../", " *.edf;;" "*.csv;;" " *.txt;;" " *.mat;;"
        )
        file_path = fname[0]
        if file_path.endswith(".edf"):
            self.read_edf(file_path)
        elif file_path.endswith(".csv"):
            self.read_csv(file_path)
        elif file_path.endswith(".mat"):
            self.read_mat(file_path)
        elif file_path.endswith(".txt"):
            self.read_txt(file_path)

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
        self.mdi = QtWidgets.QMdiArea()
        MainWindow.setCentralWidget(self.mdi)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 579, 21))

        self.menubar.setStyleSheet(
            "QMenulBar{\n" "border-bottom: 1px solid #888888;\n" "}"
        )
        self.menubar.setInputMethodHints(
            QtCore.Qt.ImhEmailCharactersOnly
            | QtCore.Qt.ImhFormattedNumbersOnly
            | QtCore.Qt.ImhUrlCharactersOnly
        )
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
        self.toolBar.setStyleSheet(
            "QToolBar{\n"
            "background-color: rgb(255, 255, 255);\n"
            "padding: 0px;\n"
            "}\n"
            "\n"
            ""
        )
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
        icon6.addPixmap(
            QtGui.QPixmap("zoom in.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionZoomIn.setIcon(icon6)
        self.actionZoomIn.setObjectName("actionZoomIn")
        self.actionZoomOut = QtWidgets.QAction(MainWindow)
        self.actionZoomOut.setEnabled(False)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(
            QtGui.QPixmap("zoom out.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionZoomOut.setIcon(icon7)
        self.actionZoomOut.setObjectName("actionZoomOut")
        self.actionSpectrogram = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(
            QtGui.QPixmap("spectr.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionSpectrogram.setIcon(icon8)
        self.actionSpectrogram.setObjectName("actionSpectrogram")
        self.actionSpectrogram.setEnabled(False)
        self.actionSave_as = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap("save.png"),
                        QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave_as.setIcon(icon9)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionSave_as.setEnabled(False)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionPlay_as_fast_as_possible_2 = QtWidgets.QAction(MainWindow)
        self.actionPlay_as_fast_as_possible_2.setObjectName(
            "actionPlay_as_fast_as_possible_2"
        )
        self.actionRepeat_forever_play_in_loop = QtWidgets.QAction(MainWindow)
        self.actionRepeat_forever_play_in_loop.setObjectName(
            "actionRepeat_forever_play_in_loop"
        )
        self.actionPlay_as_fast_as_possible = QtWidgets.QAction(MainWindow)
        self.actionPlay_as_fast_as_possible.setObjectName(
            "actionPlay_as_fast_as_possible"
        )
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
        self.actionZoomIn.triggered.connect(
            lambda: self.zoomIn(self.mdi.activeSubWindow())
        )
        self.actionZoomOut.triggered.connect(
            lambda: self.zoomOut(self.mdi.activeSubWindow())
        )
        self.actionPlay.triggered.connect(
            lambda: self.play(self.mdi.activeSubWindow()))
        self.actionPause.triggered.connect(lambda: self.stopClicked())
        self.actionSpectrogram.triggered.connect(
            lambda: self.openSpectro(self.mdi.activeSubWindow())
        )
        self.actionNext.triggered.connect(
            lambda: self.openSpectro(
                self.scrollRight(self.mdi.activeSubWindow()))
        )
        self.actionBack.triggered.connect(
            lambda: self.openSpectro(
                self.scrollLeft(self.mdi.activeSubWindow()))
        )
        self.actionSave_as.triggered.connect(
            lambda: self.printPDF(self.mdi.subWindowList())
        )

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
        self.actionBack.setText(_translate("MainWindow", "Backward"))
        self.actionBack.setShortcut(_translate("MainWindow", "Ctrl+Left"))
        self.actionNext.setText(_translate("MainWindow", "Forward"))
        self.actionNext.setShortcut(_translate("MainWindow", "Ctrl+Right"))
        self.actionZoomIn.setText(_translate("MainWindow", "Zoom In"))
        self.actionZoomIn.setStatusTip(
            _translate("MainWindow", "Zoom selected part"))
        self.actionZoomIn.setShortcut(_translate("MainWindow", "Ctrl+Up"))
        self.actionZoomOut.setText(_translate("MainWindow", "Zoom Out"))
        self.actionZoomOut.setStatusTip(
            _translate("MainWindow", "Show previous zoom"))
        self.actionZoomOut.setShortcut(_translate("MainWindow", "Ctrl+Down"))
        self.actionSpectrogram.setText(_translate("MainWindow", "Spectrogram"))
        self.actionSpectrogram.setStatusTip(
            _translate(
                "MainWindow", "Spectrum of the visible part of the signal")
        )
        self.actionSpectrogram.setShortcut(_translate("MainWindow", "Ctrl+G"))
        self.actionSave_as.setText(_translate(
            "MainWindow", "Save signal as..."))
        self.actionSave_as.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Alt+F4"))
        self.actionPlay_as_fast_as_possible_2.setText(
            _translate("MainWindow", "Play as fast as possible")
        )
        self.actionRepeat_forever_play_in_loop.setText(
            _translate("MainWindow", "Repeat forever (play in loop)")
        )
        self.actionRepeat_forever_play_in_loop.setStatusTip(
            _translate(
                "MainWindow",
                "Start playing signal from the beginning each time its end has been reached",
            )
        )
        self.actionPlay_as_fast_as_possible.setText(
            _translate("MainWindow", "Play as fast as possible")
        )


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWind()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
