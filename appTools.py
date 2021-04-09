import pandas as pd
import pyedflib
import os
import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from fpdf import FPDF
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QMainWindow,
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QSlider,
    QLabel,
    QGroupBox,
)
import math
import librosa
from librosa import display
import scipy.fftpack
from scipy.io.wavfile import write
from scipy import signal as sig
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


class EQWindow(QWidget):
    def __init__(self, parent=None):
        super(EQindow, self).__init__(parent)
        self.sliders = []
        self.gains = []
        for i in range(10):
            self.sliders.append(QSlider(Qt.Vertical))
        for i in range(10):
            self.gains.append(QLabel)
        grid = QGridLayout()
        grid.addWidget(self.createExampleGroup(60, 0), 0, 0)
        grid.addWidget(self.createExampleGroup(170, 1), 0, 1)
        grid.addWidget(self.createExampleGroup(310, 2), 0, 2)
        grid.addWidget(self.createExampleGroup(600, 3), 0, 3)
        grid.addWidget(self.createExampleGroup(1000, 4), 0, 4)
        grid.addWidget(self.createExampleGroup(3000, 5), 0, 5)
        grid.addWidget(self.createExampleGroup(6000, 6), 0, 6)
        grid.addWidget(self.createExampleGroup(12000, 7), 0, 7)
        grid.addWidget(self.createExampleGroup(14000, 8), 0, 8)
        grid.addWidget(self.createExampleGroup(16000, 9), 0, 9)
        self.setLayout(grid)

        self.setWindowTitle("PyQt5 Sliders")

    def createExampleGroup(self, txt, ind):
        groupBox = QGroupBox()

        self.sliders[ind].setMaximum(199)
        self.sliders[ind].setMinimum(-199)
        self.sliders[ind].setValue(0)
        self.sliders[ind].valueChanged.connect(lambda: self.valuechange(ind))
        freq = QLabel(self)
        freq.setText(str(txt) + " Hz")
        self.gains[ind] = QLabel()
        self.gains[ind].setText("0.0 dB")
        vbox = QVBoxLayout()
        vbox.addWidget(self.sliders[ind], alignment=Qt.AlignHCenter)
        vbox.addWidget(freq, alignment=Qt.AlignCenter)
        vbox.addWidget(self.gains[ind], alignment=Qt.AlignCenter)
        groupBox.setLayout(vbox)

        return groupBox

    def valuechange(self, ind):
        gainInc = float(self.sliders[ind].value() / 10)
        self.gains[ind].setText(str(gainInc) + " dB")


class MdiWind(QtWidgets.QMdiSubWindow):
    def closeEvent(self, event):
        # Checkes if there is an open subwindow
        if "Time-FFT" not in self.windowTitle():
            ui.activeWinds -= 1
            if ui.activeWinds == 0:
                ui.hideIcons()
        itr = 0
        # Adds closed subwindows to a list
        for widget in ui.mdi.subWindowList():
            if widget.windowTitle() == self.windowTitle():
                ui.deletedWinds.append(itr)
            itr += 1


class MainWind(QtWidgets.QMainWindow):
    def closeEvent(self, event):
        # Confirmation message when the user closes the app
        if ui.closeMssgBox:
            reply = QtWidgets.QMessageBox()
            reply.setText("Do you really want to close Sigview?")
            reply.setStandardButtons(
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            reply.setWindowTitle("Close")
            reply.setWindowIcon(QtGui.QIcon("icons/blank.png"))
            reply = reply.exec()

            if reply == QtWidgets.QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()


class Ui_MainWindow(QMainWindow):
    signals = []  # stores signals arrays
    graphRanges = (
        []
    )  # checkpoints for the latest zoom/seek action on the graph's X-axis for all subwindows
    zoomRanges = []  # Stores the shown range of the X-axis for each graph
    deletedWinds = []  # Stores the closed windows to erase them from the subWindowList
    windowsCount = 0  # Apply an index for each window
    activeWinds = 0  # Stores the number of active windows
    stop = False  # Checks if stop is clicked to affect the play function
    closeMssgBox = False  # Checks if a message box should appear on close event
    speedFactor = 1

    def equalizer(self):
        self.EQWind = EQWindow()
        mydialog = MdiWind(self)
        mydialog.setWidget(self.ewind)
        self.mdi.addSubWindow(mydialog)
        mydialog.show()

    def hideIcons(self):
        self.actionZoomIn.setEnabled(False)
        self.actionZoomOut.setEnabled(False)
        self.actionPlay.setEnabled(False)
        self.actionSpectrogram.setEnabled(False)
        self.actionSave_as.setEnabled(False)
        self.actionForward.setEnabled(False)
        self.actionBackward.setEnabled(False)
        self.actionPause.setEnabled(False)
        self.action0_5x.setEnabled(False)
        self.action1x.setEnabled(False)
        self.action2x.setEnabled(False)
        self.actionCascade.setEnabled(False)
        self.actionTile.setEnabled(False)
        self.actionCloseAll.setEnabled(False)

    def showIcons(self):
        self.actionZoomIn.setEnabled(True)
        self.actionZoomOut.setEnabled(True)
        self.actionPlay.setEnabled(True)
        self.actionSpectrogram.setEnabled(True)
        self.actionSave_as.setEnabled(True)
        self.actionForward.setEnabled(True)
        self.actionBackward.setEnabled(True)
        self.action0_5x.setEnabled(True)
        self.action1x.setEnabled(True)
        self.action2x.setEnabled(True)
        self.actionCascade.setEnabled(True)

        self.actionTile.setEnabled(True)
        self.actionCloseAll.setEnabled(True)

    def titleIndex(self, subWindowTitle):
        # Extracts the index of the subwindow from the window title and checks if the window is a spectrogram or a normal graph
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

    # PDF
    def generatePDF(self, widget_list, filename):
        # prints all opened signals and their spectrograms (if required)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_xy(0, 0)
        pdf.set_font("Arial", "B", 10)  # Font settings
        titlesList = []  # stores the titles of open widgets
        yCord = 0  # Y-coordinate on the PDF page
        itr = 0
        # To iterate on all the opened widgets to get their title
        for widget in widget_list:
            if itr not in self.deletedWinds:
                if widget.windowTitle().find("Time-FFT") == -1:
                    titlesList.append(widget.windowTitle())
                else:
                    # We put an indicator on the spectrogram widgets to mark them
                    if widget.windowTitle()[1] != "#":
                        tempStr = widget.windowTitle()[13:]
                    else:
                        tempStr = widget.windowTitle()[12:]
                    tempStr = tempStr + "x"
                    titlesList.append(tempStr)
            itr += 1
        titlesList.sort()
        for title in titlesList:
            windowIndx, _ = self.titleIndex(title)
            if title[-1] != "x":
                # The widgets are transformed into images to get inserted into the PDF
                graphPlot = self.graphDraw(self.signals[windowIndx - 1])
                imgName = f"fileName{str(windowIndx)}.png"
                exporter = pg.exporters.ImageExporter(graphPlot.plotItem)
                exporter.parameters()["width"] = 250
                exporter.parameters()["height"] = 250
                exporter.export(imgName)
                title = title[2:] if title[1] == "#" else title[3:]
                pdf.cell(0, 10, txt=title, ln=1, align="C")
                # We change the index of the Y-Coordinate to insert the next image
                yCord = pdf.get_y()
                pdf.image(
                    imgName,
                    x=None,
                    y=None,
                    w=95,
                    h=57,
                    type="PNG",
                    link="",
                )
                os.remove(imgName)
            else:
                fig, _ = self.spectroDraw(self.signals[windowIndx - 1])
                imgName = f".fileName{str(windowIndx + 99)}.png"
                fig.savefig(imgName)
                pdf.image(
                    imgName,
                    x=110,
                    y=yCord - 2,
                    w=95,
                    h=60,
                    type="PNG",
                    link="",
                )
                os.remove(imgName)
        pdf.output(filename)

    def printPDF(self, widget_list):
        # allows the user to save the file and name it as they like
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export PDF", None, "PDF files (.pdf);;All Files()"
        )
        if filename:
            if QtCore.QFileInfo(filename).suffix() == "":
                filename += ".pdf"
            self.generatePDF(widget_list, filename)

    # Scroll/Zoom
    def scrollRight(self, subWindow):
        subWindowIndex, graphFlag = self.titleIndex(subWindow.windowTitle())
        if graphFlag:
            subWindow.graphWidget.plotItem.getViewBox().translateBy(x=100, y=0)
            self.graphRanges[subWindowIndex - 1] += 100

    def scrollLeft(self, subWindow):
        subWindowIndex, graphFlag = self.titleIndex(subWindow.windowTitle())
        if graphFlag:
            subWindow.graphWidget.plotItem.getViewBox().translateBy(x=-100, y=0)
            self.graphRanges[subWindowIndex - 1] -= 100

    def zoomIn(self, subWindow):
        subWindowIndex, graphFlag = self.titleIndex(subWindow.windowTitle())
        if graphFlag:
            self.zoomRanges[subWindowIndex - 1] = (
                subWindow.graphWidget.viewRange()[0][1]
                - subWindow.graphWidget.viewRange()[0][0]
            )
            zoomRange = self.zoomRanges[subWindowIndex - 1]

            if self.zoomRanges[subWindowIndex - 1] > 50:
                subWindow.graphWidget.plotItem.getViewBox().scaleBy(x=0.5, y=1)
                self.zoomRanges[subWindowIndex - 1] *= 0.5
                self.graphRanges[
                    subWindowIndex - 1
                ] = subWindow.graphWidget.viewRange()[0][0]

                # Disables the zoom in button when the user reaches a certain range
                if self.zoomRanges[subWindowIndex - 1] <= 50:
                    self.actionZoomIn.setEnabled(False)
                if self.zoomRanges[subWindowIndex - 1] < len(
                    self.signals[subWindowIndex - 1]
                ):
                    # Enables the zoom out button when the user reaches a certain zoom-in-range
                    self.actionZoomOut.setEnabled(True)
            if self.plays and zoomRange >= len(self.signals[subWindowIndex - 1]):
                self.play(subWindow)

    def zoomOut(self, subWindow):
        subWindowIndex, graphFlag = self.titleIndex(subWindow.windowTitle())

        if graphFlag:
            self.zoomRanges[subWindowIndex - 1] = (
                subWindow.graphWidget.viewRange()[0][1]
                - subWindow.graphWidget.viewRange()[0][0]
            )

            if self.zoomRanges[subWindowIndex - 1] < len(
                self.signals[subWindowIndex - 1]
            ):
                subWindow.graphWidget.plotItem.getViewBox().scaleBy(x=2, y=1)
                self.zoomRanges[subWindowIndex - 1] *= 2
                self.graphRanges[
                    subWindowIndex - 1
                ] = subWindow.graphWidget.viewRange()[0][0]

                if self.zoomRanges[subWindowIndex - 1] >= len(
                    self.signals[subWindowIndex - 1]
                ):
                    self.actionZoomOut.setEnabled(False)
                if self.zoomRanges[subWindowIndex - 1] > 50:
                    self.actionZoomIn.setEnabled(True)

    # Play/Pause
    def setStep(self, value):
        self.speedFactor = value

    def play(self, subWindow):
        self.actionPause.setEnabled(True)
        self.actionPlay.setEnabled(False)
        self.plays = True
        subWindowIndex, graphFlag = self.titleIndex(subWindow.windowTitle())
        self.zoomRanges[subWindowIndex - 1] = (
            subWindow.graphWidget.viewRange()[0][1]
            - subWindow.graphWidget.viewRange()[0][0]
        )

        step = 0  # Cumulative variable that increases with time
        if graphFlag:
            # Check if this is the max limit of the signal is reached or not
            while step + 40 + self.graphRanges[subWindowIndex - 1] <= len(
                self.signals[subWindowIndex - 1]
                # while subWindow.graphWidget.viewRange()[0][1] < len(self.signals[subWindowIndex-1]):
            ):

                if self.stop:
                    self.stop = False
                    self.graphRanges[subWindowIndex - 1] += step
                    break
                step += 40 * self.speedFactor
                self.playProcess(subWindow, subWindowIndex, step)

    def playProcess(self, subWindow, subWindowIndex, step):
        subWindow.graphWidget.setXRange(
            self.graphRanges[subWindowIndex - 1] + step,
            self.zoomRanges[subWindowIndex - 1]
            + step
            + self.graphRanges[subWindowIndex - 1],
        )

        QtWidgets.QApplication.processEvents()

    def stopClicked(self):
        self.actionPause.setEnabled(False)
        self.actionPlay.setEnabled(True)
        self.stop = True
        self.plays = False

    # Spectrogram
    def spectroDraw(self, signal):
        # Draws the spectrogram of the signal
        figure = plt.figure()
        canvas = FigureCanvas(figure)
        figure.clear()
        f, t, Sxx = sig.spectrogram(signal, fs=200)
        ax = figure.add_subplot()
        ax.pcolormesh(t, f, 10 * np.log10(Sxx))
        canvas.draw()
        return (figure, canvas)

    def Spectrogram(self, signal, title):
        # Inserts the drawn spectrogram into a widget
        mydialog = MdiWind(self)
        mydialog.figure, mydialog.canvas = self.spectroDraw(signal)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap("icons/sig.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        mydialog.setWindowIcon(icon)
        mydialog.setWindowTitle(str(self.windowsCount) + "#Time-FFT: " + title)
        mydialog.setWidget(mydialog.canvas)
        self.mdi.addSubWindow(mydialog)
        mydialog.show()

    def checkSpectro(self, subWindow):
        # checks if the selected widget is a graph
        subWindowIndex, graphFlag = self.titleIndex(subWindow.windowTitle())
        if graphFlag:
            self.windowsCount = self.windowsCount + 1
            self.graphRanges.append(0)
            self.signals.append(0)
            self.zoomRanges.append(0)
            self.Spectrogram(
                self.signals[subWindowIndex - 1], subWindow.windowTitle())

    # Graphs
    def graphDraw(self, signal):
        # Plot the signal
        graphWidget = pg.PlotWidget()
        self.subwindow = graphWidget
        graphWidget.setBackground("w")
        graphWidget.plot(signal, pen="b")
        graphWidget.showGrid(x=True, y=True)
        return graphWidget

    def Graph(self, signal, title):
        # insert the plot to a widget
        self.windowsCount = self.windowsCount + 1
        self.activeWinds += 1
        mydialog = MdiWind(self)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap("icons/sig.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        mydialog.setWindowIcon(icon)
        mydialog.setWindowTitle(str(self.windowsCount) + "#" + title)
        mydialog.graphWidget = self.graphDraw(signal)
        mydialog.graphWidget.setXRange(0, 400, padding=0)
        mydialog.graphWidget.setLimits(
            xMin=0, xMax=len(signal), yMin=min(signal), yMax=max(signal)
        )
        mydialog.setWidget(mydialog.graphWidget)
        self.mdi.addSubWindow(mydialog)
        mydialog.show()
        self.showIcons()

    # Reading files
    def read_edf(self, filename):
        f = pyedflib.EdfReader(filename)
        n = f.signals_in_file  # number of signals in the file
        signal_labels = f.getSignalLabels()
        # initiates an ndarry of zeroes
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            # store the signal values in the array
            sigbufs[i, :] = f.readSignal(i)
        # Graph each sample in the file
        for i in range(0, n):
            self.signals.append(sigbufs[i])
            self.graphRanges.append(0)
            self.zoomRanges.append(400)
            self.graphWidget = pg.PlotWidget()
            self.Graph(sigbufs[i], signal_labels[i])
        self.mdi.cascadeSubWindows()

    def read_txt(self, filename):
        with open(filename) as fp:
            signal_label = os.path.basename(filename)
            signal = []
            for line in fp:
                signal.append((line.rstrip().split(" ")[1]))
            data = np.array(signal).astype(np.float)
            self.signals.append(data)
            self.graphRanges.append(0)
            self.zoomRanges.append(400)
            self.graphWidget = pg.PlotWidget()
            self.Graph(data, signal_label[0:-4])

    def read_csv(self, filename):
        data = pd.read_csv(filename).iloc[:, 1]
        signal_label = os.path.basename(filename)
        array = data.to_numpy()
        self.signals.append(array)
        self.graphRanges.append(0)
        self.zoomRanges.append(400)
        self.graphWidget = pg.PlotWidget()
        self.Graph(array, signal_label[0:-4])

    def read_wav(self, filename):
        signal_label = os.path.basename(filename)

        samples, sampling_rate = librosa.load(
            filename, sr=None, mono=True, offset=0.0, duration=None
        )

        self.signals.append(samples)
        self.graphRanges.append(0)
        self.zoomRanges.append(400)

        duration = len(samples) / sampling_rate
        time = np.arange(0, duration, 1 / sampling_rate)
        # librosa.display.waveplot(y=samples, sr=sampling_rate)
        frequency = []
        fftphase = np.angle(scipy.fft.rfft(samples))
        fft = abs(scipy.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(fft), (1.0 / sampling_rate))

        self.Graph(samples, signal_label)
        ffti = []
        x = 10 * 5
        y = int(5 * 2500)

        for i in range(x, y):
            fft[i] *= 50

        for i in range(50001):
            ffti.append(
                fft[i] * math.cos((fftphase[i])) + fft[i] *
                math.sin((fftphase[i])) * 1j
            )

        s = scipy.fft.irfft(ffti)
        s = np.array(s)

        self.signals.append(s)
        self.graphRanges.append(0)
        self.zoomRanges.append(400)
        self.Graph(s, signal_label + " modified")
        write("test.wav", sampling_rate, s)
        self.equalizer()

    def browsefiles(self):
        self.closeMssgBox = True
        fname = QFileDialog.getOpenFileName(
            self, "Open file", "../", " *.wav;;" " *.edf;;" "*.csv;;" " *.txt;;"
        )
        file_path = fname[0]
        if file_path.endswith(".edf"):
            self.read_edf(file_path)
        elif file_path.endswith(".csv"):
            self.read_csv(file_path)
        elif file_path.endswith(".txt"):
            self.read_txt(file_path)
        elif file_path.endswith(".wav"):
            self.read_wav(file_path)

    # GUI
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(
            QtGui.QPixmap("icons/sig.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
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
        self.menuInstruments_markers = QtWidgets.QMenu(self.menub99ar)
        self.menuInstruments_markers.setObjectName("menuInstruments_markers")
        self.menuWindow = QtWidgets.QMenu(self.menubar)
        self.menuWindow.setObjectName("menuWindow")
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
        icon1.addPixmap(
            QtGui.QPixmap(
                "icons/open.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionOpen.setIcon(icon1)
        self.actionOpen.setObjectName("actionOpen")
        self.actionPlay = QtWidgets.QAction(MainWindow)
        self.actionPlay.setEnabled(False)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(
            QtGui.QPixmap(
                "icons/play.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionPlay.setIcon(icon2)
        self.actionPlay.setObjectName("actionPlay")
        self.action0_5x = QtWidgets.QAction(MainWindow)
        self.action0_5x.setEnabled(False)
        icon20 = QtGui.QIcon()
        icon20.addPixmap(
            QtGui.QPixmap(
                "icons/0.5x.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.action0_5x.setIcon(icon20)
        self.action0_5x.setObjectName("action0_5x")
        self.action1x = QtWidgets.QAction(MainWindow)
        self.action1x.setEnabled(False)
        icon21 = QtGui.QIcon()
        icon21.addPixmap(
            QtGui.QPixmap("icons/1x.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.action1x.setIcon(icon21)
        self.action1x.setObjectName("action1x")
        self.action2x = QtWidgets.QAction(MainWindow)
        self.action2x.setEnabled(False)
        icon22 = QtGui.QIcon()
        icon22.addPixmap(
            QtGui.QPixmap("icons/2x.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.action2x.setIcon(icon22)
        self.action2x.setObjectName("action2x")
        self.actionPause = QtWidgets.QAction(MainWindow)
        self.actionPause.setEnabled(False)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(
            QtGui.QPixmap(
                "icons/stop.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionPause.setIcon(icon3)
        self.actionPause.setObjectName("actionPause")
        self.actionBackward = QtWidgets.QAction(MainWindow)
        self.actionBackward.setEnabled(False)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(
            QtGui.QPixmap(
                "icons/back.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionBackward.setIcon(icon4)
        self.actionBackward.setObjectName("actionBackward")
        self.actionForward = QtWidgets.QAction(MainWindow)
        self.actionForward.setEnabled(False)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(
            QtGui.QPixmap(
                "icons/next.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionForward.setIcon(icon5)
        self.actionForward.setObjectName("actionForward")
        self.actionZoomIn = QtWidgets.QAction(MainWindow)
        self.actionZoomIn.setEnabled(False)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(
            QtGui.QPixmap(
                "icons/zoom in.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionZoomIn.setIcon(icon6)
        self.actionZoomIn.setObjectName("actionZoomIn")
        self.actionZoomOut = QtWidgets.QAction(MainWindow)
        self.actionZoomOut.setEnabled(False)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(
            QtGui.QPixmap(
                "icons/zoom out.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionZoomOut.setIcon(icon7)
        self.actionZoomOut.setObjectName("actionZoomOut")
        self.actionSpectrogram = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(
            QtGui.QPixmap(
                "icons/spectr.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionSpectrogram.setIcon(icon8)
        self.actionSpectrogram.setObjectName("actionSpectrogram")
        self.actionSpectrogram.setEnabled(False)
        self.actionSave_as = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(
            QtGui.QPixmap(
                "icons/save.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off
        )
        self.actionSave_as.setIcon(icon9)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionSave_as.setEnabled(False)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionCascade = QtWidgets.QAction(MainWindow)
        self.actionCascade.setEnabled(False)
        self.actionCascade.setObjectName("actionCascade")
        self.actionTile = QtWidgets.QAction(MainWindow)
        self.actionTile.setEnabled(False)
        self.actionTile.setObjectName("actionTile")
        self.actionCloseAll = QtWidgets.QAction(MainWindow)
        self.actionCloseAll.setEnabled(False)
        self.actionCloseAll.setObjectName("actionCloseAll")

        self.menus.addAction(self.actionOpen)
        self.menus.addSeparator()
        self.menus.addAction(self.actionSave_as)
        self.menus.addSeparator()
        self.menus.addAction(self.actionExit)
        self.menuEdit.addAction(self.actionZoomIn)
        self.menuEdit.addAction(self.actionZoomOut)
        self.menuPlay_navigate.addAction(self.actionBackward)
        self.menuPlay_navigate.addAction(self.actionForward)
        self.menuPlay_navigate.addSeparator()
        self.menuPlay_navigate.addAction(self.actionPlay)
        self.menuPlay_navigate.addAction(self.action0_5x)
        self.menuPlay_navigate.addAction(self.action1x)
        self.menuPlay_navigate.addAction(self.action2x)
        self.menuPlay_navigate.addSeparator()
        self.menuPlay_navigate.addAction(self.actionPause)
        self.menuInstruments_markers.addAction(self.actionSpectrogram)
        self.menuWindow.addAction(self.actionCascade)
        self.menuWindow.addAction(self.actionTile)
        self.menuWindow.addAction(self.actionCloseAll)
        self.menubar.addAction(self.menus.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuPlay_navigate.menuAction())
        self.menubar.addAction(self.menuInstruments_markers.menuAction())
        self.menubar.addAction(self.menuWindow.menuAction())
        self.toolBar.addAction(self.actionOpen)
        self.toolBar.addAction(self.actionSave_as)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionZoomIn)
        self.toolBar.addAction(self.actionZoomOut)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionBackward)
        self.toolBar.addAction(self.actionPlay)
        self.toolBar.addAction(self.actionForward)
        self.toolBar.addAction(self.actionPause)
        self.toolBar.addAction(self.action0_5x)
        self.toolBar.addAction(self.action1x)
        self.toolBar.addAction(self.action2x)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.actionSpectrogram)

        # Connect the UI buttons to the logic of the program
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
            lambda: self.checkSpectro(self.mdi.activeSubWindow())
        )
        self.actionForward.triggered.connect(
            lambda: self.scrollRight(self.mdi.activeSubWindow())
        )
        self.actionBackward.triggered.connect(
            lambda: self.scrollLeft(self.mdi.activeSubWindow())
        )
        self.actionSave_as.triggered.connect(
            lambda: self.printPDF(self.mdi.subWindowList())
        )
        self.action0_5x.triggered.connect(lambda: self.setStep(0.5))
        self.action1x.triggered.connect(lambda: self.setStep(1))
        self.action2x.triggered.connect(lambda: self.setStep(2))
        self.actionCascade.triggered.connect(
            lambda: self.mdi.cascadeSubWindows())
        self.actionTile.triggered.connect(lambda: self.mdi.tileSubWindows())
        self.actionCloseAll.triggered.connect(
            lambda: self.mdi.closeAllSubWindows())

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
        self.menuWindow.setTitle(_translate("MainWindow", "Window"))
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
        self.actionPause.setShortcut(_translate("MainWindow", "F6"))
        self.actionBackward.setText(_translate("MainWindow", "Backward"))
        self.actionBackward.setShortcut(_translate("MainWindow", "Ctrl+Left"))
        self.actionForward.setText(_translate("MainWindow", "Forward"))
        self.actionForward.setShortcut(_translate("MainWindow", "Ctrl+Right"))
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
        self.actionCascade.setText(_translate("MainWindow", "Cascade"))
        self.actionTile.setText(_translate("MainWindow", "Tile"))
        self.actionCloseAll.setText(_translate("MainWindow", "Close All"))
        self.action0_5x.setText(_translate("MainWindow", "Slower"))
        self.action1x.setText(_translate("MainWindow", "Normal"))
        self.action2x.setText(_translate("MainWindow", "Faster"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWind()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
