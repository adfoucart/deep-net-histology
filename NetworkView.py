import sys
from PyQt4 import QtGui

class NetworkView(QtGui.QMainWindow):

	def __init__(self):
		super(NetworkView, self).__init__()

		self.unsaved = False

		self.initUI()

	def initUI(self):
		# textEdit = QtGui.QTextEdit()
		# self.setCentralWidget(textEdit)

		btn = QtGui.QPushButton("Button", self)
		btn.setToolTip("what's this ?")
		btn.resize(btn.sizeHint())
		btn.move(50,200)

		self.buildMenu()
		self.buildToolbar()

		self.statusBar().showMessage("Ready")

		self.resize(800,600)
		self.center()
		self.setWindowTitle('Network View')
		self.setWindowIcon(QtGui.QIcon('icons/network.png'))

		self.show()

	def buildMenu(self):
		self.exitAction = ExitAction(self)
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(self.exitAction)

	def buildToolbar(self):
		self.toolbar = self.addToolBar('Exit')
		self.toolbar.addAction(self.exitAction)

	def center(self):
		qr = self.frameGeometry()
		cp = QtGui.QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def closeEvent(self, event):
		if self.unsaved:
			reply = QtGui.QMessageBox.question(self, 'Message', "<b>You have unsaved work</b>.<br />Are you sure you want to quit ?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.No)

			if reply == QtGui.QMessageBox.Yes:
				event.accept()
			else:
				event.ignore()
		else:
			event.accept()

class ExitAction(QtGui.QAction):
	def __init__(self, window):
		super(ExitAction, self).__init__(QtGui.QIcon('icons/exit.png'), '&Exit', window)

		self.setShortcut('Ctrl+Q')
		self.setStatusTip('Exit application')
		self.triggered.connect(window.close)


def main():
	app = QtGui.QApplication(sys.argv)

	view = NetworkView()

	sys.exit(app.exec_())

if __name__ == "__main__":
	main()