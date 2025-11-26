# main.py - Wersja z Poliliniami, Skalą 'seismic', Detached Viewer i Poprawioną Logiką Atrybutów

from qgis.PyQt import QtWidgets, QtCore
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QDockWidget, QPushButton, QComboBox, QVBoxLayout, QWidget, QDialog
from qgis.utils import iface
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, 
    QgsFields, QgsField
)
import numpy as np
import segyio
import os
import traceback  

# matplotlib imports must use the Qt backend available w QGIS
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# ==============================================================================
# 0. STAŁE
# ==============================================================================

STANDARD_CMAPS = [
    'seismic', 'gray', 'gray_r', 'viridis', 'plasma', 'inferno', 'magma', 'bone'
]
DEFAULT_CMAP = 'seismic'

# ==============================================================================
# 1. OKNO STARTOWE (LAUNCHER)
# ==============================================================================

class SeismicLauncherDialog(QDialog):
    def __init__(self, plugin_ref, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SEGY Tools Launcher")
        self.plugin_ref = plugin_ref
        
        layout = QVBoxLayout(self)
        
        btn_segy_loader = QPushButton("1. Load SEGY Section & Viewer")
        btn_segy_loader.clicked.connect(self.run_segy_loader)
        layout.addWidget(btn_segy_loader)
        
        btn_future = QPushButton("2. Future Tool (Disabled)")
        btn_future.setEnabled(False)
        layout.addWidget(btn_future)

    def run_segy_loader(self):
        self.accept()
        self.plugin_ref.open_segy_file_dialog()

# ==============================================================================
# 2. DIALOG WYBORU BAJTÓW WSPÓŁRZĘDNYCH
# ==============================================================================

class SegyLoadDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, fname=""):
        super().__init__(parent)
        self.setWindowTitle(f"Load SEGY: {os.path.basename(fname)}")
        
        layout = QtWidgets.QVBoxLayout(self)
        byte_group = QtWidgets.QGroupBox("Source Coordinates (Byte Locations)")
        byte_layout = QtWidgets.QFormLayout(byte_group)
        self.x_byte_input = QtWidgets.QLineEdit("73") 
        self.y_byte_input = QtWidgets.QLineEdit("77") 
        byte_layout.addRow("Source X (Byte #):", self.x_byte_input)
        byte_layout.addRow("Source Y (Byte #):", self.y_byte_input)
        layout.addWidget(byte_group)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_coords_bytes(self):
        try:
            x_byte = int(self.x_byte_input.text())
            y_byte = int(self.y_byte_input.text())
            return x_byte, y_byte
        except ValueError:
            return None, None 

# ==============================================================================
# 3. CANVAS (WIZUALIZACJA SEJSMICZNA)
# ==============================================================================

class SeismicImageCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(6, 8))
        super().__init__(self.fig)
        self.setParent(parent) 
        self.data = None 
        self.times = None
        self.dt_ms = 2.0
        self.max_abs_amplitude = 1.0

    def plot_seismic_image(self, data, dt_ms=2.0, clip_percent=100.0, cmap_name=DEFAULT_CMAP):
        self.ax.clear()
        
        self.data = data 
        self.dt_ms = dt_ms
        
        if self.data is None or self.data.size == 0: 
            self.ax.set_title('No Seismic Data Loaded or Selected')
            self.ax.set_xlabel('')
            self.ax.set_ylabel('')
            self.draw()
            return

        n_samp, n_tr = self.data.shape 
        self.times = np.arange(n_samp) * self.dt_ms
        
        if self.max_abs_amplitude == 1.0:
            self.max_abs_amplitude = np.nanmax(np.abs(self.data))
        
        vmax = self.max_abs_amplitude * (clip_percent / 100.0)
        
        extent = (0, n_tr, self.times[0], self.times[-1]) 

        self.ax.imshow(self.data, 
                       cmap=cmap_name, 
                       aspect='auto', 
                       extent=extent,
                       vmin=-vmax, 
                       vmax=vmax)
        
        self.ax.set_xlabel('Trace Index (Left to Right)')
        self.ax.set_ylabel('Time (ms)')
        self.ax.set_title(f'Seismic Section (Clip: {clip_percent:.1f}%)')
        self.draw()


# ==============================================================================
# 4. SEISMIC PLUGIN
# ==============================================================================

class SeismicPlugin:
    
    def __init__(self, iface):
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.action = QAction('SEGY Tools Launcher', iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.viewer_dock = None
        self.viewer_canvas = None 
        self.trace_positions = None
        self.dt_ms = 2.0
        self.trace_layer = None
        self.x_byte = 73
        self.y_byte = 77
        self.segy_handle = None 
        self.fname = None
        self.traces = None 
        self.launcher_dialog = None
        
        self.loaded_segy_data = {} 


    def initGui(self):
        iface.addPluginToMenu('&SEGY Tools', self.action)
        iface.addToolBarIcon(self.action)
        
        iface.layerTreeView().currentLayerChanged.connect(self.on_layer_selection_changed)


    def unload(self):
        try:
            iface.removePluginMenu('&SEGY Tools', self.action)
            iface.removeToolBarIcon(self.action)
        except Exception:
            pass
            
        try:
            iface.layerTreeView().currentLayerChanged.disconnect(self.on_layer_selection_changed)
        except Exception:
            pass
            
        if self.segy_handle: 
            self.segy_handle.close()
        self.segy_handle = None
        self.traces = None
        self.loaded_segy_data = {}


    def run(self):
        """Uruchamia okno launchera."""
        if self.launcher_dialog is None:
            self.launcher_dialog = SeismicLauncherDialog(self, iface.mainWindow())
        self.launcher_dialog.exec_()


    def open_segy_file_dialog(self):
        """Uruchamia dialog wyboru pliku SEGY i inicjuje ładowanie."""
        fname, _ = QFileDialog.getOpenFileName(iface.mainWindow(), 'Open SEG-Y 2D profile', '', 'SEG-Y files (*.sgy *.segy)')
        if not fname: return

        dialog = SegyLoadDialog(iface.mainWindow(), fname)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
            
        x_byte, y_byte = dialog.get_coords_bytes()
        self.x_byte = x_byte
        self.y_byte = y_byte
        self.fname = fname

        if self.segy_handle: self.segy_handle.close()
        try:
             self.segy_handle = segyio.open(fname, strict=False)
        except Exception as e:
            iface.messageBar().pushCritical("Błąd SEGYIO", f"Nie udało się otworzyć pliku SEGY: {e}")
            self.segy_handle = None
            return 

        iface.statusBarIface().showMessage("Ładowanie metadanych i warstwy polilinii SEGY...")
        
        # 1. Ładowanie metadanych i dodawanie warstwy do projektu
        if not self.load_metadata_and_layer(): 
            iface.statusBarIface().clearMessage()
            if self.segy_handle: self.segy_handle.close() 
            self.segy_handle = None
            return 
        
        # 2. Ładowanie wszystkich danych śladów
        iface.statusBarIface().showMessage("Ładowanie wszystkich danych śladów do pamięci RAM...")
        self.traces = self.load_all_traces()
        
        if self.traces is None:
            iface.messageBar().pushCritical("Błąd Ładowania", "Nie udało się wczytać wszystkich danych śladów.")
            iface.statusBarIface().clearMessage()
            if self.segy_handle: self.segy_handle.close() 
            self.segy_handle = None
            return

        # Ustawiamy nowo utworzoną warstwę jako aktywną
        if self.trace_layer:
            iface.layerTreeView().setCurrentLayer(self.trace_layer)
        
        iface.statusBarIface().clearMessage()


    def on_layer_selection_changed(self, layer):
        """Reaguje na zmianę aktywnej warstwy w Panelu Warstw, ładując odpowiednią sekcję SEGY."""
        
        if layer is None:
            self.traces = None 
            if self.viewer_canvas:
                self.viewer_canvas.plot_seismic_image(None)
            return

        layer_id = layer.id()
        
        if layer_id in self.loaded_segy_data:
            data_info = self.loaded_segy_data[layer_id]
            
            if self.fname != data_info['path']:
                iface.statusBarIface().showMessage(f"Przełączanie sekcji: {os.path.basename(data_info['path'])}...")

                self.fname = data_info['path']
                self.x_byte = data_info['x_byte']
                self.y_byte = data_info['y_byte']
                
                if self.segy_handle: self.segy_handle.close()
                try:
                    self.segy_handle = segyio.open(self.fname, strict=False)
                except Exception as e:
                    iface.messageBar().pushCritical("Błąd SEGYIO", f"Nie udało się otworzyć pliku dla {layer.name()}: {e}")
                    self.traces = None
                    self.segy_handle = None
                    if self.viewer_canvas: self.viewer_canvas.plot_seismic_image(None)
                    return
                
                try:
                    self.dt_ms = self.segy_handle.bin[segyio.BinField.Interval] / 1000.0
                except Exception:
                    self.dt_ms = 2.0
                    
                self.traces = self.load_all_traces() 
                if self.traces is None:
                    iface.messageBar().pushCritical("Błąd Ładowania", f"Nie udało się wczytać danych dla {layer.name()}.")
                    return
                
                if self.viewer_canvas:
                    self.viewer_canvas.max_abs_amplitude = 1.0 
                    
            self.trace_layer = layer
            self.update_viewer_with_current_data()
            iface.statusBarIface().clearMessage()

        else:
            self.traces = None
            if self.viewer_canvas:
                self.viewer_canvas.plot_seismic_image(None)


    def load_all_traces(self):
        """Wczytuje CAŁY blok danych śladów w formacie (Sample, Trace)."""
        if not self.segy_handle: return None
        try:
            f = self.segy_handle
            n_traces = f.tracecount
            data = np.stack([f.trace[i] for i in range(n_traces)], axis=1) 
            return data
        except Exception as e:
            iface.messageBar().pushCritical("Błąd I/O", f"Błąd podczas ładowania całego bloku danych: {e}\n{traceback.format_exc()}")
            return None


    def load_metadata_and_layer(self):
        """Wczytuje metadane (nagłówki) i tworzy warstwę polilinii."""
        if not self.segy_handle: return False
        
        try:
            f = self.segy_handle
            n_traces = f.tracecount
            
            # Pobieramy współrzędne wszystkich śladów
            sx = f.attributes(segyio.TraceField.SourceX)[:]
            sy = f.attributes(segyio.TraceField.SourceY)[:]
            
            try:
                dt = f.bin[segyio.BinField.Interval]
                self.dt_ms = dt / 1000.0
            except Exception:
                self.dt_ms = 2.0

        except Exception as e:
            iface.messageBar().pushCritical("Błąd SEGYIO", f"Nie udało się wczytać nagłówków SEGY: {e}")
            return False

        # Utworzenie listy punktów dla polilinii
        all_points = []
        for i in range(n_traces):
            all_points.append(QgsPointXY(float(sx[i]), float(sy[i])))
        
        self.trace_positions = all_points 
        
        # 1. Definicja i inicjalizacja warstwy
        layer_name = os.path.basename(self.fname) + ' Line'
        layer = QgsVectorLayer('LineString?crs=EPSG:4326', layer_name, 'memory')
        prov = layer.dataProvider()
        
        # 2. Definicja i dodanie pól do warstwy
        fields = QgsFields()
        fields.append(QgsField('traces_count', QtCore.QVariant.Int))
        if not prov.addAttributes(fields):
            iface.messageBar().pushCritical("Błąd", "Nie udało się dodać atrybutów do warstwy.")
            return False
            
        layer.updateFields() # Aktualizujemy, aby pola były dostępne
        
        # 3. Tworzenie i ustawianie geometrii obiektu (CECHY)
        line_feature = QgsFeature()
        line_feature.setGeometry(QgsGeometry.fromPolylineXY(self.trace_positions))
        
        # POPRAWKA: Ustawiamy definicję pól dla cechy po ich dodaniu do warstwy
        line_feature.setFields(layer.fields()) 

        # 4. Ustawienie wartości atrybutu
        line_feature['traces_count'] = n_traces
        
        # 5. Dodanie cechy do warstwy
        prov.addFeatures([line_feature])
        layer.updateExtents()
        
        QgsProject.instance().addMapLayer(layer)
        
        # REJESTRACJA
        self.loaded_segy_data[layer.id()] = {
            'path': self.fname,
            'x_byte': self.x_byte,
            'y_byte': self.y_byte,
        }
        
        self.trace_layer = layer 
        
        self.iface.mapCanvas().setExtent(layer.extent())
        self.iface.mapCanvas().refresh()
        return True

    def open_seismic_viewer_dock(self):
        """Tworzy docka, jeśli nie istnieje."""
        if self.viewer_dock is None:
            dock = QDockWidget('Seismic Image Viewer', iface.mainWindow())
            dock.setFloating(True) # Okno startuje jako pływające
            dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
            
            widget = QtWidgets.QWidget()
            vbox = QtWidgets.QVBoxLayout()
            
            self.viewer_canvas = SeismicImageCanvas(parent=widget) 
            
            clip_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            clip_slider.setMinimum(1)  
            clip_slider.setMaximum(100)
            clip_slider.setValue(100)
            clip_slider.setToolTip("Amplitude Clipping (%)")
            clip_slider.valueChanged.connect(self.update_clip_and_cmap)
            
            cmap_combo = QComboBox()
            cmap_combo.addItems(STANDARD_CMAPS)
            cmap_combo.setCurrentText(DEFAULT_CMAP) 
            cmap_combo.setToolTip("Select Colormap")
            cmap_combo.currentTextChanged.connect(self.update_clip_and_cmap)
            
            vbox.addWidget(cmap_combo)
            vbox.addWidget(clip_slider)
            vbox.addWidget(self.viewer_canvas)
            widget.setLayout(vbox)
            dock.setWidget(widget)
            iface.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            self.viewer_dock = dock

        self.viewer_dock.show() # Zapewnia widoczność po utworzeniu


    def update_viewer_with_current_data(self):
        """Otwiera docka, jeśli nie jest otwarty, i aktualizuje obraz nowymi danymi."""
        if self.traces is None: return

        if self.viewer_dock is None:
            self.open_seismic_viewer_dock()
        else:
            self.viewer_dock.show()
            self.viewer_dock.setFloating(True)
        
        self.update_clip_and_cmap()


    def update_clip_and_cmap(self, *args):
        """Pobiera wartości z suwaka i ComboBox i odświeża obraz."""
        if self.traces is None or self.viewer_canvas is None or self.viewer_dock is None:
             return
        
        clip_slider = self.viewer_dock.findChild(QtWidgets.QSlider)
        cmap_combo = self.viewer_dock.findChild(QComboBox)
        
        if clip_slider and cmap_combo:
            clip_percent = float(clip_slider.value())
            cmap_name = cmap_combo.currentText()
            
            self.viewer_canvas.plot_seismic_image(self.traces, 
                                                  dt_ms=self.dt_ms, 
                                                  clip_percent=clip_percent,
                                                  cmap_name=cmap_name)