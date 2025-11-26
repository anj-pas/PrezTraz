# main.py - Wersja z Polilinią (LineString), Wybór przez Selekcję i Poprawionym Importem QGIS

from qgis.PyQt import QtWidgets, QtCore
from qgis.PyQt.QtWidgets import QAction, QFileDialog, QDockWidget, QPushButton, QComboBox, QVBoxLayout, QWidget, QDialog
from qgis.utils import iface
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, 
    QgsFields, QgsField, Qgis 
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
GLOBAL_LAYER_NAME = "Seismic Profiles Data (Lines)"

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
        
        self.ax.set_xlabel(f'Trace Index (1 to {n_tr})')
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
        
        self.dt_ms = 2.0
        self.seismic_layer = None 
        self.selection_connection = None 
        
        self.x_byte = 73
        self.y_byte = 77
        self.segy_handle = None 
        self.fname = None
        self.full_traces = None 
        self.launcher_dialog = None
        
        self.layer_id_to_monitor = None 


    def initGui(self):
        iface.addPluginToMenu('&SEGY Tools', self.action)
        iface.addToolBarIcon(self.action)
        
        QgsProject.instance().layersRemoved.connect(self.on_layers_removed)
        QgsProject.instance().layerLoaded.connect(self.on_layer_loaded)


    def unload(self):
        try:
            iface.removePluginMenu('&SEGY Tools', self.action)
            iface.removeToolBarIcon(self.action)
        except Exception:
            pass
            
        try:
            QgsProject.instance().layersRemoved.disconnect(self.on_layers_removed)
            QgsProject.instance().layerLoaded.disconnect(self.on_layer_loaded)
        except Exception:
            pass
            
        if self.selection_connection and self.seismic_layer:
            try:
                self.seismic_layer.selectionChanged.disconnect(self.selection_connection)
            except Exception:
                pass
            
        if self.segy_handle: 
            self.segy_handle.close()
        self.segy_handle = None
        self.full_traces = None
        self.seismic_layer = None


    def run(self):
        """Uruchamia okno launchera."""
        if self.launcher_dialog is None:
            self.launcher_dialog = SeismicLauncherDialog(self, iface.mainWindow())
        self.launcher_dialog.exec_()
        
        self._ensure_layer_is_monitored()


    def _ensure_layer_is_monitored(self):
        """Sprawdza, czy warstwa globalna istnieje i podłącza sygnał selectionChanged."""
        current_layer = None
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name() == GLOBAL_LAYER_NAME and isinstance(layer, QgsVectorLayer):
                current_layer = layer
                break
        
        if current_layer is None:
            return

        self.seismic_layer = current_layer

        if self.selection_connection is None:
            if self.seismic_layer.fields().indexOf('segy_path') > -1:
                self.selection_connection = self.seismic_layer.selectionChanged.connect(self.on_profile_selection)
                self.layer_id_to_monitor = self.seismic_layer.id()
                iface.messageBar().pushMessage("Monitorowanie", f"Podłączono monitorowanie selekcji dla warstwy: {GLOBAL_LAYER_NAME}.", level=Qgis.MessageLevel.Info, duration=2)
            else:
                self.seismic_layer = None


    def on_layers_removed(self, layerIds):
        """Monitoruje usunięcie globalnej warstwy z projektu."""
        if self.layer_id_to_monitor in layerIds:
            self.seismic_layer = None
            self.selection_connection = None
            self.layer_id_to_monitor = None
            self.full_traces = None
            if self.segy_handle: self.segy_handle.close()
            self.segy_handle = None
            if self.viewer_canvas: self.viewer_canvas.plot_seismic_image(None)
            iface.messageBar().pushMessage("Monitorowanie", f"Globalna warstwa SEGY została usunięta.", level=Qgis.MessageLevel.Info, duration=2)

    
    def on_layer_loaded(self, layer):
        """Monitoruje załadowanie warstwy (np. z GeoPackage) i podłącza monitorowanie."""
        if layer.name() == GLOBAL_LAYER_NAME and isinstance(layer, QgsVectorLayer):
            self._ensure_layer_is_monitored()


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

        self._create_or_get_seismic_layer()
        if not self.seismic_layer: return
        
        iface.statusBarIface().showMessage("Ładowanie metadanych i dodawanie cechy do globalnej warstwy LineString...")
        
        new_feature_id = self.add_feature_to_seismic_layer()
        if new_feature_id is None:
            iface.statusBarIface().clearMessage()
            return 
        
        self.seismic_layer.selectByIds([new_feature_id])
        iface.statusBarIface().clearMessage()
        

    def add_feature_to_seismic_layer(self):
        """Tworzy i dodaje nową cechę (linię) do globalnej warstwy."""
        if not self.seismic_layer: return None

        if self.segy_handle: self.segy_handle.close()
        try:
             self.segy_handle = segyio.open(self.fname, strict=False)
        except Exception as e:
            iface.messageBar().pushCritical("Błąd SEGYIO", f"Nie udało się otworzyć pliku SEGY: {e}")
            self.segy_handle = None
            return None

        try:
            f = self.segy_handle
            n_traces = f.tracecount
            sx = f.attributes(segyio.TraceField.SourceX)[:]
            sy = f.attributes(segyio.TraceField.SourceY)[:]
            
            try:
                dt = f.bin[segyio.BinField.Interval]
                self.dt_ms = dt / 1000.0
            except Exception:
                self.dt_ms = 2.0
        except Exception as e:
            iface.messageBar().pushCritical("Błąd SEGYIO", f"Nie udało się wczytać nagłówków SEGY: {e}")
            return None

        all_points = [QgsPointXY(float(sx[i]), float(sy[i])) for i in range(n_traces)]
        
        prov = self.seismic_layer.dataProvider()
        
        line_feature = QgsFeature()
        line_feature.setGeometry(QgsGeometry.fromPolylineXY(all_points))
        line_feature.setFields(self.seismic_layer.fields()) 

        base_name = os.path.basename(self.fname)
        line_feature['profile_name'] = base_name.split('.')[0]
        line_feature['traces_count'] = n_traces
        line_feature['segy_path'] = self.fname
        line_feature['x_byte'] = self.x_byte
        line_feature['y_byte'] = self.y_byte
        
        if prov.addFeatures([line_feature]):
            new_feature_id = line_feature.id()
            self.seismic_layer.updateExtents()
            self.iface.mapCanvas().refresh()
            return new_feature_id
        
        return None


    def _create_or_get_seismic_layer(self):
        """Tworzy globalną warstwę Polilinii (LineString), jeśli nie istnieje."""
        
        self._ensure_layer_is_monitored()
        if self.seismic_layer:
            return

        layer = QgsVectorLayer('LineString?crs=EPSG:4326', GLOBAL_LAYER_NAME, 'memory') 
        prov = layer.dataProvider()
        
        fields = QgsFields()
        fields.append(QgsField('profile_name', QtCore.QVariant.String))
        fields.append(QgsField('traces_count', QtCore.QVariant.Int))
        fields.append(QgsField('segy_path', QtCore.QVariant.String))
        fields.append(QgsField('x_byte', QtCore.QVariant.Int))
        fields.append(QgsField('y_byte', QtCore.QVariant.Int))
        
        if not prov.addAttributes(fields):
            iface.messageBar().pushCritical("Błąd", "Nie udało się dodać atrybutów do warstwy globalnej.")
            return
            
        layer.updateFields() 
        QgsProject.instance().addMapLayer(layer)
        self.seismic_layer = layer
        
        self._ensure_layer_is_monitored() 


    def on_profile_selection(self, selected_ids, deselected_ids):
        """
        Wywoływane, gdy selekcja w warstwie globalnej się zmienia.
        Odczytuje atrybuty z PIERWSZEJ zaznaczonej cechy i ładuje SEGY.
        """
        if not self.seismic_layer: return
        
        selected_features = self.seismic_layer.selectedFeatures()
        
        if not selected_features:
            if self.full_traces is not None or self.viewer_canvas is not None:
                iface.messageBar().pushMessage("Wizualizacja", "Brak zaznaczenia. Widok SEGY wyczyszczony.", level=Qgis.MessageLevel.Info, duration=2)
                self.full_traces = None
                if self.segy_handle: self.segy_handle.close()
                self.segy_handle = None
                if self.viewer_canvas: self.viewer_canvas.plot_seismic_image(None)
            return

        feature_to_load = selected_features[0]
        
        path = feature_to_load['segy_path']
        x_byte = feature_to_load['x_byte']
        y_byte = feature_to_load['y_byte']
        profile_name = feature_to_load['profile_name']
        
        if not path or path == self.fname:
            self.update_viewer_with_current_data()
            return
            
        iface.statusBarIface().showMessage(f"Przełączanie sekcji SEGY na: {profile_name}...")
        
        self.fname = path
        self.x_byte = x_byte
        self.y_byte = y_byte
        
        if self.segy_handle: self.segy_handle.close()
        
        try:
            self.segy_handle = segyio.open(self.fname, strict=False)
            
            try:
                self.dt_ms = self.segy_handle.bin[segyio.BinField.Interval] / 1000.0
            except Exception:
                self.dt_ms = 2.0
            
            self.full_traces = self.load_all_traces() 
            
            if self.full_traces is None:
                raise Exception("Nie udało się wczytać danych śladów.")

            if self.viewer_canvas:
                self.viewer_canvas.max_abs_amplitude = 1.0
            self.update_viewer_with_current_data()
            
            # POPRAWKA: Używamy iface.messageBar() zamiast iface.statusBarIface()
            iface.messageBar().pushMessage("Wizualizacja", f"Wyświetlam profil: {profile_name}.", level=Qgis.MessageLevel.Info, duration=3) 

        except Exception as e:
            iface.messageBar().pushCritical("Błąd Ładowania SEGY", f"Błąd ładowania pliku {profile_name}: {e}")
            self.full_traces = None
            self.segy_handle = None
            if self.viewer_canvas: self.viewer_canvas.plot_seismic_image(None)

        iface.statusBarIface().clearMessage()


    def update_viewer_with_current_data(self):
        """Otwiera docka, jeśli nie jest otwarty, i aktualizuje obraz pełnymi danymi."""
        if self.full_traces is None: return

        if self.viewer_dock is None:
            self.open_seismic_viewer_dock()
        else:
            self.viewer_dock.show()
            self.viewer_dock.setFloating(True)
        
        self.update_clip_and_cmap(data_to_plot=self.full_traces)


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


    def open_seismic_viewer_dock(self):
        """Tworzy docka, jeśli nie istnieje."""
        if self.viewer_dock is None:
            dock = QDockWidget('Seismic Image Viewer', iface.mainWindow())
            dock.setFloating(True) 
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

        self.viewer_dock.show()


    def update_clip_and_cmap(self, *args, data_to_plot=None):
        """Pobiera wartości z suwaka i ComboBox i odświeża obraz. 
           Używa pełnych danych, jeśli nie podano data_to_plot."""
        
        if data_to_plot is None:
            data_to_plot = self.full_traces
            
        if data_to_plot is None or self.viewer_canvas is None or self.viewer_dock is None:
             return
        
        clip_slider = self.viewer_dock.findChild(QtWidgets.QSlider)
        cmap_combo = self.viewer_dock.findChild(QComboBox)
        
        if clip_slider and cmap_combo:
            clip_percent = float(clip_slider.value())
            cmap_name = cmap_combo.currentText()
            
            self.viewer_canvas.plot_seismic_image(data_to_plot, 
                                                  dt_ms=self.dt_ms, 
                                                  clip_percent=clip_percent,
                                                  cmap_name=cmap_name)