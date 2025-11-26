# main.py - Final Version: English, Dynamic Icon Path, CRS Selection, and Fixes

from qgis.PyQt import QtWidgets, QtCore
from qgis.PyQt.QtWidgets import (
    QAction, QFileDialog, QDockWidget, QPushButton, QComboBox, 
    QVBoxLayout, QWidget, QDialog, QGroupBox, QSlider, QLabel, 
    QHBoxLayout, QToolButton, QFormLayout, QStyle 
)
from qgis.utils import iface
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, 
    QgsFields, QgsField, Qgis, 
    QgsFeatureRequest, QgsCoordinateReferenceSystem
)
from qgis.gui import QgsCrsSelectionWidget 
from qgis.PyQt.QtGui import QIcon 

import numpy as np
import segyio
import os # Import os for path handling
import sys # Import sys for script location
import traceback  
import math 
from scipy.signal import hilbert 

# matplotlib imports must use the Qt backend available w QGIS
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar

# ==============================================================================
# 0. CONSTANTS
# ==============================================================================

STANDARD_CMAPS = [
    'seismic', 'gray', 'gray_r', 'viridis', 'plasma', 'inferno', 'magma', 'bone'
]
DEFAULT_CMAP = 'seismic'
GLOBAL_LAYER_NAME = "Seismic Profiles Data (Lines)"

DEFAULT_X_BYTE = 73
DEFAULT_Y_BYTE = 77

# DYNAMIC ICON PATH CALCULATION
# This assumes the icon file is in the same directory as this script.
try:
    # Use os.path.abspath(__file__) to get the full path to the current script file.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., in Python Console)
    # This might require manually passing the script directory or relying on QGIS environment.
    # For execution via Python console as suggested, this might be tricky, but we proceed.
    # If this fails, the icon fallback will take over.
    SCRIPT_DIR = "."

ICON_FILE_NAME = "icon.png"
ICON_PATH = os.path.join(SCRIPT_DIR, ICON_FILE_NAME) 

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

def get_bearing(p1, p2):
    """Calculates the initial bearing (angle in degrees) from point p1 to p2 (X, Y)."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    rad = math.atan2(dx, dy)
    deg = math.degrees(rad)
    bearing = (deg + 360) % 360
    return bearing

def get_cardinal_direction(bearing):
    """Converts an angle (0-360) to a cardinal direction (N, NE, E, SE, S, SW, W, NW)."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    index = round(bearing / 45.0)
    return directions[index]

def calculate_seismic_attribute(data, attribute_name):
    """Calculates the selected seismic attribute."""
       
    if data is None or not isinstance(data, np.ndarray) or data.size == 0:
        return None
        
    if attribute_name == 'None':
        return data 
    
    n_samp, n_tr = data.shape
    output_data = np.zeros_like(data, dtype=np.float32)

    for i in range(n_tr):
        trace = data[:, i]
        if np.all(trace == 0):
             analytic_signal = np.zeros_like(trace, dtype=complex)
        else:
             analytic_signal = hilbert(trace)
        
        if attribute_name == 'Envelope':
            output_data[:, i] = np.abs(analytic_signal)
        elif attribute_name == 'Instantaneous Phase':
            output_data[:, i] = np.angle(analytic_signal, deg=True)
        
    return output_data

# ==============================================================================
# 2. LAUNCHER DIALOG 
# ==============================================================================

class SeismicLauncherDialog(QDialog):
    def __init__(self, plugin_ref, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PrezTraz")
        self.plugin_ref = plugin_ref
        
        layout = QVBoxLayout(self)
        
        btn_segy_loader = QPushButton("1. Load SEGY Section(s) & Viewer")
        btn_segy_loader.clicked.connect(self.run_segy_loader)
        layout.addWidget(btn_segy_loader)
        
        btn_future = QPushButton("2. Future Tool (Disabled)")
        btn_future.setEnabled(False)
        layout.addWidget(btn_future)

    def run_segy_loader(self):
        self.accept()
        self.plugin_ref.open_segy_file_dialog()

# ==============================================================================
# 3. COORDINATE BYTES AND CRS SELECTION DIALOG 
# ==============================================================================

class SegyLoadDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Load SEGY: Settings")
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # --- 1. Byte Location Group ---
        
        byte_group = QtWidgets.QGroupBox("Source Coordinates (Byte Locations)")
        byte_layout = QFormLayout(byte_group)
        
        self.x_byte_input = QtWidgets.QLineEdit(str(DEFAULT_X_BYTE)) 
        self.y_byte_input = QtWidgets.QLineEdit(str(DEFAULT_Y_BYTE)) 
        
        byte_layout.addRow("Source X (Byte #):", self.x_byte_input)
        byte_layout.addRow("Source Y (Byte #):", self.y_byte_input)
        layout.addWidget(byte_group)

        # --- 2. CRS Group ---
        
        crs_group = QtWidgets.QGroupBox("Coordinate Reference System (CRS)")
        crs_layout = QVBoxLayout(crs_group)
        
        self.crs_widget = QgsCrsSelectionWidget(iface.mainWindow())
        # Set default to project CRS
        self.crs_widget.setCrs(QgsProject.instance().crs()) 
        
        crs_layout.addWidget(QLabel("Select the CRS of the SEGY file coordinates:"))
        crs_layout.addWidget(self.crs_widget)
        layout.addWidget(crs_group)

        # --- Buttons ---
        
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

    def get_selected_crs(self):
        # Return the selected QgsCoordinateReferenceSystem object
        return self.crs_widget.crs()

# ==============================================================================
# 4. CANVAS (SEISMIC VISUALIZATION)
# ==============================================================================

class SeismicImageCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(6, 8))
        super().__init__(self.fig)
        self.setParent(parent) 
        self.data = None 
        self.times = None
        self.dt_ms = 2.0
        self.max_abs_amplitude = 1.0
        self.im = None 
        
        self.profile_name = "Seismic Section" 
        self.start_coord = None 
        self.end_coord = None   
        
        self.info_text_top = None 
        self.info_text_bottom = None
        
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('motion_notify_event', self.on_cursor_move_update)
        
        self._drag_active = False
        self._last_x = None
        self._last_y = None


    def plot_seismic_image(self, data, dt_ms=2.0, clip_percent=100.0, cmap_name=DEFAULT_CMAP, 
                           start_coord=None, end_coord=None, profile_name="Seismic Section", 
                           attribute_name='None'): 
        self.fig.clf() 
        
        self.info_text_top = None
        self.info_text_bottom = None
        
        self.ax = self.fig.add_axes([0.05, 0.05, 0.88, 0.90]) 
        self.cax = self.fig.add_axes([0.94, 0.15, 0.02, 0.70]) 

        processed_data = calculate_seismic_attribute(data, attribute_name)
        
        if processed_data is None or processed_data.size == 0: 
            self.data = None
            self.ax.set_title(profile_name + " (No Data)")
            self.draw()
            return
            
        processed_data = np.flipud(processed_data)
        self.data = processed_data 
        
        self.dt_ms = dt_ms
        self.start_coord = start_coord
        self.end_coord = end_coord
        self.profile_name = profile_name 
        
        n_samp, n_tr = self.data.shape 
        self.times = np.arange(n_samp) * self.dt_ms
        
        if attribute_name == 'Instantaneous Phase':
            vmax = 180.0
            vmin = -180.0
            cmap_name = 'hsv' 
        elif attribute_name == 'Envelope':
            max_abs_val = np.nanmax(self.data)
            vmax = max_abs_val * (clip_percent / 100.0)
            vmin = 0.0
        else: # Original data
            if self.max_abs_amplitude == 1.0:
                self.max_abs_amplitude = np.nanmax(np.abs(self.data))
            vmax = self.max_abs_amplitude * (clip_percent / 100.0)
            vmin = -vmax
            if cmap_name not in STANDARD_CMAPS:
                 cmap_name = DEFAULT_CMAP

        
        self.im = self.ax.imshow(self.data, 
                               cmap=cmap_name, 
                               aspect='auto', 
                               extent=(0, n_tr, 0, self.times[-1]), 
                               vmin=vmin, 
                               vmax=vmax,
                               origin='upper') 
        
        self.ax.set_xlim(0, n_tr)
        self.ax.set_ylim(self.times[-1], 0) 
        
        self.ax.set_xlabel('Trace') 
        self.ax.set_ylabel('Time (ms)')
        self.ax.xaxis.set_label_coords(0.0, -0.05) 

        if self.cax.collections:
            self.cax.cla()
        
        cb = Colorbar(self.cax, self.im)
        self.cax.tick_params(labelsize=8) 
        self.cax.yaxis.set_ticks_position('right')
        self.cax.yaxis.set_label_position('right')
        
        if attribute_name == 'Instantaneous Phase':
             self.cax.set_ylabel('Phase (degrees)')

        self._draw_orientation_and_title(attribute_name)
        
        self.clear_info_texts() 

        self.draw()
        
    def _draw_orientation_and_title(self, attribute_name='None'):
        """Adds title, arrows, and cardinal directions to the plot."""
        
        title_text = self.profile_name
        if attribute_name != 'None':
             title_text += f" ({attribute_name})"
             
        self.ax.set_title(title_text, fontsize=12)

        if self.start_coord is None or self.end_coord is None:
            return

        bearing_start_to_end = get_bearing(self.start_coord, self.end_coord)
        bearing_end_to_start = get_bearing(self.end_coord, self.start_coord)

        dir_start = get_cardinal_direction(bearing_start_to_end)
        dir_end = get_cardinal_direction(bearing_end_to_start)
        
        text_left = f'\u2190 {dir_start}'
        self.ax.text(0.01, 1.02, text_left, 
                     transform=self.ax.transAxes, 
                     fontsize=9, color='black', ha='left', va='bottom',
                     fontweight='bold') 
        
        text_right = f'{dir_end} \u2192'
        self.ax.text(0.99, 1.02, text_right, 
                     transform=self.ax.transAxes, 
                     fontsize=9, color='black', ha='right', va='bottom',
                     fontweight='bold') 

    def on_cursor_move_update(self, event):
        """Updates the trace number, time, and amplitude values at the cursor position."""
        if event.inaxes != self.ax or self.data is None:
            self.clear_info_texts()
            return
            
        xdata = event.xdata  
        ydata = event.ydata 

        if xdata is None or ydata is None:
            self.clear_info_texts()
            return

        n_samp, n_tr = self.data.shape 
        
        trace_idx = int(round(xdata))
        sample_idx_display = int(round(ydata / self.dt_ms)) 
        
        if 0 <= trace_idx < n_tr and 0 <= sample_idx_display < n_samp:
            
            amplitude_value = self.data[sample_idx_display, trace_idx]
            
            time_ms = sample_idx_display * self.dt_ms 
            text_top = (f"Trace: {trace_idx} | "
                        f"Time: {time_ms:.1f} ms") 
                        
            self.info_text_top = self.update_info_text(
                self.info_text_top, text_top, x=0.95, y=0.88, ha='center', color='black') 

            text_bottom = f"Value: {amplitude_value:.2f}"
            
            self.info_text_bottom = self.update_info_text(
                self.info_text_bottom, text_bottom, x=0.95, y=0.12, ha='center', color='black') 
        
        else:
            self.clear_info_texts()
            
        self.draw_idle()

    def update_info_text(self, text_obj, new_text, x, y, ha, color):
        """Helper function to update or create a figure text object (fig.text)."""
        if text_obj:
            text_obj.set_text(new_text)
        else:
            text_obj = self.fig.text(x, y, new_text, ha=ha, va='center', fontsize=9, color=color, transform=self.fig.transFigure)
        return text_obj
        
    def clear_info_texts(self):
        """Clears info texts."""
        if self.info_text_top:
            self.info_text_top.set_text("")
        if self.info_text_bottom:
            self.info_text_bottom.set_text("")
            
    def on_scroll(self, event):
        if event.inaxes != self.ax: return 
        
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata

        scale_factor = 1.5 if event.button == 'up' else 1/1.5

        if xdata is not None:
            new_xlim = [
                xdata - (xdata - xlim[0]) * scale_factor,
                xdata + (xlim[1] - xdata) * scale_factor
            ]
            self.ax.set_xlim(new_xlim)
        
        if ydata is not None:
            # Note that the Y axis is inverted (time increases downwards)
            new_ylim = [
                ylim[0] + (ydata - ylim[0]) * (1-scale_factor), 
                ylim[1] - (ylim[1] - ydata) * (1-scale_factor)  
            ]
            self.ax.set_ylim(new_ylim[0], new_ylim[1])
            
        self.draw_idle()

    def on_press(self, event):
        if event.button == 2 and event.inaxes == self.ax: 
            self._drag_active = True
            self._last_x = event.xdata
            self._last_y = event.ydata
            self.figure.canvas.setCursor(QtCore.Qt.ClosedHandCursor) 
    
    def on_release(self, event):
        if event.button == 2:
            self._drag_active = False
            self.figure.canvas.setCursor(QtCore.Qt.ArrowCursor)
    
    def on_motion(self, event):
        if not self._drag_active or event.inaxes != self.ax: return

        if event.xdata is None or event.ydata is None: return

        dx = event.xdata - self._last_x
        dy = event.ydata - self._last_y

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim() 

        new_xlim = [cur_xlim[0] - dx, cur_xlim[1] - dx]
        
        new_ylim_bottom = cur_ylim[0] - dy 
        new_ylim_top = cur_ylim[1] - dy    

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim_bottom, new_ylim_top) 
        
        self.draw_idle()


# ==============================================================================
# 5. SEISMIC PLUGIN 
# ==============================================================================

class SeismicPlugin:
    
    def __init__(self, iface):
        self.iface = iface
        self.canvas = iface.mapCanvas()
        
        # --- ICON FIX: Using a QIcon from the provided PNG file path ---
        try:
            # ICON_PATH is now dynamically calculated
            icon = QIcon(ICON_PATH) 
            if icon.isNull():
                 # Fallback to a standard QGIS icon if the file path is invalid/missing
                 icon = iface.mainWindow().style().standardIcon(QStyle.SP_FileDialogDetailedView) 
        except Exception:
             icon = QIcon() # Final fallback: empty icon
        
        self.action = QAction(icon, 'PrezTraz', iface.mainWindow()) 
        self.action.triggered.connect(self.run)
        
        self.viewer_dock = None
        self.viewer_canvas = None 
        self.controls_group = None 
        self.attributes_group = None 
        
        self.dt_ms = 2.0
        self.seismic_layer = None 
        self.selection_connection = None 
        
        self.x_byte = DEFAULT_X_BYTE
        self.y_byte = DEFAULT_Y_BYTE
        self.segy_crs = QgsProject.instance().crs() 
        
        self.segy_handle = None 
        self.file_name = None 
        self.full_traces = None 
        self.launcher_dialog = None
        
        self.layer_id_to_monitor = None 
        
        self.current_profile_name = "No Profile Selected"
        self.current_start_coord = None 
        self.current_end_coord = None   


    def initGui(self):
        # Adding the action to the Plugin menu and Toolbar
        self.iface.addPluginToMenu('&SEGY Tools', self.action)
        self.iface.addToolBarIcon(self.action)
        
        QgsProject.instance().layersRemoved.connect(self.on_layers_removed)
        QgsProject.instance().layerLoaded.connect(self.on_layer_loaded)


    def unload(self):
        try:
            self.iface.removePluginMenu('&SEGY Tools', self.action)
            self.iface.removeToolBarIcon(self.action)
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
        if self.launcher_dialog is None:
            self.launcher_dialog = SeismicLauncherDialog(self, self.iface.mainWindow())
        self.launcher_dialog.exec_()
        
        self._ensure_layer_is_monitored()


    def _ensure_layer_is_monitored(self):
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
                self.iface.messageBar().pushMessage("Monitoring", f"Selection monitoring connected for layer: {GLOBAL_LAYER_NAME}. Select a line to view the profile.", level=Qgis.MessageLevel.Info, duration=3)
            else:
                self.seismic_layer = None


    def on_layers_removed(self, layerIds):
        if self.layer_id_to_monitor in layerIds:
            self.seismic_layer = None
            self.selection_connection = None
            self.layer_id_to_monitor = None
            self.full_traces = None
            if self.segy_handle: self.segy_handle.close()
            self.segy_handle = None
            if self.viewer_canvas: self.viewer_canvas.plot_seismic_image(None, profile_name=self.current_profile_name)
            self.iface.messageBar().pushMessage("Monitoring", f"Global SEGY layer has been removed.", level=Qgis.MessageLevel.Info, duration=2)

    
    def on_layer_loaded(self, layer):
        if layer.name() == GLOBAL_LAYER_NAME and isinstance(layer, QgsVectorLayer):
            self._ensure_layer_is_monitored()


    def open_segy_file_dialog(self):
        fnames, _ = QFileDialog.getOpenFileNames(
            self.iface.mainWindow(), 
            'Open SEG-Y 2D profile(s)', 
            '', 
            'SEG-Y files (*.sgy *.segy)'
        )
        if not fnames: return

        dialog = SegyLoadDialog(self.iface.mainWindow())
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
            
        x_byte, y_byte = dialog.get_coords_bytes()
        self.x_byte = x_byte
        self.y_byte = y_byte
        # Retrieve the selected CRS
        self.segy_crs = dialog.get_selected_crs()
        
        if not self.segy_crs.isValid():
             self.iface.messageBar().pushCritical("CRS Error", "The selected CRS is invalid.")
             return

        self._create_or_get_seismic_layer()
        if not self.seismic_layer: 
            return
        
        count = 0
        
        for i, fname in enumerate(fnames):
            self.iface.statusBarIface().showMessage(f"Processing file {i+1}/{len(fnames)}: {os.path.basename(fname)}...")
            
            new_feature_id = self.add_feature_to_seismic_layer(fname)
            
            if new_feature_id is not None:
                count += 1
            else:
                continue 
        
        self.iface.statusBarIface().clearMessage()
        
        if count > 0:
            self.iface.messageBar().pushMessage("Finished", 
                                          f"Successfully loaded {count} profiles as lines (CRS: {self.segy_crs.authid()}). Select a line to display the SEGY profile.", 
                                          level=Qgis.MessageLevel.Info, 
                                          duration=5)
        else:
            self.iface.messageBar().pushWarning("No Data", "None of the selected SEGY files were loaded correctly.")


    def _manual_load_and_display(self, feature_id):
        
        request = QgsFeatureRequest().setFilterFid(feature_id)
        feature_to_load = next(self.seismic_layer.getFeatures(request), None)

        if not feature_to_load:
            self.iface.messageBar().pushCritical("Internal Error", "Could not find feature by ID.")
            return

        path = feature_to_load['segy_path']
        x_byte = feature_to_load['x_byte']
        y_byte = feature_to_load['y_byte']
        profile_name = feature_to_load['profile_name']
        
        self.current_profile_name = profile_name
        self.current_start_coord = (feature_to_load['start_x'], feature_to_load['start_y'])
        self.current_end_coord = (feature_to_load['end_x'], feature_to_load['end_y'])
        
        if path == self.file_name and self.full_traces is not None:
            self.update_viewer_with_current_data()
            return

        self.file_name = path
        self.x_byte = x_byte
        self.y_byte = y_byte
        
        if self.segy_handle: self.segy_handle.close()
        
        self.iface.statusBarIface().showMessage(f"Loading seismic data: {profile_name}...")
        
        # Store original values
        original_x = segyio.TraceField.SourceX
        original_y = segyio.TraceField.SourceY
        
        try:
            # Modify mapping before opening
            segyio.TraceField.SourceX = self.x_byte
            segyio.TraceField.SourceY = self.y_byte

            f = segyio.open(self.file_name, strict=False)
            self.segy_handle = f 
            
            try:
                # Dividing by 1000.0 gives the result in ms
                self.dt_ms = self.segy_handle.bin[segyio.BinField.Interval] / 1000.0
            except Exception:
                self.dt_ms = 2.0 
            
            self.full_traces = self.load_all_traces() 
            
            if self.full_traces is None:
                raise Exception("Failed to load trace data.")

            if self.viewer_canvas:
                self.viewer_canvas.max_abs_amplitude = 1.0 
            
            # Ensure the dock is open before updating
            self.open_seismic_viewer_dock()
            self.update_viewer_with_current_data()
            
            self.iface.messageBar().pushMessage("Visualization", f"Displaying profile: {profile_name}.", level=Qgis.MessageLevel.Info, duration=3) 

        except Exception as e:
            self.iface.messageBar().pushCritical("SEGY Load Error", f"Error loading file {profile_name}: {e}\n{traceback.format_exc()}")
            self.full_traces = None
            if self.segy_handle: self.segy_handle.close()
            self.segy_handle = None
            if self.viewer_canvas: 
                # Ensure the dock is open before clearing
                self.open_seismic_viewer_dock()
                self.viewer_canvas.plot_seismic_image(None, profile_name=self.current_profile_name) 

        finally:
            # Restore original mapping values
            segyio.TraceField.SourceX = original_x
            segyio.TraceField.SourceY = original_y
            
        self.iface.statusBarIface().clearMessage()


    def add_feature_to_seismic_layer(self, fname):
        if not self.seismic_layer: return None
        
        temp_segy_handle = None 
        
        # Store original values
        original_x = segyio.TraceField.SourceX
        original_y = segyio.TraceField.SourceY
        
        try:
            # Modify mapping before opening
            segyio.TraceField.SourceX = self.x_byte
            segyio.TraceField.SourceY = self.y_byte
            
            temp_segy_handle = segyio.open(fname, strict=False)
            f = temp_segy_handle
            
        except Exception as e:
            self.iface.messageBar().pushCritical("SEGYIO Error", f"Failed to open and configure file {os.path.basename(fname)}: {e}")
            return None
        finally:
             # Restore original mapping values
             segyio.TraceField.SourceX = original_x
             segyio.TraceField.SourceY = original_y


        try:
            f = temp_segy_handle
            n_traces = f.tracecount
            sx = f.attributes(segyio.TraceField.SourceX)[:]
            sy = f.attributes(segyio.TraceField.SourceY)[:]

        except Exception as e:
            self.iface.messageBar().pushCritical("SEGYIO Error", f"Failed to load SEGY headers for {os.path.basename(fname)}: {e}")
            return None
        finally:
             if temp_segy_handle: temp_segy_handle.close()

        start_x, start_y = float(sx[0]), float(sy[0])
        end_x, end_y = float(sx[-1]), float(sy[-1])
        
        # The vector layer stores data in its own CRS, and QGIS automatically transforms it to the project CRS.
        # It is important to set the geometry coordinates in the layer's CRS.
        all_points = [QgsPointXY(float(sx[i]), float(sy[i])) for i in range(n_traces)]
        
        prov = self.seismic_layer.dataProvider()
        
        line_feature = QgsFeature()
        line_feature.setGeometry(QgsGeometry.fromPolylineXY(all_points))
        line_feature.setFields(self.seismic_layer.fields()) 

        base_name = os.path.basename(fname)
        line_feature['profile_name'] = base_name.split('.')[0]
        line_feature['traces_count'] = n_traces
        line_feature['segy_path'] = fname
        line_feature['x_byte'] = self.x_byte
        line_feature['y_byte'] = self.y_byte
        line_feature['start_x'] = start_x
        line_feature['start_y'] = start_y
        line_feature['end_x'] = end_x
        line_feature['end_y'] = end_y
        
        if prov.addFeatures([line_feature]):
            new_feature_id = line_feature.id()
            self.seismic_layer.updateExtents()
            self.iface.mapCanvas().refresh()
            return new_feature_id
        
        return None

    def _create_or_get_seismic_layer(self):
        self._ensure_layer_is_monitored()
        
        # Check if the existing layer has the same CRS as the selected one
        if self.seismic_layer:
             if self.seismic_layer.crs().authid() == self.segy_crs.authid(): # Compare by authid
                  return
             else:
                  # If CRS is different, remove the old layer and create a new one.
                  QgsProject.instance().removeMapLayer(self.seismic_layer.id())
                  self.seismic_layer = None
                  self.selection_connection = None
                  self.layer_id_to_monitor = None
                  self.iface.messageBar().pushWarning("Warning", "CRS change detected. The previous profile layer was removed to use the new CRS.")


        # Use the selected CRS!
        layer = QgsVectorLayer(f'LineString?crs={self.segy_crs.authid()}', GLOBAL_LAYER_NAME, 'memory') 
        prov = layer.dataProvider()
        
        fields = QgsFields()
        fields.append(QgsField('profile_name', QtCore.QVariant.String))
        fields.append(QgsField('traces_count', QtCore.QVariant.Int))
        fields.append(QgsField('segy_path', QtCore.QVariant.String))
        fields.append(QgsField('x_byte', QtCore.QVariant.Int))
        fields.append(QgsField('y_byte', QtCore.QVariant.Int))
        fields.append(QgsField('start_x', QtCore.QVariant.Double))
        fields.append(QgsField('start_y', QtCore.QVariant.Double))
        fields.append(QgsField('end_x', QtCore.QVariant.Double))
        fields.append(QgsField('end_y', QtCore.QVariant.Double))
        
        if not prov.addAttributes(fields):
            self.iface.messageBar().pushCritical("Error", "Failed to add attributes to the global layer.")
            return
            
        layer.updateFields() 
        QgsProject.instance().addMapLayer(layer)
        self.seismic_layer = layer
        
        self._ensure_layer_is_monitored() 


    def on_profile_selection(self, selected_ids, deselected_ids):
        if not self.seismic_layer: return
        
        selected_features = self.seismic_layer.selectedFeatures()
        
        if not selected_features:
            self.current_profile_name = "No Profile Selected"
            self.current_start_coord = None
            self.current_end_coord = None
            if self.full_traces is not None or self.viewer_canvas is not None:
                self.iface.messageBar().pushMessage("Visualization", "No selection. SEGY view cleared.", level=Qgis.MessageLevel.Info, duration=2)
                self.full_traces = None
                if self.segy_handle: self.segy_handle.close()
                self.segy_handle = None
                if self.viewer_canvas: 
                     # Ensure the dock is open before clearing
                     self.open_seismic_viewer_dock()
                     self.viewer_canvas.plot_seismic_image(None, profile_name=self.current_profile_name) 
            return

        feature_to_load = selected_features[0]
        self._manual_load_and_display(feature_to_load.id())


    def update_viewer_with_current_data(self):
        """Updates the viewer with current Cmap, Clipping, and Attribute settings."""

        # Ensure the dock is always open if we have data or want to load it
        self.open_seismic_viewer_dock()
        
        if self.full_traces is None or self.viewer_canvas is None:
             if self.viewer_canvas:
                 self.viewer_canvas.plot_seismic_image(None, profile_name=self.current_profile_name)
             return
             
        if self.viewer_dock:
            self.viewer_dock.show()

        # Find controls in the Dock (now it's safer as the dock is already initialized)
        clip_slider = self.viewer_dock.findChild(QtWidgets.QSlider)
        cmap_combo = self.viewer_dock.findChild(QComboBox, 'cmap_combo')
        attribute_combo = self.viewer_dock.findChild(QComboBox, 'attribute_combo') 

        clip_percent = 100.0
        cmap_name = DEFAULT_CMAP
        attribute_name = 'None'

        if clip_slider:
            clip_percent = float(clip_slider.value())
        if cmap_combo:
            cmap_name = cmap_combo.currentText()
        if attribute_combo:
            attribute_name = attribute_combo.currentText()

        self.viewer_canvas.plot_seismic_image(self.full_traces,
                                              dt_ms=self.dt_ms, 
                                              clip_percent=clip_percent,
                                              cmap_name=cmap_name,
                                              start_coord=self.current_start_coord, 
                                              end_coord=self.current_end_coord,
                                              profile_name=self.current_profile_name,
                                              attribute_name=attribute_name)


    def load_all_traces(self):
        if not self.segy_handle: return None
        try:
            f = self.segy_handle
            n_traces = f.tracecount
            
            try:
                 data = f.trace.array().T # Transpose to get (Samples, Traces)
            except Exception:
                 # Fallback to stacking in a loop
                 data = np.stack([f.trace[i] for i in range(n_traces)], axis=1) 
                 
            return data
        except Exception as e:
            self.iface.messageBar().pushCritical("I/O Error", f"Error loading the entire data block: {e}\n{traceback.format_exc()}")
            return None


    def open_seismic_viewer_dock(self):
        """Creates the dock and adds the Attributes and Settings panel in a horizontal layout."""
        if self.viewer_dock is None:
            dock = QDockWidget('Seismic Image Viewer', self.iface.mainWindow())
            dock.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
            
            dock.setMinimumWidth(400) 
            
            main_widget = QtWidgets.QWidget()
            main_vbox = QtWidgets.QVBoxLayout(main_widget)
            
            self.viewer_canvas = SeismicImageCanvas(parent=main_widget) 
            main_vbox.addWidget(self.viewer_canvas)
            
            # --- Horizontal Container (for both groups) ---
            controls_container = QWidget()
            controls_layout = QHBoxLayout(controls_container)
            controls_layout.setContentsMargins(0, 0, 0, 0)

            # -----------------------------------------------------------------
            # 1. Attributes Panel
            # -----------------------------------------------------------------
            
            attr_widget = QWidget()
            attr_vbox = QVBoxLayout(attr_widget)
            attr_vbox.setContentsMargins(0, 0, 0, 0)
            
            toggle_button_attr = QToolButton()
            toggle_button_attr.setText(">> Attributes")
            toggle_button_attr.setCheckable(True)
            toggle_button_attr.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            
            self.attributes_group = QGroupBox("Attribute Selection")
            attr_group_layout = QVBoxLayout(self.attributes_group)
            self.attributes_group.setMaximumHeight(0) 
            self.attributes_group.setContentsMargins(5, 5, 5, 5)
            self.attributes_group.setFlat(True)

            attribute_combo = QComboBox()
            attribute_combo.setObjectName('attribute_combo') 
            attribute_combo.addItems(['None', 'Envelope', 'Instantaneous Phase'])
            attribute_combo.setToolTip("Select Seismic Attribute to display")
            attribute_combo.currentTextChanged.connect(self.update_viewer_with_current_data)
            
            attr_group_layout.addWidget(QLabel("Seismic Attribute:"))
            attr_group_layout.addWidget(attribute_combo)
            
            def toggle_attr_group(checked):
                # Dynamically set height
                self.attributes_group.setMaximumHeight(80 if checked else 0)   
                self.attributes_group.setFlat(not checked)

            toggle_button_attr.clicked.connect(toggle_attr_group)
            
            attr_vbox.addWidget(toggle_button_attr)
            attr_vbox.addWidget(self.attributes_group)
            
            # -----------------------------------------------------------------
            # 2. Visualization Settings Panel
            # -----------------------------------------------------------------
            
            vis_widget = QWidget()
            vis_vbox = QVBoxLayout(vis_widget)
            vis_vbox.setContentsMargins(0, 0, 0, 0)
            
            toggle_button_vis = QToolButton()
            toggle_button_vis.setText(">> Visualization Settings")
            toggle_button_vis.setCheckable(True)
            toggle_button_vis.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)

            self.controls_group = QGroupBox("Visualization Settings")
            controls_group_layout = QVBoxLayout(self.controls_group)
            self.controls_group.setMaximumHeight(0) 
            self.controls_group.setContentsMargins(5, 5, 5, 5)
            self.controls_group.setFlat(True)
            
            cmap_combo = QComboBox()
            cmap_combo.setObjectName('cmap_combo') 
            cmap_combo.addItems(STANDARD_CMAPS)
            cmap_combo.setCurrentText(DEFAULT_CMAP) 
            cmap_combo.setToolTip("Select Colormap")
            cmap_combo.currentTextChanged.connect(self.update_viewer_with_current_data)
            
            clip_slider = QSlider(QtCore.Qt.Horizontal)
            clip_slider.setMinimum(1)  
            clip_slider.setMaximum(100)
            clip_slider.setValue(100)
            clip_slider.setToolTip("Amplitude Clipping (%): 100% to Max Abs")
            clip_slider.valueChanged.connect(self.update_viewer_with_current_data)
            
            controls_group_layout.addWidget(QLabel("Colormap:"))
            controls_group_layout.addWidget(cmap_combo)
            controls_group_layout.addWidget(QLabel("Amplitude Clipping (%):"))
            controls_group_layout.addWidget(clip_slider)
            
            def toggle_vis_group(checked):
                # Dynamically set height
                self.controls_group.setMaximumHeight(150 if checked else 0)   
                self.controls_group.setFlat(not checked)

            toggle_button_vis.clicked.connect(toggle_vis_group)
            
            vis_vbox.addWidget(toggle_button_vis)
            vis_vbox.addWidget(self.controls_group)
            
            # --- Add both panels to the horizontal container ---
            controls_layout.addWidget(attr_widget)
            controls_layout.addWidget(vis_widget)
            
            # --- Add the horizontal container to the main vertical layout ---
            main_vbox.addWidget(controls_container)
            
            dock.setWidget(main_widget)
            self.iface.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock) 
            self.viewer_dock = dock

        self.viewer_dock.show()