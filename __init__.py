# __init__.py

# Zapewnia, że QGIS może znaleźć klasę główną w pliku main.py
from .main import SeismicPlugin 

def classFactory(iface):
    """
    QGIS wywołuje tę funkcję, aby utworzyć instancję wtyczki.
    Argument 'iface' jest przekazywany do konstruktora klasy SeismicPlugin.
    """
    return SeismicPlugin(iface)