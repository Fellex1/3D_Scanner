"""=======TODO-Liste v0.6=======
Objekt-Detection muss verbessert werden
Lade-Dialog schöner gestalten
Fehlerbehandlung bei Kamerazugriff
SAP-Integration                 (Platzhalter-Button/optional)
Lokal speichern Integration     (Platzhalter-Button)
GUI-Design verbessern           (optional)
================================"""

import sys
import cv2
import os
import numpy as np
import threading
import logging
import platform
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
import json

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QFrame, QSizePolicy, QStackedWidget, QScrollArea, 
    QToolButton, QMessageBox, QDialog, QGridLayout, QProgressBar
)
from PyQt6.QtGui import QPixmap, QIcon, QKeySequence, QShortcut, QMovie, QImage
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal, QObject, QRunnable, QThreadPool

# ==================== Konfiguration ====================
@dataclass
class AppConfig:
    """Konfigurationsklasse für die Anwendung"""
    NUM_CAMERAS: int = 4
    IMAGE_WIDTH: int = 640
    IMAGE_HEIGHT: int = 480
    DEBUG_SINGLE_CAMERA: bool = True
    DEFAULT_LANGUAGE: str = "de"
    GUI_RESOURCES_PATH: str = "GUI_Anzeige"
    LOG_LEVEL: str = "INFO"
    YOLO_MODEL_PATH: str = "YOLOV8s_Barcode_Detection.pt"
    
    @classmethod
    def load_from_file(cls, config_path: str = "config.json"):
        """Lädt Konfiguration aus JSON-Datei"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    return cls(**config_data)
        except Exception as e:
            logging.warning(f"Konfigurationsdatei konnte nicht geladen werden: {e}")
        return cls()

# ==================== Logging Setup ====================
def setup_logging(level: str = "INFO"):
    """Initialisiert das Logging-System"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('3d_scanner.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==================== Globale Konstante ====================
CONFIG = AppConfig.load_from_file()
logger = setup_logging(CONFIG.LOG_LEVEL)

# ==================== Translation Manager ====================
class TranslationManager:
    """Verwaltet mehrsprachige Texte"""
    
    def __init__(self):
        # Struktur: (Deutsch, Englisch, Italienisch)
        self.translations = {
            "start": {
                "title": ("3D-Scanner Interface", "3D Scanner Interface", "Interfaccia Scanner 3D"),
                "subtitle": ("Interface um den 3D-Scanner zu bedienen", "Interface to operate the 3D scanner", "Interfaccia per gestire lo scanner 3D"),
                "instruction1": ("Bitte lege den Artikel der gescannt werden soll in die Box ein", "Please place the item to be scanned in the box", "Si prega di posizionare l'articolo nella scatola"),
                "instruction2": ("Stellen Sie sicher, dass der Artikel vollständig im Sichtfeld aller Kameras liegt", "Make sure the item is completely in the field of view of all cameras", "Assicurarsi che l'articolo sia completamente nel campo visivo di tutte le telecamere"),
                "instruction3": ("Maximale Größe: 50x50x50 cm", "Maximum size: 50x50x50 cm", "Dimensione massima: 50x50x50 cm"),
                "instruction4": ("Maximales Gewicht: 20 kg", "Maximum weight: 20 kg", "Peso massimo: 20 kg"),
                "scan_btn": ("Scan Starten", "Start Scan", "Avvia Scan"),
                "save_btn": ("Lokal speichern", "Save Locally", "Salva localmente"),
                "status_title": ("System Status", "System Status", "Stato del Sistema"),
                "camera_status": ("Kamera System", "Camera System", "Sistema Fotocamera"),
                "light_status": ("Beleuchtung", "Lighting", "Illuminazione"),
                "measure_status": ("Mess-System", "Measurement System", "Sistema di Misura"),
                "scale_status": ("Waage", "Scale", "Bilancia"),
                "storage_status": ("Speicher", "Storage", "Memoria"),
                "ready": ("Bereit", "Ready", "Pronto"),
                "active": ("Aktiv", "Active", "Attivo"),
                "calibrated": ("Kalibriert", "Calibrated", "Calibrato"),
                "connected": ("Verbunden", "Connected", "Connesso"),
                "available": ("Verfügbar", "Available", "Disponibile"),
                "refresh_btn": ("Status aktualisieren", "Refresh Status", "Aggiorna Stato"),
                "check_camera": ("Kamera prüfen", "Check Camera", "Controlla Fotocamera"),
                "check_light": ("Beleuchtung prüfen", "Check Lighting", "Controlla Illuminazione"),
                "check_measure": ("Mess-System prüfen", "Check Measurement", "Controlla Sistema Misura"),
                "calibrate_scale": ("Waage kalibrieren", "Calibrate Scale", "Calibra Bilancia"),
                "check_storage": ("Speicher prüfen", "Check Storage", "Controlla Memoria")
            },
            "photo": {
                "title": ("Foto-Auswahl", "Photo Selection", "Selezione Foto"),
                "retry_btn": ("Wiederholen", "Retake", "Ripeti"),
                "discard_btn": ("Verwerfen", "Discard", "Scarta")
            },
            "overview": {
                "title": ("Kamera-Übersicht", "Camera Overview", "Panoramica Fotocamera"),
                "dimensions": ("Abmessungen:", "Dimensions:", "Dimensioni:"),
                "weight": ("Gewicht:", "Weight:", "Peso:"),
                "mm": ("mm", "mm", "mm"),
                "kg": ("kg", "kg", "kg")
            },
            "storage": {
                "title": ("Speicher Option", "Storage Options", "Opzioni di Memorizzazione"),
                "barcode_label": ("Ausgewerteter Barcode:", "Barcode:", "Barcode:"),
                "barcode_type": ("Barcode-Typ:", "Barcode Type:", "Tipo di barcode:"),
                "sap_btn": ("SAP-Eintrag", "SAP Entry", "SAP Entry"),
                "save_btn": ("Lokal speichern", "Save Locally", "Salva localmente")
            }
        }
        
        # Sprach-Mapping: 0=Deutsch, 1=Englisch, 2=Italienisch
        self.language_map: Dict[str, int] = {"de": 0, "en": 1, "it": 2}
    
    def get_text(self, language: str, page: str, key: str) -> str:
        """Holt übersetzten Text für gegebene Sprache, Seite und Schlüssel"""
        lang_index = self.language_map.get(language, 0)  # Default zu Deutsch
        page_dict = self.translations.get(page, {})
        text_tuple = page_dict.get(key, ("[FEHLER]", "[ERROR]", "[ERRORE]"))
        
        # Sicherstellen, dass wir immer einen String zurückgeben
        if isinstance(text_tuple, tuple) and len(text_tuple) > lang_index:
            return text_tuple[lang_index]
        return f"[{key}]"

# ==================== Camera Manager ====================
class CameraManager:
    """Verwaltet Kamerazugriff und Bildaufnahme"""
    
    def __init__(self, debug_single_camera: bool = CONFIG.DEBUG_SINGLE_CAMERA):
        self.debug_single_camera = debug_single_camera
        self.available_cameras = self._find_cameras()
        logger.info(f"Verfügbare Kameras gefunden: {self.available_cameras}")
    
    def _get_camera_backend(self) -> int:
        """Bestimmt den passenden Camera Backend für das Betriebssystem"""
        if platform.system() == "Windows":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY
    
    def _find_cameras(self) -> List[int]:
        """Findet verfügbare Kameras"""
        available: List[int] = []
        backend = self._get_camera_backend()
        
        for i in range(CONFIG.NUM_CAMERAS):
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
                else:
                    cap.release()
            except Exception as e:
                logger.warning(f"Fehler beim Zugriff auf Kamera {i}: {e}")
        
        return available
    
    def _enable_flash(self):
        """Aktiviert den Blitz (Platzhalter)"""
        # BLITZ-IMPLEMENTIERUNG HIER EINFÜGEN
        # GPIO, serielle Schnittstelle
        pass
    
    def _disable_flash(self):
        """Deaktiviert den Blitz (Platzhalter)"""
        pass
    
    def _make_placeholder(self, camera_id: int = -1) -> np.ndarray:
        """Erstellt ein Platzhalterbild für fehlende Kameras"""
        img = np.zeros((CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, 3), dtype=np.uint8)
        text = f"Kamera {camera_id} nicht verfügbar" if camera_id >= 0 else "BILD NICHT AUFGENOMMEN"
        cv2.putText(img, text, 
                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (255, 255, 255), 2)
        return img
    
    def take_picture(self, camera_id: int) -> np.ndarray:
        """Nimmt ein Bild mit der angegebenen Kamera auf"""
        if camera_id not in self.available_cameras:
            logger.warning(f"Kamera {camera_id} nicht verfügbar")
            return self._make_placeholder(camera_id)
        
        backend = self._get_camera_backend()
        
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            if not cap.isOpened():
                logger.error(f"Kamera {camera_id} konnte nicht geöffnet werden")
                cap.release()
                return self._make_placeholder(camera_id)
            
            self._enable_flash()
            ret, frame = cap.read()
            self._disable_flash()
            cap.release()
            
            if ret:
                logger.info(f"Bild erfolgreich von Kamera {camera_id} aufgenommen")
                return frame
            else:
                logger.error(f"Bildaufnahme von Kamera {camera_id} fehlgeschlagen")
                return self._make_placeholder(camera_id)
                
        except Exception as e:
            logger.error(f"Fehler bei Bildaufnahme von Kamera {camera_id}: {e}")
            return self._make_placeholder(camera_id)
        
    def take_all_pictures(self) -> List[np.ndarray]:
        """Nimmt Bilder von allen Kameras auf"""
        images: List[np.ndarray] = []
        
        if self.debug_single_camera:
            # Debug: Eine Kamera für alle Bilder
            logger.debug("Debug-Modus: Verwende eine Kamera für alle Bilder")
            for i in range(CONFIG.NUM_CAMERAS):
                img = self.take_picture(0)
                images.append(img)
        else:
            # Normal: Jede Kamera macht ein Bild
            for i in range(CONFIG.NUM_CAMERAS):
                if i < len(self.available_cameras):
                    img = self.take_picture(i)
                else:
                    img = self._make_placeholder(i)
                images.append(img)
        
        return images

# ==================== Detection Manager ====================
class DetectionManager:
    def __init__(self):
        self.yolo_model = None
        self.barcode_model = None
        
    def load_yolo_model(self, model_path: str = CONFIG.YOLO_MODEL_PATH):
        """Lädt das YOLO-Modell (einmalig)"""
        try:
            from ultralytics import YOLO
            self.barcode_model = YOLO(model_path)
            logger.info(f"YOLO-Modell geladen von {model_path}")
        except ImportError as e:
            logger.error(f"YOLO nicht verfügbar: {e}")
        except Exception as e:
            logger.error(f"Fehler beim Laden des YOLO-Modells: {e}")
    
    def run_yolo_detection(self, images: List[np.ndarray]) -> Tuple[List[str], List[np.ndarray]]:
        """Führt YOLO-Objekterkennung durch"""
        all_dimensions: List[str] = []
        all_frames: List[np.ndarray] = []
        
        # Direkt importieren und verwenden
        try:
            # Importiere das Modul neu
            import BoundingBox_Yolo03 as yolo_module
            logger.info("BoundingBox_Yolo03 erfolgreich importiert")
        except ImportError as e:
            logger.error(f"YOLO-Modul nicht gefunden: {e}")
            # Fallback: Leere Ergebnisse
            for _ in range(len(images)):
                all_dimensions.append("0 x 0")
                all_frames.append(None)
            return all_dimensions, all_frames
        
        for idx, frame in enumerate(images):
            logger.info(f"Verarbeite Bild {idx} für YOLO-Erkennung")
            
            if frame is None:
                logger.warning(f"Bild {idx} ist None")
                all_dimensions.append("0 x 0")
                all_frames.append(None)
                continue

            try:
                # Stelle sicher, dass das Bild das richtige Format hat
                if frame is not None and isinstance(frame, np.ndarray):
                    logger.debug(f"Bild {idx} - Größe: {frame.shape}, Typ: {frame.dtype}")
                    
                    # WICHTIG: Konvertiere BGR zu RGB für YOLO
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame
                    
                    # Führe YOLO-Erkennung durch
                    boxes_info = yolo_module.get_boxes_and_dimensions(frame_rgb)
                    logger.debug(f"Bild {idx}: {len(boxes_info)} Boxen erkannt")

                    if boxes_info:
                        dim_str = f"{boxes_info[0]['width']} x {boxes_info[0]['height']}"
                        logger.info(f"Bild {idx}: YOLO erkannt - {dim_str}")
                    else:
                        dim_str = "0 x 0"
                        logger.warning(f"Bild {idx}: Keine Boxen erkannt")

                    all_dimensions.append(dim_str)

                    # Annotiertes Bild erstellen
                    try:
                        # Verwende das Original-BGR-Bild für die Annotation
                        frame_with_boxes = yolo_module.draw_boxes(frame.copy(), boxes_info)
                        all_frames.append(frame_with_boxes)
                        logger.debug(f"Bild {idx}: Annotiertes Frame erstellt")
                    except Exception as draw_e:
                        logger.error(f"Fehler beim Zeichnen der Boxen Bild {idx}: {draw_e}")
                        # Fallback: Originalbild
                        all_frames.append(frame)
                        
                else:
                    logger.error(f"Bild {idx} hat ungültiges Format: {type(frame)}")
                    all_dimensions.append("0 x 0")
                    all_frames.append(None)
                
            except Exception as e:
                logger.error(f"Fehler bei YOLO-Erkennung Bild {idx}: {e}", exc_info=True)
                all_dimensions.append("0 x 0")
                all_frames.append(None)
                
        logger.info(f"YOLO-Erkennung abgeschlossen. Dimensionen: {all_dimensions}")
        return all_dimensions, all_frames
    
    def run_barcode_detection(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Erkennt Barcodes in den Bildern"""
        results: List[Dict[str, Any]] = []
        
        try:
            from BarCode_v02 import process_roi
        except ImportError as e:
            logger.error(f"Barcode-Modul nicht gefunden: {e}")
            for idx in range(len(images)):
                results.append({"index": idx, "found": False, "error": "Modul nicht verfügbar"})
            return results
        
        if self.barcode_model is None:
            self.load_yolo_model()
        
        for idx, img in enumerate(images):
            if img is None:
                results.append({"index": idx, "found": False})
                continue

            try:
                found = False
                decoded_value = None
                decoded_type = None

                if self.barcode_model:
                    model_results = self.barcode_model.predict(img)

                    for r in model_results:
                        for box in r.boxes.xyxy:
                            x1, y1, x2, y2 = map(int, box)
                            roi = img[y1:y2, x1:x2]
                            result = process_roi(roi, f"image_{idx}")
                            if result["found"]:
                                found = True
                                decoded_value = result["value"]
                                decoded_type = result["type"]
                                break

                        if found:
                            break

                results.append({
                    "index": idx,
                    "found": found,
                    "value": decoded_value,
                    "type": decoded_type
                })
                
            except Exception as e:
                logger.error(f"Fehler bei Barcode-Erkennung Bild {idx}: {e}")
                results.append({"index": idx, "found": False, "error": str(e)})
        
        return results

# ==================== Parallel Worker ====================
class ParallelWorker(QThread):   
    output_received = pyqtSignal(str, object)  # (task_name, result)
    progress_updated = pyqtSignal(int)  # Fortschritt in %
    finished = pyqtSignal()

    def __init__(self, images: List[np.ndarray]):
        super().__init__()
        self.images = images
        self.detection_manager = DetectionManager()
        self.progress = 0

    def _update_progress(self, increment: int):
        """Aktualisiert den Fortschritt"""
        self.progress += increment
        self.progress_updated.emit(self.progress)

    def run(self):
        """Führt parallele Verarbeitung mit ThreadPoolExecutor durch"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Alle Tasks parallel starten
            futures = {
                executor.submit(self._run_yolo_task): "yolo",
                executor.submit(self._run_barcode_task): "barcode",
                executor.submit(self._run_weight_task): "weight"
            }
            
            completed = 0
            total = len(futures)
            
            # Auf alle Futures warten und Ergebnisse verarbeiten
            for future in concurrent.futures.as_completed(futures):
                task_type = futures[future]
                try:
                    result = future.result()
                    self._process_result(task_type, result)
                except Exception as e:
                    logger.error(f"Fehler in {task_type} Task: {e}")
                
                completed += 1
                progress = int((completed / total) * 100)
                self.progress_updated.emit(progress)
        
        self.finished.emit()
    
    def _run_yolo_task(self):
        """Führt YOLO-Erkennung durch"""
        try:
            dimensions, frames = self.detection_manager.run_yolo_detection(self.images)
            return {"dimensions": dimensions, "frames": frames}
        except Exception as e:
            logger.error(f"YOLO Task Fehler: {e}")
            return {"dimensions": [], "frames": []}
    
    def _run_barcode_task(self):
        """Führt Barcode-Erkennung durch"""
        try:
            barcodes = self.detection_manager.run_barcode_detection(self.images)
            return {"barcodes": barcodes}
        except Exception as e:
            logger.error(f"Barcode Task Fehler: {e}")
            return {"barcodes": []}
    
    def _run_weight_task(self):
        """Führt Gewichtsmessung durch"""
        try:
            import Gewichts_Messung
            weight = Gewichts_Messung.get_weight()
            return {"weight": weight}
        except ImportError as e:
            logger.error(f"Gewichtsmodul nicht verfügbar: {e}")
            return {"weight": "Undefiniert"}
        except Exception as e:
            logger.error(f"Gewicht Task Fehler: {e}")
            return {"weight": "Undefiniert"}
    
    def _process_result(self, task_type: str, result: dict):
        """Verarbeitet Ergebnisse der Tasks"""
        if task_type == "yolo":
            dimensions = result.get("dimensions", [])
            frames = result.get("frames", [])
            self.output_received.emit("Abmessung", dimensions)
            self.output_received.emit("yolo_frames", frames)
        elif task_type == "barcode":
            barcodes = result.get("barcodes", [])
            for barcode in barcodes:
                self.output_received.emit("barcode", barcode)
        elif task_type == "weight":
            weight = result.get("weight", "Undefiniert")
            self.output_received.emit("weight", weight)



# ==================== Main Application ====================
class FullscreenApp(QMainWindow):
    """Hauptanwendung für den 3D-Scanner"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D-Scanner")
        self.showFullScreen()

        # Initialisierung
        self.camera = CameraManager(debug_single_camera=True)
        self.translator = TranslationManager()
        self.language = CONFIG.DEFAULT_LANGUAGE
        self.Explorer_Structure = CONFIG.GUI_RESOURCES_PATH

        # Datenvariablen
        self.abmessung: Optional[str] = None
        self.gewicht: Optional[str] = None
        self.barcode: Optional[str] = None
        self.barcode_type: Optional[str] = None

        self.images: List[Optional[np.ndarray]] = [None] * CONFIG.NUM_CAMERAS
        self.image_labels: List[Optional[QLabel]] = [None] * CONFIG.NUM_CAMERAS
        self.final_images: List[Optional[np.ndarray]] = [None] * CONFIG.NUM_CAMERAS
        self.final_image_labels: List[Optional[QLabel]] = [None] * CONFIG.NUM_CAMERAS

        self.keep: List[bool] = [True] * CONFIG.NUM_CAMERAS
        self.scan_start = False
        self.bilder_namen = ["iso_Bild", "top_Bild", "right_Bild", "behind_Bild"]

        # GUI Setup
        self._setup_ui()
        self.load_pages()
        self.update_buttons()

    def _setup_ui(self):
        """Initialisiert die Benutzeroberfläche"""
        container = QWidget()
        container.setStyleSheet("background-color: #292929;")
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 20, 20, 10)
        main_layout.setSpacing(10)
        self.setCentralWidget(container)

        # Stacked widget für die Seiten
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, stretch=1)

        # Navigation-Buttons
        bar_layout = QHBoxLayout()
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(10)
        
        self.back_btn = QPushButton("←")
        self.next_btn = QPushButton("→")
        self.back_btn.setFixedSize(100, 50)
        self.next_btn.setFixedSize(100, 50)
        
        font = self.back_btn.font()
        font.setPointSize(26)
        self.back_btn.setFont(font)
        self.next_btn.setFont(font)
        
        self.back_btn.clicked.connect(self.go_back)
        self.next_btn.clicked.connect(self.go_next)

        # Navigation-Buttons Styling
        nav_style = """
            QPushButton {
                font-size: 26px;
                font-weight: bold;
                background: #3498db;
                color: #ecf0f1;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background: #2980b9;
            }
            QPushButton:pressed {
                background: #21618c;
            }
        """
        self.back_btn.setStyleSheet(nav_style)
        self.next_btn.setStyleSheet(nav_style)

        bar_layout.addWidget(self.back_btn)
        bar_layout.addStretch()
        bar_layout.addWidget(self.next_btn)
        main_layout.addLayout(bar_layout)
        
        # Tastaturkürzel
        QShortcut(QKeySequence("Left"), self, activated=self.go_back)
        QShortcut(QKeySequence("Right"), self, activated=self.go_next)
        QShortcut(QKeySequence("Escape"), self, activated=self.toggle_fullscreen)

    def toggle_fullscreen(self):
        """Wechselt zwischen Vollbild und Fenstermodus"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def set_language(self, language: str):
        """Setzt die Sprache der Anwendung"""
        self.language = language
        self.load_pages()
        self.update_buttons()
        logger.info(f"Sprache geändert zu: {language}")

    def create_flag_button(self, flag_file: str, language_code: str) -> QToolButton:
        """Erstellt einen Sprachumschalt-Button"""
        btn = QToolButton()
        btn.setIcon(QIcon(os.path.join(self.Explorer_Structure, flag_file)))
        btn.setIconSize(QSize(32, 32))
        btn.setFixedSize(40, 40)
        btn.setStyleSheet("""
            QToolButton {
                background: #323f4d;
                border: 2px solid #5d6d7e;
                border-radius: 6px;
                padding: 5px;
            }
            QToolButton:hover {
                background: #3d566e;
                border: 2px solid #3498db;
            }
            QToolButton:pressed {
                background: #21618c;
            }
        """)
        btn.clicked.connect(lambda _, lang=language_code: self.set_language(lang))
        return btn

    def create_start_page(self) -> QWidget:
        """Erstellt die Startseite"""
        page = QWidget()
        page.setStyleSheet("background-color: #333333;")
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(25)

        # Header mit Logo und Sprachbuttons
        header_widget = self.create_start_header()
        main_layout.addWidget(header_widget)

        # Hauptinhalt mit zwei Spalten
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(40)
        content_layout.setContentsMargins(0, 20, 0, 0)

        # Linke Spalte - Hauptinformationen
        left_column = self.create_start_left_column()
        content_layout.addWidget(left_column, stretch=3)

        # Rechte Spalte - Systemstatus
        right_column = self.create_start_right_column()
        content_layout.addWidget(right_column, stretch=2)

        main_layout.addWidget(content_widget, stretch=1)
        
        return page

    def create_start_header(self) -> QWidget:
        """Erstellt den Header der Startseite"""
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Logo links
        logo_label = QLabel()
        logo_path = os.path.join(self.Explorer_Structure, "logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            pixmap = pixmap.scaled(150, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("3D-SCANNER")
            logo_label.setStyleSheet("""
                color: #ecf0f1; 
                font-size: 24px; 
                font-weight: bold;
                font-family: Arial;
                padding: 10px;
                background: #323f4d;
                border-radius: 6px;
            """)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        header_layout.addWidget(logo_label)
        
        header_layout.addStretch()
        
        # Sprachbuttons rechts
        lang_widget = QWidget()
        lang_layout = QHBoxLayout(lang_widget)
        lang_layout.setSpacing(8)
        lang_layout.setContentsMargins(0, 0, 0, 0)
        
        btn_de = self.create_flag_button("de.png", "de")
        btn_it = self.create_flag_button("it.png", "it") 
        btn_en = self.create_flag_button("en.png", "en")

        for btn in [btn_de, btn_it, btn_en]:
            lang_layout.addWidget(btn)

        header_layout.addWidget(lang_widget)
        return header_widget

    def create_start_left_column(self) -> QWidget:
        """Erstellt die linke Spalte der Startseite"""
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setSpacing(25)
        
        # Titel
        title_label = QLabel(self.translator.get_text(self.language, "start", "title"))
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet("""
            font-size: 36px; 
            font-weight: bold; 
            color: #3498db;
            margin-bottom: 10px;
            font-family: Arial;
            padding-bottom: 15px;
            border-bottom: 2px solid #5d6d7e;
        """)
        left_layout.addWidget(title_label)

        # Untertitel
        subtitle_label = QLabel(self.translator.get_text(self.language, "start", "subtitle"))
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("""
            color: #bdc3c7; 
            font-size: 18px; 
            padding: 15px 0; 
            line-height: 1.4;
        """)
        left_layout.addWidget(subtitle_label)

        # Anweisungen
        instructions_widget = QWidget()
        instructions_layout = QVBoxLayout(instructions_widget)
        instructions_layout.setSpacing(12)
        
        texts = [
            self.translator.get_text(self.language, "start", "instruction1"),
            self.translator.get_text(self.language, "start", "instruction2"),
            self.translator.get_text(self.language, "start", "instruction3"),
            self.translator.get_text(self.language, "start", "instruction4")
        ]
        
        for i, text in enumerate(texts):
            instruction_frame = QFrame()
            instruction_frame.setStyleSheet("""
                QFrame {
                    background: #323f4d;
                    border: 1px solid #5d6d7e;
                    border-radius: 6px;
                    padding: 15px;
                }
            """)
            frame_layout = QHBoxLayout(instruction_frame)
            
            # Text
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            label.setWordWrap(True)
            label.setStyleSheet("color: #ecf0f1; font-size: 16px; line-height: 1.4;")
            frame_layout.addWidget(label, stretch=1)
            
            instructions_layout.addWidget(instruction_frame)

        left_layout.addWidget(instructions_widget)
        left_layout.addStretch()

        # Aktion-Buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setSpacing(20)
        
        scan_btn = QPushButton(self.translator.get_text(self.language, "start", "scan_btn"))
        save_btn = QPushButton(self.translator.get_text(self.language, "start", "save_btn"))
        
        # Scan-Button - Primäre Aktion
        scan_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: 600;
                padding: 16px 35px;
                border: none;
                border-radius: 6px;
                background: #3498db;
                color: #ecf0f1;
                min-width: 180px;
            }
            QPushButton:hover {
                background: #2980b9;
            }
            QPushButton:pressed {
                background: #21618c;
            }
        """)
        
        # Save-Button - Sekundäre Aktion
        save_btn.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: 600;
                padding: 16px 35px;
                border: 2px solid #5d6d7e;
                border-radius: 6px;
                background: #323f4d;
                color: #ecf0f1;
                min-width: 180px;
            }
            QPushButton:hover {
                background: #3d566e;
                color: #ecf0f1;
            }
            QPushButton:pressed {
                background: #21618c;
            }
        """)
        
        for btn in [scan_btn, save_btn]:
            btn.setFixedHeight(55)
            button_layout.addWidget(btn)

        scan_btn.clicked.connect(self.go_next)
        left_layout.addWidget(button_widget)

        return left_column

    def create_start_right_column(self) -> QFrame:
        """Erstellt die rechte Spalte (Systemstatus) der Startseite"""
        right_column = QFrame()
        right_column.setStyleSheet("""
            QFrame {
                background: #323f4d; 
                border: 1px solid #5d6d7e;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        right_layout = QVBoxLayout(right_column)
        right_layout.setSpacing(20)

        # Status-Überschrift
        status_title = QLabel(self.translator.get_text(self.language, "start", "status_title"))
        status_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_title.setStyleSheet("""
            font-size: 22px; 
            font-weight: 600; 
            color: #3498db;
            padding: 10px 0;
            border-bottom: 1px solid #5d6d7e;
        """)
        right_layout.addWidget(status_title)

        # Status-Buttons
        status_buttons = [
            (self.translator.get_text(self.language, "start", "check_camera"), self.check_camera),
            (self.translator.get_text(self.language, "start", "check_light"), self.check_light),
            (self.translator.get_text(self.language, "start", "check_measure"), self.check_measure),
            (self.translator.get_text(self.language, "start", "calibrate_scale"), self.calibrate_scale),
            (self.translator.get_text(self.language, "start", "check_storage"), self.check_storage)
        ]

        for name, callback in status_buttons:
            status_button = self.create_status_button(name, callback)
            right_layout.addWidget(status_button)

        right_layout.addStretch()

        # Refresh Button
        refresh_btn = QPushButton(self.translator.get_text(self.language, "start", "refresh_btn"))
        refresh_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                padding: 12px; 
                background: #333333;
                color: #ecf0f1; 
                border: 1px solid #5d6d7e; 
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #3498db;
                color: #ecf0f1;
            }
        """)
        refresh_btn.setFixedHeight(45)
        right_layout.addWidget(refresh_btn)

        return right_column

    def create_status_button(self, name: str, callback) -> QPushButton:
        """Erstellt einen Status-Button"""
        button = QPushButton(name)
        button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                font-weight: 500;
                padding: 12px 15px;
                border: 1px solid #5d6d7e;
                border-radius: 6px;
                background: #333333;
                color: #ecf0f1;
                text-align: left;
            }
            QPushButton:hover {
                background: #3498db;
                border-color: #3498db;
            }
            QPushButton:pressed {
                background: #21618c;
            }
        """)
        button.setFixedHeight(45)
        if callback:
            button.clicked.connect(callback)
        return button

    def convert_to_pixmap(self, frame: np.ndarray, width: int = 300, height: int = 300) -> QPixmap:
        """Konvertiert OpenCV-Bild zu QPixmap"""
        if frame is None or (isinstance(frame, np.ndarray) and np.all(frame == 0)):
            gray_pixmap = QPixmap(width, height)
            gray_pixmap.fill(Qt.GlobalColor.lightGray)
            return gray_pixmap
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        return pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def retry_image(self, idx: int):
        """Wiederholt die Aufnahme eines einzelnen Bildes"""
        logger.info(f"Wiederhole Bild {idx+1}")
        self.scan_start = True
        new_img = self.camera.take_picture(idx)
        if new_img is not None:
            self.images[idx] = new_img
            pixmap = self.convert_to_pixmap(new_img)
            self.image_labels[idx].setPixmap(pixmap)

    def discard_image(self, idx: int):
        """Verwirft ein Bild"""
        logger.info(f"Verworfen Bild {idx+1}")
        self.scan_start = True
        self.keep[idx] = False
        label = self.image_labels[idx]
        gray_pixmap = QPixmap(label.pixmap().size())
        gray_pixmap.fill(Qt.GlobalColor.lightGray)
        label.setPixmap(gray_pixmap)

    def make_card(self, text: str) -> QLabel:
        """Erstellt eine Textkarte"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                color: #dedede;
                font-size: 17px;
                font-weight: 400;
                padding: 18px 30px;
                margin: 10px 0;
                background: transparent;
                border-bottom: 1px solid #323f4d;
                line-height: 1.5;
            }
        """)
        label.setWordWrap(True)
        return label

    def make_card_with_input(self, label_text: str = "", preset_text: str = "", placeholder: str = "") -> QFrame:
        """Erstellt eine Eingabekarte"""
        frame = QFrame()
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        frame.setMinimumHeight(120)
        frame.setStyleSheet("""
            QFrame {
                background-color: #dedede;
                border-radius: 12px;
                border: 1px solid #bbb;
                padding: 12px;
            } QLabel {
                font-size: 18px;
                color: #333333;
            } QLineEdit {
                font-size: 20px;
                color: #333333;
                background: transparent;
                border: none;
                border-bottom: 2px solid #333333;
            }""")

        layout = QVBoxLayout(frame)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Überschrift
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        # Eingabefeld
        field = QLineEdit()
        field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        field.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if preset_text:
            field.setText(preset_text)
        if placeholder:
            field.setPlaceholderText(placeholder)

        layout.addWidget(field)
        return frame

    def _make_widget(self, item) -> QWidget:
        """Erstellt ein Widget basierend auf der Beschreibung"""
        if not isinstance(item, tuple):
            return self.make_card(str(item))
        
        widget_type = item[0]
        widget_creators = {
            "button": self._create_button_widget,
            "image": self._create_image_widget, 
            "ram_image": self._create_ram_image_widget,
            "ram_image_final": self._create_ram_image_final_widget,
            "title": self._create_title_widget,
            "input": self._create_input_widget
        }
        creator = widget_creators.get(widget_type)
        if creator:
            return creator(*item[1:])
        return self.make_card(str(item))

    def _create_button_widget(self, text: str, callback=None) -> QPushButton:
        """Erstellt einen Button"""
        btn = QPushButton(text)
        btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: 600;
                padding: 14px 30px;
                border: none;
                border-radius: 6px;
                background: #495057;
                color: #ffffff;
                min-width: 140px;
            } QPushButton:hover {
                background: #6c757d;
            } QPushButton:pressed {
                background: #343a40;
            }
        """)
        
        if callback and callable(callback):
            btn.clicked.connect(callback)
        else:
            btn.clicked.connect(lambda: print(f"Button '{text}' gedrückt"))
        
        return btn

    def _create_image_widget(self, base_name: str) -> QLabel:
        """Erstellt ein Bild-Widget"""
        label = QLabel()
        path = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            test_path = os.path.join(self.Explorer_Structure, base_name + ext)
            if os.path.exists(test_path):
                path = test_path
                break
        
        if path:
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                pixmap = pixmap.scaledToWidth(250, Qt.TransformationMode.SmoothTransformation)
                label.setPixmap(pixmap)
            else:
                label.setText(f"Bild konnte nicht geladen werden:\n{path}")
        else:
            label.setText(f"Kein Bild gefunden für '{base_name}'")

        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_ram_image_widget(self, idx: int) -> QLabel:
        """Erstellt ein RAM-Bild-Widget"""
        label = QLabel()
        self.image_labels[idx] = label
        if self.images[idx] is not None:
            pixmap = self.convert_to_pixmap(self.images[idx])
            label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_ram_image_final_widget(self, idx: int) -> QLabel:
        """Erstellt ein finales RAM-Bild-Widget"""
        label = QLabel()
        self.final_image_labels[idx] = label
        if self.final_images[idx] is not None:
            label.setPixmap(self.convert_to_pixmap(self.final_images[idx]))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_title_widget(self, text: str) -> QLabel:
        """Erstellt einen Titel"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 28px; font-weight: bold; color: #dedede;")
        return label

    def _create_input_widget(self, label_text: str, placeholder: str = "", preset_text: str = "") -> QFrame:
        """Erstellt ein Eingabewidget"""
        return self.make_card_with_input(label_text, preset_text, placeholder)

    def add_page(self, title: str, widgets: List[Any]):
        """Fügt eine Seite zum Stack hinzu"""
        page = QWidget()
        page.setStyleSheet("background-color: #333333;")
        page_layout = QVBoxLayout(page)
        page_layout.setSpacing(16)

        # Taskbar für die Seite
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(20)

        # Linke Seite: Seitentitel
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        title_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #3498db;")
        title_layout.addWidget(title_label, stretch=1)

        # Sprachbuttons
        btn_de = self.create_flag_button("de.png", "de")
        btn_it = self.create_flag_button("it.png", "it")
        btn_en = self.create_flag_button("en.png", "en")

        for btn in [btn_de, btn_it, btn_en]:
            title_layout.addWidget(btn)

        page_layout.addWidget(title_bar)

        # Scrollbereich
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)

        # Widgets hinzufügen
        for item in widgets:
            if isinstance(item, list):
                row_layout = QHBoxLayout()
                row_layout.setSpacing(12)
                for sub in item:
                    widget = self._make_widget(sub)
                    row_layout.addWidget(widget)
                layout.addLayout(row_layout)
            else:
                widget = self._make_widget(item)
                layout.addWidget(widget)

        layout.addStretch()
        scroll.setWidget(content)
        page_layout.addWidget(scroll)

        self.stack.addWidget(page)

    def load_pages(self):
        """Lädt alle Seiten neu, behält aber die aktuelle Seite bei"""
        # Speichere aktuelle Seite
        current_page = self.stack.currentIndex() if hasattr(self, 'stack') else 0
        
        if self.abmessung is None:
            self.abmessung = "Undefiniert"
        if self.gewicht is None:
            self.gewicht = "Undefiniert"
        if self.barcode is None:
            self.barcode = "Undefiniert"
            self.barcode_type = "Undefiniert"
        
        # Alte Seiten entfernen
        while self.stack.count() > 0:
            widget = self.stack.widget(0)
            self.stack.removeWidget(widget)
            widget.deleteLater()

        # Startseite mit speziellem Layout
        start_page = self.create_start_page()
        self.stack.addWidget(start_page)

        # Gemeinsame Seitenstruktur für alle Sprachen
        page_configs = {
            "photo": {
                "title_key": "photo",
                "content": [
                    [("ram_image", 0), ("ram_image", 1)],
                    [("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=0: self.retry_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=1: self.retry_image(idx))],
                    [("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=0: self.discard_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=1: self.discard_image(idx))],
                    [("ram_image", 2), ("ram_image", 3)],
                    [("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=2: self.retry_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "retry_btn"), lambda _, idx=3: self.retry_image(idx))],
                    [("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=2: self.discard_image(idx)),
                    ("button", self.translator.get_text(self.language, "photo", "discard_btn"), lambda _, idx=3: self.discard_image(idx))]
                ]
            },
            "overview": {
                "title_key": "overview", 
                "content": [
                    [("ram_image_final", 0), ("ram_image_final", 1)],
                    [("ram_image_final", 2), ("ram_image_final", 3)],
                    f"{self.translator.get_text(self.language, 'overview', 'dimensions')} {self.abmessung}{self.translator.get_text(self.language, 'overview', 'mm')}",
                    f"{self.translator.get_text(self.language, 'overview', 'weight')} {self.gewicht}{self.translator.get_text(self.language, 'overview', 'kg')}"
                ]
            },
            "storage": {
                "title_key": "storage",
                "content": [
                    ("image", "barcode"),
                    ("input", self.translator.get_text(self.language, "storage", "barcode_label"), f"{self.barcode}"),
                    ("input", self.translator.get_text(self.language, "storage", "barcode_type"), f"{self.barcode_type}"),
                    [("button", self.translator.get_text(self.language, "storage", "sap_btn")),
                    ("button", self.translator.get_text(self.language, "storage", "save_btn"))]
                ]
            }
        }
        
        # Seiten hinzufügen
        for page_key in ["photo", "overview", "storage"]:
            config = page_configs[page_key]
            self.add_page(
                self.translator.get_text(self.language, config["title_key"], "title"),
                config["content"]
            )
        
        # Zurück zur ursprünglichen Seite springen (maximale Seitenzahl beachten)
        max_pages = self.stack.count()
        if current_page >= max_pages:
            current_page = max_pages - 1
        
        self.stack.setCurrentIndex(current_page)
        self.update_buttons()
        

    def go_back(self):
        """Geht zur vorherigen Seite"""
        idx = self.stack.currentIndex()
        logger.info(f"go_back: Aktuelle Seite {idx}, scan_start={self.scan_start}")
        
        # Spezialfall: Von Foto-Auswahl (Index 1) zurück zur Startseite (Index 0)
        if idx == 1:
            self.scan_start = False
            reply = QMessageBox.question(
                self,
                "Datenverlust bestätigen",
                "Möchten Sie wirklich zurück zur Startseite? Alle erfassten Daten gehen verloren.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

            if reply == QMessageBox.StandardButton.Cancel:
                return
        
        # Spezialfall: Von Kamera-Übersicht (Index 2) zurück zur Foto-Auswahl (Index 1)
        if idx == 2:
            # Setze scan_start zurück, damit wir neue Bilder aufnehmen können
            self.scan_start = True
            logger.info("Zurück zur Foto-Auswahl: scan_start=True gesetzt")
        
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
            self.update_buttons()
            self.centralWidget().updateGeometry()

    def go_next(self):
        """Geht zur nächsten Seite"""
        idx = self.stack.currentIndex()
        logger.info(f"go_next: Aktuelle Seite {idx}, scan_start={self.scan_start}")
        
        if idx >= self.stack.count() - 1:
            logger.info("Bereits auf letzter Seite")
            return
        
        # Von Startseite (Index 0) zu Foto-Auswahl (Index 1)
        elif idx == 0:
            if not self.scan_start:
                self.scan_start = True
                if not hasattr(self, "images"):
                    self.images = [None] * CONFIG.NUM_CAMERAS

                logger.info("Starte Bildaufnahme von allen Kameras")
                all_images = self.camera.take_all_pictures()
                for i, img in enumerate(all_images):
                    self.images[i] = img
                    if self.image_labels[i] is not None:
                        self.image_labels[i].setPixmap(self.convert_to_pixmap(img))

                self.stack.setCurrentIndex(idx + 1)
                self.update_buttons()
            else:
                # Falls scan_start schon True ist (z.B. nach Zurück-Navigation)
                self.stack.setCurrentIndex(idx + 1)
                self.update_buttons()
            return

        # Von Foto-Auswahl (Index 1) zu Kamera-Übersicht (Index 2)
        elif idx == 1:
            if self.scan_start:
                self.show_loading_dialog()
            else:
                # Falls keine Bilder aufgenommen wurden
                QMessageBox.warning(
                    self,
                    "Keine Bilder",
                    "Bitte nehmen Sie zuerst Bilder auf, bevor Sie fortfahren."
                )
        
        # Von Kamera-Übersicht (Index 2) zu Storage (Index 3)
        elif idx == 2:
            self.stack.setCurrentIndex(idx + 1)
            self.update_buttons()
            
    def show_loading_dialog(self):
        """Zeigt den Lade-Dialog mit Fortschrittsbalken"""
        self.loading_dialog = QDialog(self)
        self.loading_dialog.setWindowTitle("Ladevorgang der Daten")
        self.loading_dialog.setModal(True)
        self.loading_dialog.setFixedSize(350, 450)

        layout = QVBoxLayout(self.loading_dialog)
        
        # Lade-GIF
        movie = QMovie(os.path.join(self.Explorer_Structure, "loading.gif"))
        gif_label = QLabel()
        gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gif_label.setMovie(movie)
        movie.start()
        layout.addWidget(gif_label)

        # Status-Label
        status_label = QLabel("Daten werden verarbeitet...")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_label.setStyleSheet("font-size: 16px; margin: 20px;")
        layout.addWidget(status_label)

        # Fortschrittsbalken
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #5d6d7e;
                border-radius: 5px;
                text-align: center;
                padding: 1px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Abbrechen-Button
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.setFixedSize(120, 40)
        cancel_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px;
            }
        """)
        layout.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Worker starten
        self.start_worker()

        def finish_loading():
            if self.loading_dialog.isVisible():
                self.loading_dialog.accept()
                self.stack.setCurrentIndex(2)  # Übersichtsseite
                self.update_buttons()
                self.scan_start = False
                QMessageBox.information(
                    self,
                    "Scan abgeschlossen",
                    "Der Scan war erfolgreich!\nDie Daten stehen nun zur Verfügung."
                )
        
        def cancel_loading():
            try:
                if hasattr(self, 'worker') and self.worker.isRunning():
                    self.worker.terminate()
                    self.worker.wait(1000)  # 1 Sekunde warten
            except Exception as e:
                logger.error(f"Fehler beim Abbrechen: {e}")
            
            self.loading_dialog.reject()
            self.stack.setCurrentIndex(1)
            self.update_buttons()
            QMessageBox.warning(
                self,
                "Scan abgebrochen",
                "Der Scan wurde abgebrochen."
            )
        
        # Verbindungen herstellen
        self.worker.finished.connect(finish_loading)
        cancel_btn.clicked.connect(cancel_loading)
        
        self.loading_dialog.exec()

    def update_buttons(self):
        """Aktualisiert die Sichtbarkeit der Navigationsbuttons"""
        current_index = self.stack.currentIndex()
        total_pages = self.stack.count()
        
        logger.debug(f"update_buttons: Seite {current_index}/{total_pages-1}, scan_start={self.scan_start}")

        if current_index == 0:
            self.back_btn.hide()
            self.next_btn.hide()
            if not self.camera.available_cameras:
                self.next_btn.setEnabled(False)
                self.next_btn.setToolTip("Keine Kamera verfügbar")
            else:
                self.next_btn.setEnabled(True)
                self.next_btn.setToolTip("")
        
        elif current_index == total_pages - 1:
            self.back_btn.show()
            self.next_btn.hide()
        else:
            self.back_btn.show()
            self.next_btn.show()
            
            # Auf Foto-Auswahl (Index 1): Weiter nur wenn Bilder vorhanden
            if current_index == 1:
                has_images = any(img is not None for img in self.images)
                self.next_btn.setEnabled(has_images and self.scan_start)
                if not has_images:
                    self.next_btn.setToolTip("Keine Bilder aufgenommen")
                elif not self.scan_start:
                    self.next_btn.setToolTip("Bitte Bilder aufnehmen")
                else:
                    self.next_btn.setToolTip("")
            else:
                self.next_btn.setEnabled(True)
                self.next_btn.setToolTip("")
    
    def start_worker(self):
        """Startet den Worker-Thread für parallele Verarbeitung"""
        self.worker = ParallelWorker(self.images)
        self.worker.output_received.connect(self.handle_output)
        self.worker.progress_updated.connect(self.update_progress_bar)
        self.worker.finished.connect(lambda: logger.info("Alle Tasks fertig"))
        self.worker.start()  # Wichtig: start() nicht start_processing()

    def update_progress_bar(self, value: int):
        """Aktualisiert den Fortschrittsbalken"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setValue(value)

    def handle_output(self, script_name: str, data: Any):
        """Verarbeitet die Ergebnisse der Worker-Threads"""
        logger.info(f"Ergebnis von {script_name} erhalten: Typ={type(data)}")
        
        if script_name == "Abmessung":
            if isinstance(data, list):
                self.abmessungen = data
                logger.info(f"Abmessungen erhalten: {data}")
                
                # Versuche, eine aussagekräftige Abmessung zu extrahieren
                try:
                    # Finde die erste nicht-leere Abmessung
                    valid_dimensions = [d for d in data if d != "0 x 0"]
                    if valid_dimensions:
                        # Nimm die erste gültige Abmessung
                        first_dim = valid_dimensions[0]
                        parts = first_dim.split(" x ")
                        if len(parts) == 2:
                            try:
                                width = int(float(parts[0]))
                                height = int(float(parts[1]))
                                # Für einen Würfel: Länge = Breite = Höhe (vereinfacht)
                                length = width
                                self.abmessung_gesamt = f"{length} x {width} x {height}"
                                self.abmessung = self.abmessung_gesamt
                                logger.info(f"Gesamt-Abmessung berechnet: {self.abmessung}")
                            except:
                                self.abmessung = first_dim
                                logger.info(f"Abmessung übernommen: {first_dim}")
                        else:
                            self.abmessung = first_dim
                    else:
                        self.abmessung = "Undefiniert"
                        logger.warning("Keine gültigen Abmessungen gefunden")
                except Exception as e:
                    logger.error(f"Fehler beim Berechnen der Abmessung: {e}")
                    self.abmessung = "Undefiniert"
            else:
                logger.error(f"Unerwartetes Format für Abmessungen: {type(data)}")
                self.abmessung = "Undefiniert"

        elif script_name == "yolo_frames":
            if isinstance(data, list):
                self.annotierte_frames = data
                logger.info(f"Erhalte {len(data)} annotierte Frames")
                
                # WICHTIG: Aktualisiere final_images SOFORT
                for i in range(min(len(data), CONFIG.NUM_CAMERAS)):
                    if data[i] is not None and self.keep[i]:
                        # Konvertiere RGB zu BGR für korrekte Anzeige
                        if len(data[i].shape) == 3 and data[i].shape[2] == 3:
                            # Prüfe ob es bereits BGR ist (OpenCV Standard)
                            # Normalerweise ist YOLO-Ausgabe RGB, konvertiere zu BGR
                            try:
                                self.final_images[i] = cv2.cvtColor(data[i], cv2.COLOR_RGB2BGR)
                                logger.debug(f"Bild {i}: Konvertiert von RGB zu BGR")
                            except:
                                self.final_images[i] = data[i]
                        else:
                            self.final_images[i] = data[i]
                        
                        logger.debug(f"Final image {i} gesetzt (Größe: {self.final_images[i].shape})")
                    else:
                        self.final_images[i] = None
                        logger.debug(f"Final image {i} ist None (keep={self.keep[i]})")
            else:
                logger.error(f"Unerwartetes Format für yolo_frames: {type(data)}")

        elif script_name == "barcode":
            idx = data.get("index", -1)
            found = data.get("found", False)
            value = data.get("value", None)
            b_type = data.get("type", None)
            
            if found:
                self.barcode = value
                self.barcode_type = b_type
                logger.info(f"Barcode erkannt: Wert='{value}', Typ='{b_type}'")
            else:
                logger.debug(f"Kein Barcode in Bild {idx}")

        elif script_name == "weight":
            self.gewicht = data
            logger.info(f"Gewicht: {data}")
        
        # Nach jedem Update prüfen, ob wir die GUI aktualisieren können
        self._check_and_update_gui()

    def _check_and_update_gui(self):
        """Prüft ob alle Daten vorhanden sind und aktualisiert die GUI"""
        # Prüfe ob alle notwendigen Daten vorhanden sind
        has_abmessung = hasattr(self, "abmessung") and self.abmessung not in [None, "Undefiniert"]
        has_barcode = hasattr(self, "barcode") and self.barcode not in [None, "Undefiniert"]
        has_gewicht = hasattr(self, "gewicht") and self.gewicht not in [None, "Undefiniert"]
        
        # Wir aktualisieren die GUI wenn:
        # 1. Alle Daten vorhanden sind, ODER
        # 2. Die YOLO-Frames vorhanden sind (für Bildanzeige)
        
        update_needed = False
        
        if has_abmessung and has_barcode and has_gewicht:
            logger.info("Alle Daten vorhanden - aktualisiere GUI")
            update_needed = True
        elif hasattr(self, "annotierte_frames") and self.annotierte_frames:
            logger.info("YOLO-Frames vorhanden - aktualisiere Bilder in GUI")
            update_needed = True
        
        if update_needed:
            # Stelle sicher, dass alle final_images gesetzt sind
            for i in range(CONFIG.NUM_CAMERAS):
                if self.final_images[i] is None and i < len(self.images):
                    self.final_images[i] = self.images[i]
            
            # Lade die Seiten neu
            self.load_pages()
            QApplication.processEvents()  # Erzwinge GUI-Update

    # Platzhalter-Funktionen für die Systemprüfung
    def check_camera(self):
        QMessageBox.information(self, "Kamera-Prüfung", "Kamera-System wird geprüft...")
        logger.info("Kamera-Prüfung gestartet")

    def check_light(self):
        QMessageBox.information(self, "Beleuchtungs-Prüfung", "Beleuchtung wird geprüft...")
        logger.info("Beleuchtungs-Prüfung gestartet")

    def check_measure(self):
        QMessageBox.information(self, "Mess-System-Prüfung", "Mess-System wird geprüft...")
        logger.info("Mess-System-Prüfung gestartet")

    def calibrate_scale(self):
        QMessageBox.information(self, "Waagen-Kalibrierung", "Waage wird kalibriert...")
        logger.info("Waagen-Kalibrierung gestartet")

    def check_storage(self):
        QMessageBox.information(self, "Speicher-Prüfung", "Speicher wird geprüft...")
        logger.info("Speicher-Prüfung gestartet")

    def keyPressEvent(self, event):
        """Behandelt Tastatureingaben"""
        if event.key() == Qt.Key.Key_Left:
            self.go_back()
        elif event.key() == Qt.Key.Key_Right:
            self.go_next()
        elif event.key() == Qt.Key.Key_Escape:
            self.toggle_fullscreen()
        else:
            super().keyPressEvent(event)

if __name__ == "__main__":
    logger.info("3D-Scanner wird gestartet...")
    app = QApplication(sys.argv)
    w = FullscreenApp()
    w.show()
    sys.exit(app.exec())





