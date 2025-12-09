# BarCode_v02.py - Vollständig überarbeitete Version
import cv2
import contextlib
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from pyzbar.pyzbar import decode
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import logging

# Logging setup
logger = logging.getLogger(__name__)

class BarcodeDetector:
    """Erweiterte Barcode-Detection-Klasse mit YOLO-Unterstützung"""
    
    def __init__(self, model_path: str = "YOLOV8s_Barcode_Detection.pt"):
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO-Modell geladen von {model_path}")
        except Exception as e:
            logger.error(f"Fehler beim Laden des YOLO-Modells: {e}")
            self.model = None
        
        self.detected_barcodes: List[Dict[str, Any]] = []
        
    def detect_barcodes_in_image(self, image: np.ndarray, image_index: int = 0, image_name: str = "") -> List[Dict[str, Any]]:
        """Erkennt alle Barcodes in einem Bild"""
        if image is None:
            logger.warning(f"Bild {image_index} ist None")
            return []
        
        if self.model is None:
            logger.error("YOLO-Modell nicht verfügbar")
            return []
        
        try:
            # Konvertiere Bild zu RGB für YOLO (falls BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Prüfe ob es BGR ist (OpenCV Standard)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # YOLO Vorhersage
            results = self.model.predict(image_rgb, verbose=False)
            image_barcodes = []
            
            for r_idx, r in enumerate(results):
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box_idx, box in enumerate(r.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Extrahiere ROI
                        roi = image_rgb[y1:y2, x1:x2]
                        
                        # Überspringe zu kleine ROIs
                        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                            logger.debug(f"ROI zu klein: {roi.shape}")
                            continue
                        
                        # Barcode dekodieren
                        barcode_data = self._decode_roi(roi, f"image_{image_index}_box_{box_idx}")
                        
                        if barcode_data["found"]:
                            barcode_info = {
                                "image_index": image_index,
                                "image_name": image_name if image_name else f"Bild_{image_index}",
                                "value": barcode_data["value"],
                                "type": barcode_data["type"],
                                "confidence": float(box.conf) if hasattr(box, 'conf') else 1.0,
                                "bbox": (x1, y1, x2, y2),
                                "cropped_image": roi.copy(),
                                "full_image": image_rgb.copy(),  # Für Kontext
                                "bbox_on_full": (x1, y1, x2, y2)  # Bounding Box auf Originalbild
                            }
                            image_barcodes.append(barcode_info)
                            logger.info(f"Barcode erkannt in Bild {image_index}: {barcode_data['value']} (Typ: {barcode_data['type']})")
            
            logger.info(f"Bild {image_index}: {len(image_barcodes)} Barcode(s) erkannt")
            return image_barcodes
            
        except Exception as e:
            logger.error(f"Fehler bei Barcode-Erkennung Bild {image_index}: {e}")
            return []
    
    def _decode_roi(self, roi: np.ndarray, img_name: str) -> Dict[str, Any]:
        """Dekodiert einen Barcode in einer ROI mit verschiedenen Vorverarbeitungen"""
        if roi is None or roi.size == 0:
            return {"found": False, "value": None, "type": None}
        
        preprocess_methods = {
            "orig": lambda x: x,
            "unsharp": self._unsharp,
            "clahe": self._clahe_equalize,
            "invert": self._invert,
            "adaptive": self._adaptive_thresh,
            "clahe+adaptive+inv": lambda x: self._invert(self._adaptive_thresh(self._clahe_equalize(x)))
        }
        
        scales = [1.0, 1.5, 2.0, 3.0]
        rotations = list(range(-40, 41, 5))
        
        # Konvertiere zu Grau
        if len(roi.shape) == 3:
            if roi.shape[2] == 3:
                # RGB zu Grau
                gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            elif roi.shape[2] == 4:
                # RGBA zu Grau
                gray = cv2.cvtColor(roi, cv2.COLOR_RGBA2GRAY)
            else:
                gray = roi[:,:,0]  # Nimm ersten Kanal
        else:
            gray = roi
        
        # Hauptvorverarbeitung: CLAHE für besseren Kontrast
        gray = self._clahe_equalize(gray)
        
        for pname, func in preprocess_methods.items():
            img_proc = func(gray)
            
            for s in scales:
                scaled = self._scale(img_proc, s)
                
                for angle in rotations:
                    rotated = self._rotate(scaled, angle)
                    dec = self._try_decode(rotated)
                    
                    if dec:
                        d = dec[0]
                        try:
                            val = d.data.decode("utf-8", errors="ignore").strip()
                            typ = d.type
                            
                            if val:  # Nur gültige Werte zurückgeben
                                logger.debug(f"Barcode erfolgreich dekodiert mit {pname}, Skala {s}, Winkel {angle}")
                                return {"found": True, "value": val, "type": typ}
                        except Exception as decode_error:
                            logger.warning(f"Fehler beim Dekodieren: {decode_error}")
                            continue
        
        return {"found": False, "value": None, "type": None}
    
    def detect_barcodes_in_images(self, images: List[np.ndarray], image_names: List[str] = None) -> List[Dict[str, Any]]:
        """Erkennt Barcodes in mehreren Bildern"""
        self.detected_barcodes.clear()
        
        for idx, img in enumerate(images):
            if img is None:
                continue
            
            img_name = image_names[idx] if image_names and idx < len(image_names) else f"Bild_{idx}"
            barcodes = self.detect_barcodes_in_image(img, idx, img_name)
            self.detected_barcodes.extend(barcodes)
        
        logger.info(f"Insgesamt {len(self.detected_barcodes)} Barcodes in {len(images)} Bildern erkannt")
        return self.detected_barcodes
    
    def get_all_barcodes(self) -> List[Dict[str, Any]]:
        """Gibt alle erkannten Barcodes zurück"""
        return self.detected_barcodes
    
    def clear_barcodes(self):
        """Löscht alle gespeicherten Barcodes"""
        self.detected_barcodes.clear()
    
    # Hilfsmethoden für Bildverarbeitung
    @staticmethod
    def _unsharp(gray, amount=1.5):
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
    
    @staticmethod
    def _clahe_equalize(gray):
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(gray)
        except:
            return gray
    
    @staticmethod
    def _invert(gray):
        return 255 - gray
    
    @staticmethod
    def _adaptive_thresh(gray):
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    
    @staticmethod
    def _rotate(img, angle):
        if img is None or img.size == 0:
            return img
            
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    
    @staticmethod
    def _scale(img, factor):
        if img is None or img.size == 0:
            return img
            
        h, w = img.shape[:2]
        new_w, new_h = int(w * factor), int(h * factor)
        
        if new_w <= 0 or new_h <= 0:
            return img
            
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    @staticmethod
    def _try_decode(img_np):
        if img_np is None or img_np.size == 0:
            return []
            
        try:
            # Konvertiere zu 8-bit wenn nötig
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)
                
            pil_img = Image.fromarray(img_np)
            with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                return decode(pil_img)
        except Exception as e:
            logger.debug(f"Fehler bei decode-Versuch: {e}")
            return []


# Kompatibilitätsfunktion für bestehenden Code
def process_roi(roi: np.ndarray, img_name: str) -> Dict[str, Any]:
    """Kompatibilitätsfunktion für bestehenden Code"""
    detector = BarcodeDetector()
    return detector._decode_roi(roi, img_name)


# Hauptfunktion für eigenständigen Betrieb
def main():
    """Hauptfunktion für eigenständigen Betrieb"""
    USE_CAMERA = True
    INPUT_DIR = Path("GUI_Anzeige")
    
    detector = BarcodeDetector()
    
    # Kamera-Modus
    if USE_CAMERA:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Kamera konnte nicht geöffnet werden.")
            return
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("[ERROR] Kein Frame von Kamera erhalten.")
            return
        
        # Barcodes erkennen
        barcodes = detector.detect_barcodes_in_image(frame, 0, "Kamera_Bild")
        
        # Bild mit Bounding Boxes anzeigen
        display_img = frame.copy()
        for barcode in barcodes:
            x1, y1, x2, y2 = barcode["bbox"]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Text hinzufügen
            label = f"{barcode['value']} ({barcode['type']})"
            cv2.putText(display_img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Kamera Barcode Detection", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    # Ordner-Modus
    else:
        bilder = [
            f for f in INPUT_DIR.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            and "barcode" in f.stem.lower()
        ]
        
        if not bilder:
            print(f"Keine passenden Bilder gefunden in {INPUT_DIR}")
            return
        
        gesamt, erkannt = 0, 0
        for img_path in sorted(bilder):
            gesamt += 1
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"{img_path.name}: Bild konnte nicht geladen werden!")
                continue
            
            barcodes = detector.detect_barcodes_in_image(img, 0, img_path.name)
            if barcodes:
                erkannt += 1
        
        quote = (erkannt / gesamt * 100) if gesamt > 0 else 0
        print(f"\nTrefferquote: {erkannt}/{gesamt} erkannt ({quote:.2f} %)")

if __name__ == "__main__":
    main()