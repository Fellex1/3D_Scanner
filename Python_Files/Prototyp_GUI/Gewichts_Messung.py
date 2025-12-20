# Gewichts_Messung.py
import time
import board
import busio
import adafruit_nau7802
import json
import os

class NAU7802_Waage:
    def __init__(self):
        self.i2c = None
        self.nau7802 = None
        self.offset = 0
        self.scale = 1.0
        self.calibration_file = "kalibrierung.json"
        self.initialize_sensor()
        self.load_calibration()
    
    def initialize_sensor(self):
        """Initialisiert den NAU7802 Sensor"""
        try:
            # I2C Bus initialisieren
            self.i2c = busio.I2C(board.SCL, board.SDA)
            
            # NAU7802 Sensor initialisieren
            self.nau7802 = adafruit_nau7802.NAU7802(self.i2c, address=0x2A)
            
            # Sensor konfigurieren
            self.nau7802.gain = 128  # Verstärkung: 128, 64, 32, 16, 8, 4, 2, 1
            self.nau7802.low_power = False
            self.nau7802.ldo_voltage = adafruit_nau7802.LDOVoltage.V3V3
            
            # Sample Rate einstellen
            self.nau7802.rate = adafruit_nau7802.Rate.RATE_10SPS
            
            # Kalibrierung durchführen
            self.nau7802.calibrate(adafruit_nau7802.CalibrationMode.INTERNAL)
            time.sleep(1)
            
            # Warten auf erste Messung
            while not self.nau7802.ready:
                time.sleep(0.1)
                
            print("NAU7802 initialisiert")
            
        except Exception as e:
            print(f"Fehler bei Initialisierung: {e}")
            self.nau7802 = None
    
    def load_calibration(self):
        """Lädt Kalibrierungsdaten aus Datei"""
        try:
            if os.path.exists(self.calibration_file):
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)
                    self.offset = data.get('offset', 0)
                    self.scale = data.get('scale', 1.0)
                print("Kalibrierung geladen")
            else:
                print("Keine Kalibrierung gefunden. Bitte kalibrieren.")
        except Exception as e:
            print(f"Fehler beim Laden der Kalibrierung: {e}")
    
    def save_calibration(self):
        """Speichert Kalibrierungsdaten in Datei"""
        try:
            data = {
                'offset': self.offset,
                'scale': self.scale,
                'timestamp': time.time()
            }
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f, indent=4)
            print("Kalibrierung gespeichert")
        except Exception as e:
            print(f"Fehler beim Speichern der Kalibrierung: {e}")
    
    def read_raw(self, samples=10):
        """Liest Rohwerte vom ADC"""
        if not self.nau7802:
            return 0
        
        values = []
        for i in range(samples):
            while not self.nau7802.ready:
                time.sleep(0.001)
            values.append(self.nau7802.read())
            time.sleep(0.01)
        
        # Median zur Eliminierung von Ausreißern
        values.sort()
        return values[len(values)//2]
    
    def get_weight(self, samples=10, raw=False):
        """Gibt das Gewicht in kg zurück"""
        if not self.nau7802:
            return 0
        
        raw_value = self.read_raw(samples)
        
        if raw:
            return raw_value
        
        # Kalibrierung anwenden
        calibrated_value = (raw_value - self.offset) / self.scale
        return calibrated_value
    
    def calibrate_scale(self, known_weight_kg=1.0, tare_samples=10, weight_samples=50):
        """Kalibriert die Waage mit bekanntem Gewicht"""
        if not self.nau7802:
            print("Sensor nicht initialisiert!")
            return
        
        print("=== KALIBRIERUNG ===")
        print("1. Entfernen Sie alles von der Waage")
        input("Drücken Sie Enter, wenn bereit...")
        
        # Tara (Nullpunkt) messen
        print("Messe Tara...")
        tare_values = []
        for i in range(tare_samples):
            tare_values.append(self.read_raw(1))
            time.sleep(0.1)
        
        tare_offset = sum(tare_values) / len(tare_values)
        self.offset = tare_offset
        print(f"Tara Offset: {tare_offset}")
        
        # Bekanntes Gewicht auflegen
        print(f"\n2. Legen Sie {known_weight_kg}kg auf die Waage")
        input("Drücken Sie Enter, wenn bereit...")
        
        print("Messe bekanntes Gewicht...")
        weight_values = []
        for i in range(weight_samples):
            weight_values.append(self.read_raw(1))
            time.sleep(0.1)
        
        weight_avg = sum(weight_values) / len(weight_values)
        
        # Skalierungsfaktor berechnen
        self.scale = (weight_avg - self.offset) / known_weight_kg
        
        print(f"Gewicht Rohwert: {weight_avg}")
        print(f"Skalierungsfaktor: {self.scale}")
        
        # Kalibrierung speichern
        self.save_calibration()
        
        print("\nKalibrierung abgeschlossen!")
        print(f"Offset: {self.offset}")
        print(f"Scale: {self.scale}")
        
        # Testmessung
        test_weight = self.get_weight(20)
        print(f"Testmessung: {test_weight:.3f} kg")
    
    def tare(self, samples=20):
        """Setzt den aktuellen Wert als Nullpunkt"""
        if not self.nau7802:
            return
        
        print("Tara wird durchgeführt...")
        tare_values = []
        for i in range(samples):
            tare_values.append(self.read_raw(1))
            time.sleep(0.1)
        
        self.offset = sum(tare_values) / len(tare_values)
        self.save_calibration()
        print(f"Neuer Offset: {self.offset}")

# Globale Instanz der Waage
waage = NAU7802_Waage()

def get_weight(samples=10):
    """Gibt das aktuelle Gewicht in kg zurück"""
    return waage.get_weight(samples)

def calibrate_scale(known_weight_kg=1.0):
    """Führt eine Kalibrierung mit bekanntem Gewicht durch"""
    waage.calibrate_scale(known_weight_kg)

def tare():
    """Setzt die Waage auf Null"""
    waage.tare()

def test_messung():
    """Testet kontinuierliche Messungen"""
    print("Kontinuierliche Messung - Strg+C zum Beenden")
    try:
        while True:
            gewicht = get_weight(5)
            print(f"Gewicht: {gewicht:.3f} kg")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMessung beendet")

if __name__ == "__main__":
    # Menü zur Auswahl der Funktion
    print("=== NAU7802 Wägezellen Interface ===")
    print("1: Gewicht messen")
    print("2: Kalibrieren")
    print("3: Tara (Nullpunkt setzen)")
    print("4: Kontinuierliche Messung")
    print("5: Rohwert anzeigen")
    
    try:
        auswahl = input("Auswahl (1-5): ").strip()
        
        if auswahl == "1":
            gewicht = get_weight(20)
            print(f"\nGemessenes Gewicht: {gewicht:.3f} kg")
        
        elif auswahl == "2":
            try:
                gewicht = float(input("Bekanntes Gewicht in kg (z.B. 1.0): "))
                calibrate_scale(gewicht)
            except ValueError:
                print("Ungültige Eingabe!")
        
        elif auswahl == "3":
            tare()
            print("Tara durchgeführt")
        
        elif auswahl == "4":
            test_messung()
        
        elif auswahl == "5":
            raw = waage.get_weight(10, raw=True)
            print(f"Rohwert: {raw}")
            calib = waage.get_weight(10)
            print(f"Kalibriert: {calib:.3f} kg")
        
        else:
            print("Ungültige Auswahl")
    
    except KeyboardInterrupt:
        print("\nProgramm beendet")