"""Static UI copy (no Ursina)."""

TIME_WINDOW_REJECT_TOAST = (
    "Der richtige Zeitpunkt zum Beobachten des Objekts ist noch nicht gekommen. Drücke „Beobachtung beginnen”, sobald die Uhr im Beobachtungsfenster ist."
)

STEUERUNG_TEXT = """Steuerung — [F1] dieses Fenster, [H] Hilfe zum Spielablauf.

--- Kamera ---
Rechte Maustaste halten + Maus bewegen: um die Szene drehen.
Mausrad: zoomen. Zwei Finger auseinander/zusammen (Pinch): zoomen (Touchpads und Touchscreens).
Mittlere Maustaste halten + Maus: schwenken (Pan).
Während rechte Maustaste gedrückt: WASD bewegen, Q / E senken bzw. heben (Shift = schneller).
Alt+F: Kamera-Start. Shift+F: auf Punkt unter dem Cursor fokussieren. Shift+P: Perspektive / Orthogonal.

--- Spiel (allgemein) ---
Leertaste: je nach Phase (Warten starten, Belichtung, … siehe [H]).
Z: Observatorium öffnen (wenn angezeigt). X: Kuppel-Animation zurück (Debug).
H: Hilfe zum aktuellen Schritt. I: Kuppel grob auf Teleskop ausrichten (Hilfe).
R: in der Bildphase Serie neu starten. Linksklick: Objekt wählen (wenn aktiv).

--- Teleskop (Schritt „auf Ziel ausrichten“) ---
Pfeiltaste links / rechts: RA-Achse. Pfeiltaste hoch / runter: DEC-Achse.

--- Kuppel (Schritt „Kuppel ausrichten“) ---
Pfeiltaste links / rechts: Kuppel in Azimut drehen, bis der Schlitz freie Sicht erlaubt.
Zusätzlich (Desktop): [ und ] drehen die Kuppel in Schritt 2 und 3.

"""

B1_STEUERUNG_TEXT = """Steuerung (B1 native)

[H] Hilfe zum Schritt — erneut [H] schließt. [F1] diese Übersicht — aus der Hilfe wechselt [F1] hierher, erneut [F1] schließt.
In der Hilfe: [Backspace] oder [Eingabe] schließen ebenfalls.

Kamera: Maus ziehen dreht, Mausrad zoomt; mit zwei Fingern Pinch-Geste zoomen (Touch).
Pfeiltasten halten (←→↑↓) drehen die Ansicht.

Spiel: [Leertaste] [Z] [I] (Kuppel fährt kurz animiert) — Details in [H]. Teleskop Schritt 2: [W][A][S][D] — A/D RA, W/S Dec.; Kuppel dazu [ und ]. Schritt 3: Kuppel auch mit A/D oder [ und ].
Ab Schritt 2: [X] schließt Schlitz und Klappe animiert (~10 s) zur Ruhestellung (wie Desktop-Ursina).
[R] in Schritt 5: Aufnahmeserie neu. Linksklick: Objekt wählen.

Schritt 6 (Katalog): [Tab] zwischen Objektname und Entdeckerteam; Text tippen; [Backspace] löscht ein Zeichen; [Eingabe] oder „Speichern“ speichert in user_saves.dat.
Schritt 7: Tabelle aller Katalogeinträge; [Eingabe] beendet die App (oder „Zurück zur App-Übersicht“ im Touch-Layout).

[Hinweis] Himmel / Zenit: Maus nach unten ziehen oder Pfeil ↓ (weiter neigbar). Zoom mit Mausrad —
weit herauszoomen (bis B1_CAM_MAX_DIST) und B1_CAM_PIVOT_Y bei Bedarf erhöhen für mehr Überblick.

"""
