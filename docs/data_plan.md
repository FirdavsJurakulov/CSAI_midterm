# Data plan

## Unit of labelling: cropped cars
Photos → `scripts/extract_crops.py` (YOLO) → one crop per car → label each crop.
`data/labels.csv`, one line per car:

```
crop_path,brand,colour,view,source,session
crops/2026-10-24_magic-city/IMG_0412_c1.jpg,BYD,white,front34,own,2026-10-24_magic-city
```

- `brand`: badge name in English (Daewoo ≠ Chevrolet, Jetour ≠ Chery)
- `colour`: white | black | grey | red | blue | green | brown | yellow
- `view`: front | rear | side | front34 | rear34
- `source`: own | team:<name> | stanford | compcars | screen
- `session`: place+date. **Split train/val by session**, never by crop. The same car photographed twice must never be in both train and val.

## Collection targets (crops, not photos)

| Group | Target crops/brand | Note |
|---|---|---|
| Chinese SUVs (Chery, Haval, Changan, Geely, Jetour, Exeed, Omoda, Jaecoo, Tank) | 150–250 | the hardest to separate; half the test |
| BYD, Li Auto, Zeekr, Leapmotor, Hongqi, JAC | 100–150 | |
| Chevrolet, Daewoo | 150+ | most common; the Nexia/Cobalt/Gentra types vary a lot |
| Kia, Hyundai, Toyota, Lada, Mercedes-Benz, BMW, VW, Nissan | 80–120 | distinct badges, easier |

Each brand needs every view, roughly 1/4 each of front, rear, side and three-quarter.
Colour: at least 40–50 crops of each rare colour (green, yellow, brown/beige, red).

Add 5–10% **screen photos**: show training images on a monitor and photograph them with your phone. The test photos are taken off a classroom screen.
