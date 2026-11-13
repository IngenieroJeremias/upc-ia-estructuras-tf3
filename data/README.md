# Datos: PEER NGA-West2 (RotD50, 5 % de amortiguamiento)

El flatfile **no se incluye en el repositorio** porque es un archivo público de PEER de ≈ 48 MB.

## Descarga

1. Ir a https://peer.berkeley.edu/research/data-sciences/databases (sección NGA-West2 flatfiles).
2. Descargar el flatfile RotD50 público (`NGA_West2_Flatfile_RotD50_d050_public_version.xlsx`).
3. Guardarlo exactamente en esta carpeta:

   ```
   data/NGA_West2_Flatfile_RotD50_d050_public_version.xlsx
   ```

## Columnas que usa `src/main_tf3_pipeline.py`

| Columna del flatfile | Nombre en el código | Uso |
|:---|:---|:---|
| `EQID` | `EQID` | Agrupación por sismo (GroupKFold) |
| `Earthquake Magnitude` | `Mw` | Predictor |
| `ClstD (km)` | `Rrup_km` | Predictor |
| `Vs30 (m/s) selected for analysis` | `Vs30_ms` | Predictor |
| `Northern CA/Southern CA - H11 Z1 (m)` | `Z10_m` | Predictor |
| `Mechanism Based on Rake Angle` | `Mechanism` | Predictor categórico |
| `T0.800S` | `Sa_08s_g` | Sa(0.80 s) en g → objetivo Sd (cm) |

## Resumen del conjunto

- 21,539 registros en el flatfile → **20,369 registros limpios** (Rrup ≤ 300 km) de **578 sismos**.
- El código `-999` se trata como valor faltante.
- Referencia: Ancheta, T. D., et al. (2014). NGA-West2 database. *Earthquake Spectra, 30*(3), 989–1005.
