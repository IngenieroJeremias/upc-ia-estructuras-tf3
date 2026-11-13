# TF3 — Resumen de resultados reales (PEER NGA-West2, RotD50, ξ = 5 %)

> Generado por `src/main_tf3_pipeline.py` el 2026-11-11 20:20. Python 3.12.16,
> scikit-learn 1.9.1, pandas 3.1.1, semilla 42. Tiempo de ejecución: 51 s.
> Fuente única: `data/NGA_West2_Flatfile_RotD50_d050_public_version.xlsx`. Sin datos sintéticos.

## 1. Datos

| Variable canónica | Columna exacta del flatfile | Nulos (-999/vacío) en crudo |
|:---|:---|---:|
| `EQID` | `EQID` | 11 |
| `Mw` | `Earthquake Magnitude` | 12 |
| `Rrup_km` | `ClstD (km)` | 46 |
| `Vs30_ms` | `Vs30 (m/s) selected for analysis` | 25 |
| `Z10_m` | `Northern CA/Southern CA - H11 Z1 (m)` | 4,953 |
| `Mechanism` | `Mechanism Based on Rake Angle` | 29 |
| `Sa_08s_g` | `T0.800S` | 204 |

Target: `Sd_08s_cm = Sa(T0.800S) · 9.80665 · (0.80 / 2π)² · 100`.

| Paso de depuración | Registros |
|:---|---:|
| Registros en el flatfile | 21,539 |
| EQID válido | 21,528 |
| Sa(T=0.80 s) > 0 (no -999) | 21,333 |
| Mw válida | 21,332 |
| Rrup válida | 21,298 |
| Rrup <= 300 km | 20,369 |

- **Registros limpios: 20,369** de 21,539 | **sismos únicos (EQID): 578**
- Registros por sismo: mediana 13, máximo 612; 87 sismos con un solo registro.
- Rangos: Mw 2.99–7.90; Rrup 0.05–299.8 km; Vs30 89–2100 m/s; Z1.0 0.0–3520.0 m.
- Sd(0.80 s): mediana 0.0208 cm, rango 6.31e-08–41.08 cm.
- Registros en campo cercano (Rrup < 20 km): 1,735.
- Mecanismos: Strike-Slip: 11,919, Reverse: 4,965, Reverse-Oblique: 1,959, Normal-Oblique: 936, Normal: 590.

| Predictor con nulos (imputación dentro del fold) | Nulos | % |
|:---|---:|---:|
| Vs30_ms | 6 | 0.03 % |
| Z10_m | 4,427 | 21.73 % |
| Mechanism | 0 | 0.00 % |

## 2. Validación: GroupKFold(5) por EQID

Partición por EVENTO SÍSMICO mediante GroupKFold agrupado por EQID. En una GMM el residuo total se descompone en un término inter-evento (común a todos los registros de un sismo: caída de esfuerzos, directividad, complejidad de la ruptura) y un término intra-evento (trayectoria y sitio). Una partición aleatoria por registro dejaría estaciones del mismo sismo a ambos lados del corte y el modelo aprendería el término inter-evento a partir de los registros vecinos, subestimando el error ante un sismo nuevo. Con GroupKFold cada sismo pertenece a un único fold de test y se evalúa exactamente una vez. Se verificó `set(train_eqid) ∩ set(test_eqid) = ∅` en los 5 folds.
La imputación (mediana + indicador para Vs30 y Z1.0; moda para el mecanismo), el `StandardScaler` y el
`OneHotEncoder` se ajustan dentro del `Pipeline` solo con el train de cada fold.

| Fold | Registros train | Registros test | Sismos train | Sismos test | Mw test | Rrup<20 test | Vs30 NaN train/test | Z1.0 NaN train/test | Mec. NaN train/test | Mediana Vs30 train (m/s) | Mediana Z1.0 train (m) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16295 | 4074 | 463 | 115 | 3.1-7.9 | 312 | 6/0 | 3897/530 | 0/0 | 418.0 | 60.0 |
| 2 | 16295 | 4074 | 463 | 115 | 3.0-7.7 | 397 | 6/0 | 3561/866 | 0/0 | 415.2 | 60.0 |
| 3 | 16295 | 4074 | 462 | 116 | 3.0-7.9 | 379 | 6/0 | 3238/1189 | 0/0 | 414.8 | 60.0 |
| 4 | 16295 | 4074 | 462 | 116 | 3.0-7.4 | 317 | 1/5 | 3728/699 | 0/0 | 419.0 | 60.0 |
| 5 | 16296 | 4073 | 462 | 116 | 3.0-7.3 | 330 | 5/1 | 3284/1143 | 0/0 | 416.5 | 60.0 |

## 3. Desempeño en test (media ± DE muestral sobre 5 folds)

| Rank | Modelo | RMSE (cm) | MAE (cm) | R² (cm) | R² ln(Sd) | σ_ln |
|:---|---:|---:|---:|---:|---:|---:|
| 1 | Random Forest | 1.112 ± 0.236 | 0.294 ± 0.091 | 0.566 ± 0.068 | 0.907 ± 0.007 | 0.857 ± 0.039 |
| 2 | Red Neuronal MLP | 1.113 ± 0.239 | 0.285 ± 0.100 | 0.565 ± 0.071 | 0.922 ± 0.004 | 0.792 ± 0.043 |
| 3 | Ridge (GMM lineal) | 1.979 ± 1.025 | 0.420 ± 0.208 | -0.474 ± 1.113 | 0.912 ± 0.007 | 0.839 ± 0.037 |

Mejor modelo por RMSE medio: **Random Forest**.

**Comparación pareada fold a fold frente a la GMM (RMSE en cm):**

| Modelo vs. GMM | Folds con menor RMSE | Reducción de RMSE (%) media ± DE |
|:---|---:|---:|
| Random Forest | 5/5 | 35.5 ± 23.9 |
| Red Neuronal MLP | 5/5 | 34.4 ± 27.7 |

## 4. Campo cercano (Rrup < 20 km), media ± DE sobre 5 folds

| Rank | Modelo | RMSE (cm) | MAE (cm) | R² (cm) | R² ln(Sd) | σ_ln |
|:---|---:|---:|---:|---:|---:|---:|
| 1 | Red Neuronal MLP | 3.031 ± 0.844 | 1.258 ± 0.543 | 0.517 ± 0.096 | 0.883 ± 0.024 | 0.788 ± 0.085 |
| 2 | Random Forest | 3.051 ± 0.760 | 1.278 ± 0.510 | 0.500 ± 0.122 | 0.863 ± 0.025 | 0.849 ± 0.107 |
| 3 | Ridge (GMM lineal) | 6.108 ± 3.347 | 2.566 ± 1.799 | -1.205 ± 1.844 | 0.850 ± 0.022 | 0.879 ± 0.086 |

**Métricas agregadas out-of-fold por tramo de distancia** (las 20,369 predicciones de test unidas;
sesgo ln > 0 = el modelo subestima):

| Tramo | Modelo | n | RMSE (cm) | MAE (cm) | R² (cm) | R² ln(Sd) | σ_ln | Sesgo ln (obs − pred) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Total | Ridge (GMM lineal) | 20369 | 2.181 | 0.42 | -0.584 | 0.913 | 0.845 | 0.002 |
| Total | Random Forest | 20369 | 1.132 | 0.294 | 0.574 | 0.909 | 0.863 | 0.035 |
| Total | Red Neuronal MLP | 20369 | 1.134 | 0.285 | 0.572 | 0.922 | 0.796 | -0.002 |
| Rrup < 20 km | Ridge (GMM lineal) | 1735 | 7.026 | 2.676 | -1.448 | 0.855 | 0.919 | -0.024 |
| Rrup < 20 km | Random Forest | 1735 | 3.152 | 1.289 | 0.507 | 0.87 | 0.864 | 0.107 |
| Rrup < 20 km | Red Neuronal MLP | 1735 | 3.142 | 1.27 | 0.51 | 0.89 | 0.799 | 0.031 |
| 20 ≤ Rrup < 100 km | Ridge (GMM lineal) | 9844 | 1.037 | 0.314 | 0.372 | 0.909 | 0.824 | 0.013 |
| 20 ≤ Rrup < 100 km | Random Forest | 9844 | 0.909 | 0.297 | 0.518 | 0.907 | 0.831 | 0.088 |
| 20 ≤ Rrup < 100 km | Red Neuronal MLP | 9844 | 0.928 | 0.291 | 0.498 | 0.915 | 0.797 | 0.001 |
| 100 ≤ Rrup ≤ 300 km | Ridge (GMM lineal) | 8790 | 0.279 | 0.092 | 0.522 | 0.901 | 0.852 | -0.005 |
| 100 ≤ Rrup ≤ 300 km | Random Forest | 8790 | 0.286 | 0.095 | 0.499 | 0.891 | 0.893 | -0.038 |
| 100 ≤ Rrup ≤ 300 km | Red Neuronal MLP | 8790 | 0.257 | 0.084 | 0.596 | 0.914 | 0.794 | -0.012 |

## 5. xAI — PFI del Random Forest (5 folds × 3 semillas = 15 corridas, 10 permutaciones c/u)

| Rank | Variable | ΔRMSE (cm) media ± DE | Δσ_ln media ± DE | Rango modal | % corridas en 1.er lugar |
|:---|---:|---:|---:|---:|---:|
| 1 | Mw | 0.704 ± 0.093 | 2.683 ± 0.104 | 1 | 93.3 |
| 2 | Rrup | 0.631 ± 0.120 | 0.938 ± 0.072 | 2 | 6.7 |
| 3 | Sitio (Vs30+Z1.0) | 0.043 ± 0.035 | 0.187 ± 0.007 | - | - |
| 4 | Vs30 | 0.031 ± 0.016 | 0.134 ± 0.011 | 3 | 0.0 |
| 5 | Z1.0 | 0.009 ± 0.031 | 0.044 ± 0.012 | 4 | 0.0 |
| 6 | Mecanismo | 0.006 ± 0.006 | 0.002 ± 0.002 | 5 | 0.0 |

- Orden medio de variables individuales: **Mw > Rrup > Vs30 > Z1.0 > Mecanismo**.
- Orden completo idéntico en 47 % de las corridas; top-2 idéntico en 93 %; mismo conjunto top-3 en 80 %.
- «Sitio (Vs30+Z1.0)» permuta ambas variables con la misma permutación (importancia conjunta del bloque de sitio).

## 6. Archivos generados

- `docs/TF3_metricas_por_fold.csv` — métricas por fold, modelo y subconjunto (Total / Rrup<20km).
- `docs/fig_tramo4_eda_fisico.png`, `docs/fig_tramo5_benchmark_ridge_rf.png`, `docs/fig_tramo6_benchmark_xai.png` — 300 DPI.
