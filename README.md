# Predicción de la respuesta sísmica espectral con IA (PEER NGA-West2)

**Trabajo Final (TF3), Aplicaciones de IA en Estructuras, Postgrado de Ingeniería Civil, UPC**  
**Docente:** Ing. Kurt Soncco Sinchi  
**Grupo 03:**

- Franklin Fustamante Sánchez (e201568658)
- Carlos David Vargas Marin (e202527282)
- Abrahán Ramirez More (e202527900)
- Jheremy Richard Molleapaza Infantes (e202528006)

---

## 1. Objetivo

Predecir el **desplazamiento espectral elástico Sd (cm)** de un oscilador de un grado de libertad (SDOF) con
**T = 0.80 s** y **ξ = 5 %** a partir de parámetros sísmicos del catálogo **PEER NGA-West2 (RotD50)**.
Se comparan tres modelos:

| Modelo | Rol | Hiperparámetros |
|:---|:---|:---|
| Ridge con forma funcional de GMM | Línea base física | `alpha = 1.0`; términos Mw, Mw², ln√(Rrup² + h²) con h = 6 km, ln Vs30 |
| Random Forest | Contraste y base de la auditoría xAI | `n_estimators = 100`, `max_depth = 8`, `random_state = 42` |
| **Red neuronal MLP** | **Modelo final** | capas `(64, 32)`, ReLU, Adam, `early_stopping = True`, `max_iter = 500` |

Los tres modelos se entrenan sobre **ln(Sd)** (`TransformedTargetRegressor`) y devuelven Sd en cm.

> Sd es una respuesta espectral elástica, no una deriva ni el desplazamiento real de una edificación.

**Variables predictoras:** Mw, Rrup (km), Vs30 (m/s), Z1.0 (m) y mecanismo focal.  
**Variable objetivo:** `Sd = Sa(g) · 9.80665 · (T / 2π)² · 100` [cm].

## 2. Metodología (resumen)

1. **Datos reales:** se lee el flatfile oficial NGA-West2, sin datos sintéticos. El código `-999` se convierte en `NaN`.
2. **Depuración:** EQID, Mw, Rrup y Sa(0.80 s) válidos, y Rrup ≤ 300 km. Quedan **20,369 registros de 578 sismos**.
3. **Partición por evento:** `GroupKFold(n_splits=5)` agrupado por `EQID`. Se verifica que ningún sismo aparezca a la vez en train y test, para evitar la fuga del término inter-evento.
4. **Pipeline sin fuga:** imputación (mediana + indicador), `StandardScaler` y `OneHotEncoder` se ajustan solo con el train de cada fold.
5. **Métricas:** RMSE, MAE y R² en cm; R² y σ en ln(Sd). Se reportan en total y en campo cercano (Rrup < 20 km).
6. **Explicabilidad (marco PCS, Yu & Barter):** Permutation Feature Importance del Random Forest en 5 folds × 3 semillas = 15 corridas.

## 3. Resultados principales (test, media ± DE sobre 5 folds)

| Modelo | RMSE (cm) | MAE (cm) | R² (cm) | R² ln(Sd) |
|:---|---:|---:|---:|---:|
| Random Forest | **1.112 ± 0.236** | 0.294 ± 0.091 | **0.566 ± 0.068** | 0.907 ± 0.007 |
| **Red neuronal MLP (final)** | 1.113 ± 0.239 | **0.285 ± 0.100** | 0.565 ± 0.071 | **0.922 ± 0.004** |
| Ridge (GMM lineal) | 1.979 ± 1.025 | 0.420 ± 0.208 | −0.474 ± 1.113 | 0.912 ± 0.007 |

- Random Forest y MLP empatan en RMSE (1.112 vs. 1.113 cm). Se elige la **MLP como modelo final** porque tiene menor σ_ln (0.792), menor sesgo que el RF en todos los tramos de distancia y el R² ln(Sd) más alto y estable (0.922 ± 0.004).
- Random Forest y MLP reducen el RMSE de la GMM lineal en los **5 de 5 folds** (≈ 35 % de reducción media).
- En **campo cercano** (Rrup < 20 km) el RMSE baja de 6.1 cm (GMM) a ≈ 3.0 cm (RF y MLP).
- **Importancia de variables (PFI):** Mw > Rrup > Vs30 > Z1.0 > mecanismo. Mw ocupa el primer lugar en el 93 % de las corridas.

El detalle completo está en [`docs/TF3_Resumen_Resultados_Reales.md`](docs/TF3_Resumen_Resultados_Reales.md) y en [`docs/TF3_metricas_por_fold.csv`](docs/TF3_metricas_por_fold.csv).

## 4. Estructura del repositorio

```
├── README.md
├── requirements.txt               Dependencias de Python
├── data/
│   └── README.md                  Cómo descargar el flatfile NGA-West2
├── src/
│   ├── main_tf3_pipeline.py       Código final del TF3 (datos → modelos → métricas → figuras)
│   └── eda_nga_west2.ipynb        Análisis exploratorio (T2)
└── docs/
    ├── Informe_Final_TF3_Grupo03.pdf / .docx       Informe final (incluye Bitácora de IA, Anexo A)
    ├── Presentacion_TF3_Grupo03.pdf                Diapositivas de la sustentación
    ├── BITACORA_IA_T2.md                           Bitácora de uso de IA
    ├── TF3_Resumen_Resultados_Reales.md            Resumen generado por el pipeline
    ├── TF3_metricas_por_fold.csv                   Métricas por fold y modelo
    └── fig_tramo4/5/6_*.png                        Figuras (300 DPI) generadas por el pipeline
```

## 5. Cómo ejecutar

Requiere Python 3.10 o superior (se probó con Python 3.12 y scikit-learn 1.9).

1. Descargar el flatfile y guardarlo como
   `data/NGA_West2_Flatfile_RotD50_d050_public_version.xlsx` (ver [`data/README.md`](data/README.md)).
2. Crear el entorno e instalar dependencias:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
   pip install -r requirements.txt
   ```

3. Ejecutar el pipeline (≈ 1 minuto):

   ```bash
   python src/main_tf3_pipeline.py
   ```

El script imprime la depuración de datos, las métricas por fold y el ranking de modelos. Además, regenera en
`docs/` las tres figuras, `TF3_metricas_por_fold.csv` y `TF3_Resumen_Resultados_Reales.md`. Con la semilla fija
(`SEED = 42`), los resultados coinciden con los del informe.

## 6. Uso ético de IA (Accountability)

El equipo usó asistentes de IA (Claude Code y Gemini) como apoyo para programar y redactar. El
**Anexo A del informe final** documenta los prompts, las iteraciones, la validación del código generado y las
afirmaciones de la IA que el equipo corrigió. El equipo revisó todos los datos, métricas y conclusiones
contra el flatfile real y la ejecución del código.

## 7. Referencias

- Ancheta, T. D., et al. (2014). NGA-West2 database. *Earthquake Spectra, 30*(3), 989–1005. https://doi.org/10.1193/070913EQS197M
- PEER NGA-West2 flatfiles: https://peer.berkeley.edu/research/data-sciences/databases
- Yu, B., & Barter, R. (2024). *Veridical Data Science*. MIT Press. https://vdsbook.com/
- Yu, B., & Kumbier, K. (2020). Veridical data science. *PNAS, 117*(8), 3920–3929. https://doi.org/10.1073/pnas.1901326117
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825–2830.
