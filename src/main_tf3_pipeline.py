#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
TF3 (Trabajo Final) - Pipeline científico sobre PEER NGA-West2 (RotD50, 5 %)
Postgrado de Ingeniería Civil - UPC | IA en Estructuras | Grupo 03
=============================================================================
Objetivo
    Predecir el desplazamiento espectral elástico Sd(T = 0.80 s, ξ = 5 %) en cm
    a partir de Mw, Rrup, Vs30, Z1.0 y el mecanismo focal, comparando:
        a) Ridge con forma funcional de GMM (baseline físico),
        b) Random Forest,
        c) Red neuronal MLP (64-32).

Garantías metodológicas
    1. DATOS REALES: lee exclusivamente el flatfile oficial PEER NGA-West2
       (data/NGA_West2_Flatfile_RotD50_d050_public_version.xlsx). No existe
       ninguna función de generación de datos sintéticos ni rutas alternativas.
    2. CÓDIGO -999 -> NaN en todas las columnas utilizadas.
    3. PIPELINE HERMÉTICO: SimpleImputer (mediana + indicador), StandardScaler y
       OneHotEncoder viven dentro del Pipeline de scikit-learn y se ajustan solo
       con el train de cada fold.
    4. PARTICIÓN POR EVENTO: GroupKFold(n_splits=5) agrupado por EQID, con
       verificación explícita de que ningún sismo está en train y test a la vez.
    5. xAI (marco PCS, Yu & Barter): Permutation Feature Importance del Random
       Forest en 5 folds x 3 semillas = 15 corridas de auditoría.

Salidas (docs/)
    fig_tramo4_eda_fisico.png, fig_tramo5_benchmark_ridge_rf.png,
    fig_tramo6_benchmark_xai.png (300 DPI), TF3_metricas_por_fold.csv y
    TF3_Resumen_Resultados_Reales.md.

Uso
    .venv/Scripts/python src/main_tf3_pipeline.py
=============================================================================
"""
from __future__ import annotations

import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from matplotlib.ticker import NullFormatter, ScalarFormatter
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# =============================================================================
# 0) CONFIGURACIÓN
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "NGA_West2_Flatfile_RotD50_d050_public_version.xlsx"
DOCS_DIR = REPO_ROOT / "docs"
OUT_CSV = DOCS_DIR / "TF3_metricas_por_fold.csv"
OUT_MD = DOCS_DIR / "TF3_Resumen_Resultados_Reales.md"
FIG_EDA = DOCS_DIR / "fig_tramo4_eda_fisico.png"
FIG_BENCH = DOCS_DIR / "fig_tramo5_benchmark_ridge_rf.png"
FIG_XAI = DOCS_DIR / "fig_tramo6_benchmark_xai.png"

SEED = 42
SEMILLAS_PCS = (42, 7, 2024)     # perturbación del modelo en la auditoría xAI
N_REP_PFI = 10                   # repeticiones por permutación
N_SPLITS = 5
G = 9.80665                      # m/s²
T_PERIODO = 0.80                 # s
H_SAT = 6.0                      # km, saturación en campo cercano: ln sqrt(R² + h²)
RRUP_MAX_KM = 300.0              # filtro ingenieril de distancia
R_CAMPO_CERCANO = 20.0           # km
NGA_NULL = -999
DPI = 300

# Cabeceras EXACTAS del flatfile PEER NGA-West2 -> nombre canónico
COLUMNAS_NGA = {
    "EQID": "EQID",
    "Earthquake Magnitude": "Mw",
    "ClstD (km)": "Rrup_km",
    "Vs30 (m/s) selected for analysis": "Vs30_ms",
    "Northern CA/Southern CA - H11 Z1 (m)": "Z10_m",
    "Mechanism Based on Rake Angle": "Mechanism",
    "T0.800S": "Sa_08s_g",
}
# Códigos de "Mechanism Based on Rake Angle" (PEER NGA-West2)
MECH_MAP = {0: "Strike-Slip", 1: "Normal", 2: "Reverse", 3: "Reverse-Oblique", 4: "Normal-Oblique"}

NUM_COLS = ["Mw", "Rrup_km", "Vs30_ms", "Z10_m"]
CAT_COLS = ["Mechanism"]
GMM_COLS = ["Mw", "Rrup_km", "Vs30_ms"]
MODELOS = ["Ridge (GMM lineal)", "Random Forest", "Red Neuronal MLP"]

# Métricas: clave ASCII (CSV) -> etiqueta legible
METRICAS = {"RMSE_cm": "RMSE (cm)", "MAE_cm": "MAE (cm)", "R2_cm": "R² (cm)",
            "R2_lnSd": "R² ln(Sd)", "sigma_ln": "σ_ln"}

FOLD_JUSTIFICACION = (
    "Partición por EVENTO SÍSMICO mediante GroupKFold agrupado por EQID. En una GMM el residuo "
    "total se descompone en un término inter-evento (común a todos los registros de un sismo: "
    "caída de esfuerzos, directividad, complejidad de la ruptura) y un término intra-evento "
    "(trayectoria y sitio). Una partición aleatoria por registro dejaría estaciones del mismo "
    "sismo a ambos lados del corte y el modelo aprendería el término inter-evento a partir de "
    "los registros vecinos, subestimando el error ante un sismo nuevo. Con GroupKFold cada sismo "
    "pertenece a un único fold de test y se evalúa exactamente una vez."
)


# =============================================================================
# 1) UTILIDADES
# =============================================================================
def banner(titulo: str) -> None:
    print("\n" + "=" * 79 + f"\n{titulo}\n" + "=" * 79)


def pm(media: float, de: float, dec: int = 3) -> str:
    return f"{media:.{dec}f} ± {de:.{dec}f}"


def tabla_md(df: pd.DataFrame) -> str:
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join([":---"] + ["---:"] * (len(df.columns) - 1)) + "|"
    filas = ["| " + " | ".join(map(str, f)) + " |" for f in df.itertuples(index=False)]
    return "\n".join([cab, sep, *filas])


# =============================================================================
# 2) CARGA Y DEPURACIÓN DEL FLATFILE REAL
# =============================================================================
def cargar_flatfile(ruta: Path = DATA_PATH, rrup_max: float = RRUP_MAX_KM) -> tuple[pd.DataFrame, dict]:
    """Lee las 7 columnas exactas del flatfile, convierte -999 en NaN, filtra y calcula Sd.

    Aquí NO se imputa ni se escala nada: todo lo que aprende de los datos va en el Pipeline.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el flatfile oficial: {ruta}")
    with warnings.catch_warnings():   # aviso benigno de openpyxl sobre extensiones del .xlsx de PEER
        warnings.filterwarnings("ignore", message="Unknown extension", category=UserWarning)
        crudo = pd.read_excel(ruta, usecols=list(COLUMNAS_NGA), engine="openpyxl")
    faltan = set(COLUMNAS_NGA) - set(crudo.columns)
    if faltan:
        raise KeyError(f"Cabeceras ausentes en el flatfile: {sorted(faltan)}")
    df = crudo.rename(columns=COLUMNAS_NGA)[list(COLUMNAS_NGA.values())].copy()

    # Tipos y código de nulo PEER (-999) -> NaN
    df["EQID"] = df["EQID"].astype(str).str.strip()
    for c in ["Mw", "Rrup_km", "Vs30_ms", "Z10_m", "Mechanism", "Sa_08s_g"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        df.loc[df[c] == NGA_NULL, c] = np.nan
    df.loc[df["EQID"].isin(["", "nan", "None", str(NGA_NULL)]), "EQID"] = np.nan
    nulos_crudo = {c: int(df[c].isna().sum()) for c in df.columns}
    df["Mechanism"] = df["Mechanism"].map(MECH_MAP).astype(object)   # código no listado -> NaN

    traza = [("Registros en el flatfile", len(df))]
    df = df[df["EQID"].notna()];                          traza.append(("EQID válido", len(df)))
    df = df[df["Sa_08s_g"] > 0];                          traza.append(("Sa(T=0.80 s) > 0 (no -999)", len(df)))
    df = df[df["Mw"].notna()];                            traza.append(("Mw válida", len(df)))
    df = df[df["Rrup_km"].notna() & (df["Rrup_km"] >= 0)]; traza.append(("Rrup válida", len(df)))
    df = df[df["Rrup_km"] <= rrup_max];                   traza.append((f"Rrup <= {rrup_max:g} km", len(df)))

    # Target físico: Sd = Sa·g·(T/2π)²  [m] -> cm
    df["Sd_08s_cm"] = df["Sa_08s_g"] * G * (T_PERIODO / (2 * np.pi)) ** 2 * 100
    df = df.reset_index(drop=True)

    regs = df.groupby("EQID").size()
    info = {
        "archivo": ruta.name, "n_crudo": len(crudo), "n_final": len(df),
        "n_eventos": int(df["EQID"].nunique()), "traza": traza, "nulos_crudo": nulos_crudo,
        "nulos_final": {c: int(df[c].isna().sum()) for c in ["Vs30_ms", "Z10_m", "Mechanism"]},
        "regs_mediana": float(regs.median()), "regs_max": int(regs.max()),
        "eventos_1_reg": int((regs == 1).sum()),
        "n_campo_cercano": int((df["Rrup_km"] < R_CAMPO_CERCANO).sum()),
        "rangos": {c: (float(df[c].min()), float(df[c].max()))
                   for c in ["Mw", "Rrup_km", "Vs30_ms", "Z10_m", "Sd_08s_cm"]},
        "sd_mediana": float(df["Sd_08s_cm"].median()),
        "mecanismos": df["Mechanism"].value_counts(dropna=False).rename(index={np.nan: "NaN"}).to_dict(),
    }
    if info["n_eventos"] < N_SPLITS:
        raise ValueError(f"Se requieren al menos {N_SPLITS} sismos para GroupKFold.")
    return df, info


def imprimir_datos(info: dict) -> None:
    banner("TRAMO 1 | FLATFILE REAL PEER NGA-West2 (RotD50, ξ = 5 %)")
    print(f"Archivo: data/{info['archivo']}  ({info['n_crudo']:,} registros)")
    print("\nMapeo exacto de columnas:")
    for orig, canon in COLUMNAS_NGA.items():
        print(f"  {canon:<10} <- '{orig}'  (nulos -999/vacíos en crudo: {info['nulos_crudo'][canon]:,})")
    print("\nTrazabilidad de la depuración:")
    for paso, n in info["traza"]:
        print(f"  {paso:<30} {n:>8,}")
    print("\nNulos en predictores (se imputan DENTRO de cada fold):")
    for c, n in info["nulos_final"].items():
        print(f"  {c:<10} {n:>6,} ({100 * n / info['n_final']:5.2f} %)")
    r = info["rangos"]
    print(f"\nDataset limpio: {info['n_final']:,} registros | {info['n_eventos']} sismos únicos | "
          f"Mw {r['Mw'][0]:.2f}-{r['Mw'][1]:.2f} | Rrup {r['Rrup_km'][0]:.2f}-{r['Rrup_km'][1]:.1f} km")
    print(f"Registros por sismo: mediana {info['regs_mediana']:.0f}, máximo {info['regs_max']}, "
          f"{info['eventos_1_reg']} sismos con 1 registro | Rrup < {R_CAMPO_CERCANO:g} km: "
          f"{info['n_campo_cercano']:,} registros")


# =============================================================================
# 3) MODELOS (todo preprocesamiento que aprende de los datos vive aquí)
# =============================================================================
def terminos_gmm(X: np.ndarray) -> np.ndarray:
    """[Mw, Rrup, Vs30, indicadores...] -> [Mw, Mw², ln√(Rrup²+h²), ln Vs30, indicadores...].

    Transformación determinista (sin parámetros ajustables): no introduce fuga.
    """
    X = np.asarray(X, dtype=float)
    mw, rrup, vs30 = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([mw, mw ** 2, np.log(np.sqrt(rrup ** 2 + H_SAT ** 2)),
                            np.log(np.clip(vs30, 50.0, None)), X[:, 3:]])


def _bloque_mecanismo() -> tuple:
    return ("mec", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CAT_COLS)


def construir_modelos(seed: int = SEED) -> dict[str, TransformedTargetRegressor]:
    """Tres modelos entrenados en ln(Sd); predict() devuelve Sd en cm."""
    prep_gmm = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("gmm", FunctionTransformer(terminos_gmm)),
            ("scaler", StandardScaler()),
        ]), GMM_COLS),
        _bloque_mecanismo(),
    ])
    prep_ml = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]), NUM_COLS),
        _bloque_mecanismo(),
    ])

    def en_log(pipe: Pipeline) -> TransformedTargetRegressor:
        return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp,
                                          check_inverse=False)

    return {
        "Ridge (GMM lineal)": en_log(Pipeline([("prep", prep_gmm), ("ridge", Ridge(alpha=1.0))])),
        "Random Forest": en_log(Pipeline([
            ("prep", prep_ml),
            ("rf", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1))])),
        "Red Neuronal MLP": en_log(Pipeline([
            ("prep", prep_ml),
            ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500,
                                 early_stopping=True, random_state=seed))])),
    }


def metricas(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    y_hat = np.clip(y_hat, 1e-12, None)
    res_ln = np.log(y) - np.log(y_hat)
    return {"RMSE_cm": float(np.sqrt(mean_squared_error(y, y_hat))),
            "MAE_cm": float(mean_absolute_error(y, y_hat)),
            "R2_cm": float(r2_score(y, y_hat)),
            "R2_lnSd": float(r2_score(np.log(y), np.log(y_hat))),
            "sigma_ln": float(np.std(res_ln, ddof=1)),
            "sesgo_ln": float(res_ln.mean())}


# =============================================================================
# 4) VALIDACIÓN CRUZADA POR EVENTO
# =============================================================================
def particion_por_evento(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold(5) por EQID con auditoría explícita anti-fuga."""
    groups = df["EQID"].to_numpy()
    splits = list(GroupKFold(n_splits=N_SPLITS).split(df, groups=groups))
    vistos = []
    for k, (tr, te) in enumerate(splits, 1):
        train_eqid, test_eqid = groups[tr], groups[te]
        assert set(train_eqid).intersection(set(test_eqid)) == set(), f"Fuga de EQID en el fold {k}"
        vistos.extend(set(test_eqid))
    assert len(vistos) == len(set(vistos)) == len(set(groups)), "Cada sismo debe ir a test una sola vez"
    return splits


def validacion_cruzada(df: pd.DataFrame, splits: list) -> dict:
    """Entrena y evalúa los 3 modelos en los 5 folds; devuelve métricas, OOF y auditoría."""
    banner("TRAMO 2 | GroupKFold(5) POR EQID: PARTICIÓN POR EVENTO SÍSMICO")
    print("✅ set(train_eqid) ∩ set(test_eqid) = ∅ en los 5 folds; cada sismo se evalúa una sola vez.")

    X, y = df[NUM_COLS + CAT_COLS], df["Sd_08s_cm"].to_numpy()
    cerca = (df["Rrup_km"] < R_CAMPO_CERCANO).to_numpy()
    oof = {m: np.full(len(y), np.nan) for m in MODELOS}
    filas, folds, mdi = [], [], []

    banner("TRAMO 3 | ENTRENAMIENTO Y EVALUACIÓN OUT-OF-FOLD (3 MODELOS x 5 FOLDS)")
    for k, (tr, te) in enumerate(splits, 1):
        for nombre, modelo in construir_modelos().items():   # instancias nuevas en cada fold
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                modelo.fit(X.iloc[tr], y[tr])
            y_hat = modelo.predict(X.iloc[te])
            seg = time.perf_counter() - t0
            oof[nombre][te] = y_hat
            m_tot = metricas(y[te], y_hat)
            filas.append({"Fold": k, "Modelo": nombre, "Subconjunto": "Total", "n_test": len(te),
                          **m_tot, "Tiempo_s": round(seg, 2)})
            nc = cerca[te]
            filas.append({"Fold": k, "Modelo": nombre, "Subconjunto": f"Rrup<{R_CAMPO_CERCANO:g}km",
                          "n_test": int(nc.sum()), **metricas(y[te][nc], y_hat[nc]), "Tiempo_s": np.nan})
            print(f"  Fold {k} | {nombre:<19} RMSE {m_tot['RMSE_cm']:7.3f} cm | MAE {m_tot['MAE_cm']:6.3f} cm"
                  f" | R² {m_tot['R2_cm']:6.3f} | R² ln {m_tot['R2_lnSd']:6.3f} | σ_ln {m_tot['sigma_ln']:5.3f}"
                  f" | {seg:5.1f} s")

            if nombre == "Random Forest":
                pipe = modelo.regressor_
                imp = pipe.named_steps["prep"].named_transformers_["num"].named_steps["imputer"]
                med = dict(zip(NUM_COLS, imp.statistics_))
                mdi.append(pd.Series(pipe.named_steps["rf"].feature_importances_,
                                     index=pipe.named_steps["prep"].get_feature_names_out()))
        trn, tst = df.iloc[tr], df.iloc[te]
        folds.append({
            "Fold": k, "Registros train": len(tr), "Registros test": len(te),
            "Sismos train": trn["EQID"].nunique(), "Sismos test": tst["EQID"].nunique(),
            "Mw test": f"{tst['Mw'].min():.1f}-{tst['Mw'].max():.1f}",
            "Rrup<20 test": int((tst["Rrup_km"] < R_CAMPO_CERCANO).sum()),
            "Vs30 NaN train/test": f"{trn['Vs30_ms'].isna().sum()}/{tst['Vs30_ms'].isna().sum()}",
            "Z1.0 NaN train/test": f"{trn['Z10_m'].isna().sum()}/{tst['Z10_m'].isna().sum()}",
            "Mec. NaN train/test": f"{trn['Mechanism'].isna().sum()}/{tst['Mechanism'].isna().sum()}",
            "Mediana Vs30 train (m/s)": round(float(med["Vs30_ms"]), 1),
            "Mediana Z1.0 train (m)": round(float(med["Z10_m"]), 1),
        })
        print()

    por_fold = pd.DataFrame(filas)
    tabla_folds = pd.DataFrame(folds)
    print(tabla_folds.to_string(index=False))
    return {"por_fold": por_fold, "tabla_folds": tabla_folds, "oof": oof,
            "mdi": pd.concat(mdi, axis=1).fillna(0.0)}


def resumen_media_de(por_fold: pd.DataFrame, subconjunto: str = "Total") -> pd.DataFrame:
    """Media ± DE muestral (ddof=1) sobre los 5 folds, ordenado por RMSE medio."""
    sub = por_fold[por_fold["Subconjunto"] == subconjunto]
    agg = sub.groupby("Modelo")[list(METRICAS) + ["sesgo_ln"]].agg(["mean", "std"])
    return agg.sort_values(("RMSE_cm", "mean"))


def metricas_oof(df: pd.DataFrame, oof: dict) -> pd.DataFrame:
    """Métricas agregadas out-of-fold (todas las predicciones de test unidas) por tramo de Rrup."""
    y, r = df["Sd_08s_cm"].to_numpy(), df["Rrup_km"].to_numpy()
    tramos = [("Total", np.ones_like(r, bool)),
              (f"Rrup < {R_CAMPO_CERCANO:g} km", r < R_CAMPO_CERCANO),
              (f"{R_CAMPO_CERCANO:g} ≤ Rrup < 100 km", (r >= R_CAMPO_CERCANO) & (r < 100)),
              (f"100 ≤ Rrup ≤ {RRUP_MAX_KM:g} km", r >= 100)]
    filas = []
    for etiqueta, m in tramos:
        for nombre in MODELOS:
            filas.append({"Tramo": etiqueta, "Modelo": nombre, "n": int(m.sum()),
                          **metricas(y[m], oof[nombre][m])})
    return pd.DataFrame(filas)


# =============================================================================
# 5) AUDITORÍA xAI: PFI DEL RANDOM FOREST, 5 FOLDS x 3 SEMILLAS (MARCO PCS)
# =============================================================================
GRUPOS_PFI = {"Rrup": ["Rrup_km"], "Mw": ["Mw"], "Vs30": ["Vs30_ms"], "Z1.0": ["Z10_m"],
              "Mecanismo": ["Mechanism"], "Sitio (Vs30+Z1.0)": ["Vs30_ms", "Z10_m"]}
INDIVIDUALES = ["Rrup", "Mw", "Vs30", "Z1.0", "Mecanismo"]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def auditoria_pfi(df: pd.DataFrame, splits: list) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """ΔRMSE (cm) y Δσ_ln al permutar cada variable en el test de cada fold.

    Las columnas se permutan en bruto (con sus NaN) y el Pipeline del fold vuelve a imputarlas,
    de modo que el indicador de imputación se permuta junto con la variable.
    """
    banner(f"TRAMO 4 | xAI (PCS): PFI DEL RANDOM FOREST, {N_SPLITS} FOLDS x {len(SEMILLAS_PCS)} SEMILLAS")
    X, y = df[NUM_COLS + CAT_COLS], df["Sd_08s_cm"].to_numpy()
    registros, rankings = [], []
    for semilla in SEMILLAS_PCS:
        for k, (tr, te) in enumerate(splits, 1):
            rng = np.random.default_rng([semilla, k])
            modelo = construir_modelos(seed=semilla)["Random Forest"]
            modelo.fit(X.iloc[tr], y[tr])
            X_te, y_te = X.iloc[te].reset_index(drop=True), y[te]
            p0 = modelo.predict(X_te)
            base_cm, base_ln = rmse(y_te, p0), float(np.std(np.log(y_te / p0), ddof=1))
            delta = {}
            for g, cols in GRUPOS_PFI.items():
                d_cm, d_ln = [], []
                for _ in range(N_REP_PFI):
                    perm = rng.permutation(len(X_te))
                    Xp = X_te.copy()
                    for c in cols:                        # misma permutación para todo el grupo
                        Xp[c] = X_te[c].to_numpy()[perm]
                    p = modelo.predict(Xp)
                    d_cm.append(rmse(y_te, p) - base_cm)
                    d_ln.append(float(np.std(np.log(y_te / p), ddof=1)) - base_ln)
                delta[g] = float(np.mean(d_cm))
                registros.append({"Semilla": semilla, "Fold": k, "Variable": g,
                                  "dRMSE_cm": delta[g], "dSigma_ln": float(np.mean(d_ln))})
            orden = sorted(INDIVIDUALES, key=lambda g: -delta[g])
            rankings.append(orden)
            print(f"  Semilla {semilla:>4} | Fold {k} | " + " > ".join(orden))

    pfi = pd.DataFrame(registros)
    tabla = pfi.groupby("Variable")[["dRMSE_cm", "dSigma_ln"]].agg(["mean", "std"])
    tabla.columns = [f"{a}_{b}" for a, b in tabla.columns]          # columnas planas
    tabla = tabla.sort_values("dRMSE_cm_mean", ascending=False)
    pos = {g: [r.index(g) + 1 for r in rankings] for g in INDIVIDUALES}
    tabla["Rango modal"] = [int(pd.Series(pos[g]).mode()[0]) if g in pos else "-" for g in tabla.index]
    tabla["% corridas en 1.er lugar"] = [round(100 * np.mean([r[0] == g for r in rankings]), 1)
                                         if g in pos else "-" for g in tabla.index]
    orden_medio = [g for g in tabla.index if g in INDIVIDUALES]
    est = {
        "n_corridas": len(rankings),
        "orden_medio": orden_medio,
        "pct_orden_medio_exacto": 100 * np.mean([r == orden_medio for r in rankings]),
        "pct_top2_igual": 100 * np.mean([r[:2] == orden_medio[:2] for r in rankings]),
        "pct_top3_set": 100 * np.mean([set(r[:3]) == set(orden_medio[:3]) for r in rankings]),
    }
    print(f"\n  Orden medio: {' > '.join(orden_medio)}")
    print(f"  Orden completo idéntico en {est['pct_orden_medio_exacto']:.0f} % | top-2 idéntico en "
          f"{est['pct_top2_igual']:.0f} % | top-3 (conjunto) en {est['pct_top3_set']:.0f} % "
          f"de {est['n_corridas']} corridas")
    return pfi, tabla, est


# =============================================================================
# 6) FIGURAS (300 DPI)
# =============================================================================
def estilo_figuras() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlepad": 10, "axes.labelsize": 11, "axes.labelcolor": "#333333",
        "xtick.color": "#555555", "ytick.color": "#555555", "grid.color": "#E6E6E6",
        "grid.linewidth": 0.7, "axes.edgecolor": "#BBBBBB", "legend.framealpha": 0.95,
        "legend.edgecolor": "#CCCCCC", "legend.facecolor": "white",
    })


VIR = plt.get_cmap("viridis")
C_A, C_B, C_C = VIR(0.15), VIR(0.80), VIR(0.50)
COLOR_MODELO = {"Ridge (GMM lineal)": C_A, "Random Forest": C_B, "Red Neuronal MLP": C_C}
CORTO = {"Ridge (GMM lineal)": "Ridge\n(GMM)", "Random Forest": "Random\nForest",
         "Red Neuronal MLP": "MLP\n(64-32)"}


def figura_eda(df: pd.DataFrame, splits: list, ruta: Path = FIG_EDA) -> plt.Figure:
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    ax00, ax01 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax10, ax11 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    sd, rr = df["Sd_08s_cm"], df["Rrup_km"].clip(lower=0.1)

    # (0,0) Atenuación
    sc = ax00.scatter(rr, sd, c=df["Mw"], cmap="viridis", s=5, alpha=0.5, linewidths=0)
    cb = fig.colorbar(sc, ax=ax00, pad=0.02, fraction=0.045)
    cb.set_label("Magnitud de momento, Mw")
    cb.outline.set_visible(False)
    bins = np.logspace(np.log10(rr.min()), np.log10(rr.max()), 16)
    tend = (pd.DataFrame({"R": rr, "Sd": sd, "b": pd.cut(rr, bins, include_lowest=True)})
            .groupby("b", observed=True).agg(R=("R", "median"), Sd=("Sd", "median"), n=("Sd", "size"))
            .query("n >= 20"))
    ax00.plot(tend["R"], tend["Sd"], color="#222222", lw=2, marker="o", ms=5, label="Mediana por bin de Rrup")
    b, a = np.polyfit(np.log10(rr), np.log10(sd), 1)
    rf = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 100)
    ax00.plot(rf, 10 ** a * rf ** b, color="#C0392B", lw=1.8, ls="--",
              label=f"Ajuste log-log: Sd ∝ R$^{{{b:.2f}}}$")
    ax00.axvline(R_CAMPO_CERCANO, color="#7F7F7F", ls=":", lw=1.2)
    ax00.set(xscale="log", yscale="log", xlabel="Distancia a la ruptura, $R_{rup}$ (km)",
             ylabel="$S_d$(T = 0.80 s, 5 %) (cm)")
    ax00.set_title("Atenuación: $S_d$ decae con $R_{rup}$ y crece con Mw", loc="left")
    ax00.legend(loc="lower left", fontsize=9)

    # (0,1) Efecto de sitio
    con_z1, sin_z1 = df[df["Z10_m"].notna()], df[df["Z10_m"].isna()]
    ax01.scatter(con_z1["Vs30_ms"], con_z1["Sd_08s_cm"], s=5, alpha=0.35, color=C_B, linewidths=0,
                 label=f"Z1.0 disponible (modelo H11), n = {len(con_z1):,}")
    ax01.scatter(sin_z1["Vs30_ms"], sin_z1["Sd_08s_cm"], s=5, alpha=0.5, color=C_A, linewidths=0,
                 label=f"Z1.0 = -999 (imputado en el Pipeline), n = {len(sin_z1):,}")
    y_top = sd.max() * 2
    for vs, txt in [(180, "D/E\n180"), (360, "C/D\n360"), (760, "B/C\n760")]:
        ax01.axvline(vs, color="#7F7F7F", ls="--", lw=1.1, zorder=0)
        ax01.text(vs * 1.04, y_top, txt, fontsize=8.5, color="#555555", va="top")
    ax01.set(xscale="log", yscale="log", xlabel="$V_{S30}$ (m/s)", ylabel="$S_d$(T = 0.80 s, 5 %) (cm)")
    ax01.set_ylim(top=y_top * 1.2)
    ax01.set_xticks([150, 200, 300, 500, 760, 1000, 2000])
    ax01.xaxis.set_major_formatter(ScalarFormatter())
    ax01.xaxis.set_minor_formatter(NullFormatter())
    ax01.set_title("Efecto de sitio: límites NEHRP / ASCE 7 y disponibilidad de Z1.0", loc="left")
    ax01.legend(loc="lower left", fontsize=9, markerscale=3)

    # (1,0) Correlación de Pearson en escala física (log)
    logs = pd.DataFrame({"Mw": df["Mw"], "ln Rrup": np.log(rr), "ln Vs30": np.log(df["Vs30_ms"]),
                         "ln Z1.0": np.log(df["Z10_m"].clip(lower=1)), "ln Sd": np.log(sd)})
    corr = logs.corr(method="pearson")   # pares completos (Z1.0 con NaN excluido por pares)
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool), k=1), cmap="coolwarm", vmin=-1,
                vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 11}, square=True,
                linewidths=2, linecolor="white", cbar_kws={"shrink": 0.8, "label": "Pearson r"}, ax=ax10)
    ax10.tick_params(axis="both", rotation=0)
    ax10.grid(False)
    ax10.set_title("Correlación de Pearson (variables en escala log)", loc="left")

    # (1,1) Auditoría GroupKFold, fold 1
    tr, te = splits[0]
    dtr, dte = df.iloc[tr], df.iloc[te]
    ax11.scatter(dtr["Rrup_km"].clip(lower=0.1), dtr["Mw"], s=5, alpha=0.3, color=C_A, linewidths=0,
                 label=f"TRAIN: {dtr['EQID'].nunique()} sismos, {len(dtr):,} registros")
    ax11.scatter(dte["Rrup_km"].clip(lower=0.1), dte["Mw"], s=9, alpha=0.8, color=C_B, marker="^",
                 linewidths=0, label=f"TEST: {dte['EQID'].nunique()} sismos, {len(dte):,} registros")
    n_comp = len(set(dtr["EQID"]) & set(dte["EQID"]))
    ax11.text(0.02, 0.97, f"EQID compartidos TRAIN ∩ TEST = {n_comp}", transform=ax11.transAxes,
              va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC"))
    ax11.set(xscale="log", xlabel="$R_{rup}$ (km)", ylabel="Mw")
    ax11.set_title("Auditoría GroupKFold (Fold 1): sismos completos en TRAIN o en TEST", loc="left")
    ax11.legend(loc="lower right", fontsize=9, markerscale=3)
    for ax in (ax00, ax01, ax11):
        sns.despine(ax=ax)

    fig.suptitle(f"PEER NGA-West2 (RotD50) — EDA físico de $S_d$(T = 0.80 s): {len(df):,} registros, "
                 f"{df['EQID'].nunique()} sismos", fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    return fig


def figura_benchmark(df: pd.DataFrame, cv: dict, ruta: Path = FIG_BENCH) -> plt.Figure:
    y = df["Sd_08s_cm"].to_numpy()
    oof_m = metricas_oof(df, cv["oof"]).query("Tramo == 'Total'").set_index("Modelo")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.05, 1]})

    for nombre, mk in [("Ridge (GMM lineal)", "o"), ("Random Forest", "^")]:
        p = cv["oof"][nombre]
        ax1.scatter(y, p, s=5, alpha=0.35, marker=mk, color=COLOR_MODELO[nombre], linewidths=0,
                    label=f"{nombre} (R² ln = {oof_m.loc[nombre, 'R2_lnSd']:.2f}, "
                          f"σ_ln = {oof_m.loc[nombre, 'sigma_ln']:.2f})")
    lim = [min(y.min(), *(cv["oof"][m].min() for m in MODELOS)) * 0.5,
           max(y.max(), *(cv["oof"][m].max() for m in MODELOS)) * 2]
    ax1.plot(lim, lim, color="#222222", lw=1.5, label="Línea 1:1")
    ax1.fill_between(lim, [v / 2 for v in lim], [v * 2 for v in lim], color="#999999", alpha=0.15,
                     label="Banda factor 2")
    ax1.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="$S_d$ medido (cm)",
            ylabel="$S_d$ predicho out-of-fold (cm)")
    ax1.set_title("Real vs. predicho: predicciones de test de los 5 folds (sismos no vistos)", loc="left")
    ax1.legend(loc="upper left", fontsize=9, markerscale=3)
    sns.despine(ax=ax1)

    etiq = {"num__Mw": "Mw", "num__Rrup_km": "Rrup", "num__Vs30_ms": "Vs30", "num__Z10_m": "Z1.0",
            "num__missingindicator_Z10_m": "Indicador Z1.0 imputado",
            "num__missingindicator_Vs30_ms": "Indicador Vs30 imputado"}
    mdi = cv["mdi"]
    resumen = pd.DataFrame({"media": mdi.mean(axis=1), "de": mdi.std(axis=1, ddof=1)}).sort_values("media")
    nombres = [etiq.get(i, i.replace("mec__Mechanism_", "Mec.: ")) for i in resumen.index]
    ax2.barh(nombres, resumen["media"], xerr=resumen["de"], color=VIR(0.15 + 0.65 * resumen["media"] /
             resumen["media"].max()), height=0.65, error_kw={"ecolor": "#444444", "capsize": 3})
    for yi, (v, s) in enumerate(zip(resumen["media"], resumen["de"])):
        ax2.text(v + s + 0.005, yi, f"{v:.3f}", va="center", fontsize=9, color="#333333")
    ax2.set(xlabel="Importancia MDI del Random Forest (media ± DE en 5 folds)",
            xlim=(0, (resumen["media"] + resumen["de"]).max() * 1.18))
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Random Forest: importancia interna (MDI, train)", loc="left")
    sns.despine(ax=ax2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    return fig


def figura_xai(por_fold: pd.DataFrame, pfi: pd.DataFrame, ruta: Path = FIG_XAI) -> plt.Figure:
    tot = por_fold[por_fold["Subconjunto"] == "Total"]
    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.1], wspace=0.3)
    gi = gs[0].subgridspec(1, 3, wspace=0.55)
    ejes = [fig.add_subplot(gi[i]) for i in range(3)]
    for ax, col, titulo in [(ejes[0], "RMSE_cm", "RMSE (cm) ↓"), (ejes[1], "R2_cm", "R² (cm) ↑"),
                            (ejes[2], "sigma_ln", "σ_ln ↓")]:
        g = tot.groupby("Modelo")[col].agg(["mean", "std"]).loc[MODELOS]
        bars = ax.bar([CORTO[m] for m in MODELOS], g["mean"], yerr=g["std"], capsize=4,
                      color=[COLOR_MODELO[m] for m in MODELOS], width=0.62, edgecolor="white",
                      error_kw={"ecolor": "#333333"})
        for b_, v, s in zip(bars, g["mean"], g["std"]):
            ax.text(b_.get_x() + b_.get_width() / 2, max(v, 0) + s, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=9.5, fontweight="bold", color="#333333")
        y_min = min(0.0, (g["mean"] - g["std"]).min() * 1.1)   # R² negativo (Ridge) visible
        ax.set_ylim(y_min, (g["mean"].clip(lower=0) + g["std"]).max() * 1.2)
        ax.axhline(0, color="#555555", lw=0.8)
        ax.set_title(f"{titulo}\nmedia ± DE, 5 folds", loc="left", fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)

    ax = fig.add_subplot(gs[1])
    orden = pfi.groupby("Variable")["dRMSE_cm"].mean().sort_values(ascending=False).index
    sns.boxplot(data=pfi, y="Variable", x="dRMSE_cm", order=orden, hue="Variable", hue_order=orden,
                palette="viridis_r", legend=False, width=0.6, fliersize=3, linewidth=1, ax=ax)
    sns.stripplot(data=pfi, y="Variable", x="dRMSE_cm", order=orden, color="#222222", size=3,
                  alpha=0.6, ax=ax)
    ax.axvline(0, color="#7F7F7F", ls="--", lw=1)
    ax.set(xlabel="Aumento de RMSE al permutar (cm)", ylabel="")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"xAI (PCS): PFI del Random Forest, {N_SPLITS} folds × {len(SEMILLAS_PCS)} semillas "
                 f"= {pfi.groupby(['Semilla', 'Fold']).ngroups} corridas", loc="left", fontsize=11)
    sns.despine(ax=ax)
    fig.suptitle("Benchmark Ridge vs. RF vs. MLP (GroupKFold por EQID) y estabilidad de la importancia",
                 fontsize=13.5, fontweight="bold", x=0.01, ha="left", y=1.03)
    fig.savefig(ruta, dpi=DPI, bbox_inches="tight")
    return fig


# =============================================================================
# 7) EXPORTACIÓN DE RESULTADOS
# =============================================================================
def _tabla_pm(res: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({"Modelo": res.index,
                      **{et: [pm(r[(k, "mean")], r[(k, "std")]) for _, r in res.iterrows()]
                         for k, et in METRICAS.items()}})
    t.insert(0, "Rank", range(1, len(t) + 1))
    return t


def comparacion_pareada(por_fold: pd.DataFrame) -> pd.DataFrame:
    piv = por_fold[por_fold["Subconjunto"] == "Total"].pivot(index="Fold", columns="Modelo", values="RMSE_cm")
    base = "Ridge (GMM lineal)"
    filas = []
    for m in [x for x in MODELOS if x != base]:
        d = 100 * (piv[base] - piv[m]) / piv[base]
        filas.append({"Modelo vs. GMM": m, "Folds con menor RMSE": f"{int((piv[m] < piv[base]).sum())}/{N_SPLITS}",
                      "Reducción de RMSE (%) media ± DE": pm(d.mean(), d.std(ddof=1), 1)})
    return pd.DataFrame(filas)


def exportar(info: dict, cv: dict, oof_tab: pd.DataFrame, pfi_tabla: pd.DataFrame,
             pfi_est: dict, t_total: float) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    por_fold = cv["por_fold"]
    por_fold.round(5).to_csv(OUT_CSV, index=False, encoding="utf-8")

    res_tot = resumen_media_de(por_fold, "Total")
    res_nf = resumen_media_de(por_fold, f"Rrup<{R_CAMPO_CERCANO:g}km")
    mejor = res_tot.index[0]
    r = info["rangos"]

    t_oof = oof_tab.copy()
    for c in list(METRICAS) + ["sesgo_ln"]:
        t_oof[c] = t_oof[c].round(3)
    t_oof = t_oof.rename(columns={**METRICAS, "sesgo_ln": "Sesgo ln (obs − pred)"})

    t_pfi = pd.DataFrame({
        "Rank": range(1, len(pfi_tabla) + 1),
        "Variable": pfi_tabla.index,
        "ΔRMSE (cm) media ± DE": [pm(a, b) for a, b in zip(pfi_tabla["dRMSE_cm_mean"],
                                                            pfi_tabla["dRMSE_cm_std"])],
        "Δσ_ln media ± DE": [pm(a, b) for a, b in zip(pfi_tabla["dSigma_ln_mean"],
                                                      pfi_tabla["dSigma_ln_std"])],
        "Rango modal": pfi_tabla["Rango modal"].to_numpy(),
        "% corridas en 1.er lugar": pfi_tabla["% corridas en 1.er lugar"].to_numpy(),
    })
    traza = "\n".join(f"| {p} | {n:,} |" for p, n in info["traza"])
    mapa = "\n".join(f"| `{c}` | `{o}` | {info['nulos_crudo'][c]:,} |" for o, c in COLUMNAS_NGA.items())
    nulos = "\n".join(f"| {c} | {n:,} | {100 * n / info['n_final']:.2f} % |" for c, n in info["nulos_final"].items())
    mecs = ", ".join(f"{k}: {v:,}" for k, v in info["mecanismos"].items())

    md = f"""# TF3 — Resumen de resultados reales (PEER NGA-West2, RotD50, ξ = 5 %)

> Generado por `src/main_tf3_pipeline.py` el {datetime.now():%Y-%m-%d %H:%M}. Python {sys.version.split()[0]},
> scikit-learn {sklearn.__version__}, pandas {pd.__version__}, semilla {SEED}. Tiempo de ejecución: {t_total:.0f} s.
> Fuente única: `data/{info['archivo']}`. Sin datos sintéticos.

## 1. Datos

| Variable canónica | Columna exacta del flatfile | Nulos (-999/vacío) en crudo |
|:---|:---|---:|
{mapa}

Target: `Sd_08s_cm = Sa(T0.800S) · 9.80665 · (0.80 / 2π)² · 100`.

| Paso de depuración | Registros |
|:---|---:|
{traza}

- **Registros limpios: {info['n_final']:,}** de {info['n_crudo']:,} | **sismos únicos (EQID): {info['n_eventos']}**
- Registros por sismo: mediana {info['regs_mediana']:.0f}, máximo {info['regs_max']}; {info['eventos_1_reg']} sismos con un solo registro.
- Rangos: Mw {r['Mw'][0]:.2f}–{r['Mw'][1]:.2f}; Rrup {r['Rrup_km'][0]:.2f}–{r['Rrup_km'][1]:.1f} km; Vs30 {r['Vs30_ms'][0]:.0f}–{r['Vs30_ms'][1]:.0f} m/s; Z1.0 {r['Z10_m'][0]:.1f}–{r['Z10_m'][1]:.1f} m.
- Sd(0.80 s): mediana {info['sd_mediana']:.4f} cm, rango {r['Sd_08s_cm'][0]:.2e}–{r['Sd_08s_cm'][1]:.2f} cm.
- Registros en campo cercano (Rrup < {R_CAMPO_CERCANO:g} km): {info['n_campo_cercano']:,}.
- Mecanismos: {mecs}.

| Predictor con nulos (imputación dentro del fold) | Nulos | % |
|:---|---:|---:|
{nulos}

## 2. Validación: GroupKFold(5) por EQID

{FOLD_JUSTIFICACION} Se verificó `set(train_eqid) ∩ set(test_eqid) = ∅` en los 5 folds.
La imputación (mediana + indicador para Vs30 y Z1.0; moda para el mecanismo), el `StandardScaler` y el
`OneHotEncoder` se ajustan dentro del `Pipeline` solo con el train de cada fold.

{tabla_md(cv['tabla_folds'])}

## 3. Desempeño en test (media ± DE muestral sobre 5 folds)

{tabla_md(_tabla_pm(res_tot))}

Mejor modelo por RMSE medio: **{mejor}**.

**Comparación pareada fold a fold frente a la GMM (RMSE en cm):**

{tabla_md(comparacion_pareada(por_fold))}

## 4. Campo cercano (Rrup < {R_CAMPO_CERCANO:g} km), media ± DE sobre 5 folds

{tabla_md(_tabla_pm(res_nf))}

**Métricas agregadas out-of-fold por tramo de distancia** (las {info['n_final']:,} predicciones de test unidas;
sesgo ln > 0 = el modelo subestima):

{tabla_md(t_oof)}

## 5. xAI — PFI del Random Forest ({N_SPLITS} folds × {len(SEMILLAS_PCS)} semillas = {pfi_est['n_corridas']} corridas, {N_REP_PFI} permutaciones c/u)

{tabla_md(t_pfi)}

- Orden medio de variables individuales: **{' > '.join(pfi_est['orden_medio'])}**.
- Orden completo idéntico en {pfi_est['pct_orden_medio_exacto']:.0f} % de las corridas; top-2 idéntico en {pfi_est['pct_top2_igual']:.0f} %; mismo conjunto top-3 en {pfi_est['pct_top3_set']:.0f} %.
- «Sitio (Vs30+Z1.0)» permuta ambas variables con la misma permutación (importancia conjunta del bloque de sitio).

## 6. Archivos generados

- `docs/TF3_metricas_por_fold.csv` — métricas por fold, modelo y subconjunto (Total / Rrup<{R_CAMPO_CERCANO:g}km).
- `docs/{FIG_EDA.name}`, `docs/{FIG_BENCH.name}`, `docs/{FIG_XAI.name}` — {DPI} DPI.
"""
    OUT_MD.write_text(md, encoding="utf-8")


def imprimir_resumen(cv: dict, oof_tab: pd.DataFrame, pfi_tabla: pd.DataFrame) -> None:
    por_fold = cv["por_fold"]
    banner("TRAMO 5 | RESUMEN: MEDIA ± DE EN LOS 5 FOLDS (TEST)")
    print(_tabla_pm(resumen_media_de(por_fold, "Total")).to_string(index=False))
    print("\n" + comparacion_pareada(por_fold).to_string(index=False))
    banner(f"CAMPO CERCANO (Rrup < {R_CAMPO_CERCANO:g} km): MEDIA ± DE EN LOS 5 FOLDS")
    print(_tabla_pm(resumen_media_de(por_fold, f"Rrup<{R_CAMPO_CERCANO:g}km")).to_string(index=False))
    print("\nOut-of-fold por tramo de distancia:")
    print(oof_tab.round(3).to_string(index=False))
    banner("RANKING PFI (Random Forest)")
    print(pfi_tabla.round(4).to_string())


# =============================================================================
# 8) PUNTO DE ENTRADA
# =============================================================================
def main() -> int:
    for stream in (sys.stdout, sys.stderr):          # consola Windows: ±, ², ∩, σ
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    matplotlib.use("Agg")
    t0 = time.perf_counter()
    banner("TF3 | PIPELINE PEER NGA-West2 | Grupo 03 - Postgrado de Ingeniería Civil UPC")
    print(f"scikit-learn {sklearn.__version__} | pandas {pd.__version__} | numpy {np.__version__} | semilla {SEED}")

    df, info = cargar_flatfile()
    imprimir_datos(info)
    splits = particion_por_evento(df)
    cv = validacion_cruzada(df, splits)
    oof_tab = metricas_oof(df, cv["oof"])
    pfi, pfi_tabla, pfi_est = auditoria_pfi(df, splits)

    estilo_figuras()
    for f in (figura_eda(df, splits), figura_benchmark(df, cv), figura_xai(cv["por_fold"], pfi)):
        plt.close(f)

    imprimir_resumen(cv, oof_tab, pfi_tabla)
    t_total = time.perf_counter() - t0
    exportar(info, cv, oof_tab, pfi_tabla, pfi_est, t_total)
    banner("SALIDAS")
    for p in (OUT_CSV, OUT_MD, FIG_EDA, FIG_BENCH, FIG_XAI):
        print(f"  {p.relative_to(REPO_ROOT)}")
    print(f"\nTiempo total: {t_total:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
