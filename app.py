from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

RUTA_MODELO = Path(__file__).parent / "modelo_bajo_peso.joblib"

ORDEN = {
    "T_GES_AGRU_CIE": ["Menos de 28", "28 a 36", "37 a 41", "42 o más"],
    "MUL_PARTO": ["Simple", "Doble", "Triple o más"],
    "NIV_EDUM": ["Ninguno o preescolar", "Primaria", "Secundaria", "Media",
                 "Técnica o tecnológica", "Profesional", "Posgrado"],
}
ORDEN["NIV_EDUP"] = ORDEN["NIV_EDUM"] + ["Sin información"]

NOMBRES = {
    "T_GES_AGRU_CIE": "Gestación", "MUL_PARTO": "Embarazo",
    "EDAD_MADRE": "Edad madre", "EST_CIVM": "Estado conyugal",
    "NIV_EDUM": "Educación madre", "NIV_EDUP": "Educación padre",
    "DEPARTAMENTO": "Depto.", "AREA_RES": "Área", "SEG_SOCIAL": "Seg. social",
    "IDPERTET": "Etnia", "SEXO": "Sexo bebé", "NUMCONSUL": "Consultas",
    "N_HIJOSV": "Hijos vivos", "N_EMB": "Embarazos", "EDAD_PADRE": "Edad padre",
}

MAS_COMUN = {
    "T_GES_AGRU_CIE": "37 a 41", "MUL_PARTO": "Simple", "EDAD_MADRE": "20-24",
    "EST_CIVM": "Unión libre 2 años o más", "NIV_EDUM": "Media", "NIV_EDUP": "Media",
    "DEPARTAMENTO": "Bogotá", "AREA_RES": "Cabecera municipal", "SEG_SOCIAL": "Subsidiado",
    "IDPERTET": "Ninguno", "SEXO": "Masculino",
}


@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)


def opciones(A, col):
    validas = A["opciones"][col]
    if col in ORDEN:
        return [v for v in ORDEN[col] if v in validas]
    return validas


def preparar(caso, A):
    fila = pd.get_dummies(pd.DataFrame([caso]), columns=A["categoricas"], dtype="int8")
    fila = fila.reindex(columns=A["columnas"], fill_value=0)
    fila[A["numericas"]] = A["escalador"].transform(fila[A["numericas"]])
    return fila


def predecir_riesgo(fila, A):
    puntaje = float(A["modelo"].predict_proba(fila)[0, 1])
    n = A["niveles_riesgo"]
    rango = n[(puntaje >= n["puntaje_min"]) & (puntaje <= n["puntaje_max"])].iloc[0]
    return puntaje, rango["nivel"], float(rango["pct_real_bajo_peso"])


def factores(caso, fila, A):
    # Cada dato se compara con el valor más común en los nacimientos de entrenamiento
    coef = pd.Series(A["modelo"].coef_[0], index=A["columnas"])
    aportes = {}
    for col in A["categoricas"]:
        aportes[f"{NOMBRES[col]}: {caso[col]}"] = coef[f"{col}_{caso[col]}"] - coef[f"{col}_{MAS_COMUN[col]}"]
    for col in A["numericas"]:
        aportes[f"{NOMBRES[col]}: {caso[col]}"] = coef[col] * fila[col].iloc[0]
    return pd.Series(aportes)


def formulario(A):
    with st.form("datos"):
        st.subheader("Gestación")
        c1, c2, c3 = st.columns(3)
        t_ges = c1.selectbox("Semanas de gestación", opciones(A, "T_GES_AGRU_CIE"), index=2)
        mul_parto = c2.selectbox("Tipo de embarazo", opciones(A, "MUL_PARTO"))
        sexo = c3.selectbox("Sexo del bebé", opciones(A, "SEXO"))
        c1, c2, c3 = st.columns(3)
        numconsul = c1.number_input("Consultas prenatales", 0, 25, 7)
        n_emb = c2.number_input("Número de embarazos", 1, 20, 1)
        n_hijosv = c3.number_input("Hijos nacidos vivos", 1, 17, 1)

        st.subheader("Madre")
        c1, c2, c3 = st.columns(3)
        edad_madre = c1.selectbox("Edad", opciones(A, "EDAD_MADRE"), index=3)
        est_civm = c2.selectbox("Estado conyugal", opciones(A, "EST_CIVM"))
        niv_edum = c3.selectbox("Nivel educativo", opciones(A, "NIV_EDUM"), index=3)
        c1, c2, c3 = st.columns(3)
        deptos = opciones(A, "DEPARTAMENTO")
        departamento = c1.selectbox("Departamento", deptos, index=deptos.index("Meta"))
        area_res = c2.selectbox("Área de residencia", opciones(A, "AREA_RES"))
        seg_social = c3.selectbox("Seguridad social", opciones(A, "SEG_SOCIAL"))
        idpertet = st.selectbox("Pertenencia étnica", opciones(A, "IDPERTET"), index=2)

        st.subheader("Padre")
        c1, c2 = st.columns(2)
        edad_padre = c1.number_input("Edad", 12, 82, 30, key="edad_padre")
        niv_edup = c2.selectbox("Nivel educativo", opciones(A, "NIV_EDUP"), index=3, key="niv_edup")

        if not st.form_submit_button("Calcular riesgo", type="primary"):
            return None

    return {
        "T_GES_AGRU_CIE": t_ges, "MUL_PARTO": mul_parto, "EDAD_MADRE": edad_madre,
        "EST_CIVM": est_civm, "NIV_EDUM": niv_edum, "NIV_EDUP": niv_edup,
        "DEPARTAMENTO": departamento, "AREA_RES": area_res, "SEG_SOCIAL": seg_social,
        "IDPERTET": idpertet, "SEXO": sexo, "NUMCONSUL": numconsul,
        "N_HIJOSV": n_hijosv, "N_EMB": n_emb, "EDAD_PADRE": edad_padre,
    }


def veces(aporte):
    razon = np.exp(abs(aporte))
    texto = f"{razon:.1f}".replace(".", ",")
    return f"×{texto} más" if aporte > 0 else f"×{texto} menos"


def mostrar_factores(aportes, n=6):
    top = aportes[aportes.abs().sort_values(ascending=False).index[:n]]
    tabla = pd.DataFrame({
        "Dato": [f"{d} ({veces(a)})" for d, a in top.items()],
        "Veces": np.exp(top.values),
        "Base": 1.0,
        "Efecto": ["Aumenta el riesgo" if a > 0 else "Reduce el riesgo" for a in top.values],
    })
    grafico = alt.Chart(tabla).mark_bar().encode(
        x=alt.X("Veces", title="Riesgo frente al caso más común (escala logarítmica)",
                scale=alt.Scale(type="log")),
        x2="Base",
        y=alt.Y("Dato", sort=alt.EncodingSortField("Veces", order="descending"), title=None,
                axis=alt.Axis(labelLimit=300)),
        color=alt.Color("Efecto", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["Aumenta el riesgo", "Reduce el riesgo"],
                                        range=["#E24A33", "#348ABD"])),
        tooltip=["Dato"],
    )
    st.altair_chart(grafico, width="stretch")


def pantalla_modelo(A):
    st.subheader("Resultados en el conjunto de prueba")
    st.write("Regresión logística evaluada con 100.978 nacimientos que no se usaron para entrenar.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Recall", "64,7 %", help="Casos de bajo peso que el modelo detecta")
    c2.metric("Precisión", "52,8 %", help="Alertas que sí fueron bajo peso")
    c3.metric("AUC", "0,841", help="0,5 es azar y 1 es perfecto")
    st.write("De 11.106 bebés con bajo peso, el modelo detectó 7.186. Generó 6.421 falsas alarmas "
             "entre 89.872 nacimientos de peso normal.")

    st.subheader("Bajo peso real según el puntaje")
    grupos = A["niveles_riesgo"].reset_index().rename(columns={"index": "Grupo"})
    grupos["Grupo"] += 1
    grafico = alt.Chart(grupos).mark_bar().encode(
        x=alt.X("Grupo:O", title="Grupo de puntaje (de menor a mayor)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("pct_real_bajo_peso", title="% con bajo peso"),
        color=alt.Color("nivel", title="Nivel", legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["Bajo", "Medio", "Alto"],
                                        range=["#348ABD", "#E8A33D", "#E24A33"])),
        tooltip=[alt.Tooltip("pct_real_bajo_peso", title="% bajo peso"), "nivel"],
    )
    st.altair_chart(grafico, width="stretch")
    st.caption("Cada grupo tiene unos 5.000 nacimientos. El nivel de riesgo se asigna según el "
               "porcentaje observado: bajo si está por debajo del promedio nacional (11 %), "
               "alto si es 50 % o más.")

    st.subheader("Limitaciones")
    st.markdown(
        "- Las semanas de gestación son la variable que más pesa: sin ellas el AUC baja de 0,84 a 0,68.\n"
        "- Cerca de un tercio de los casos de bajo peso no se detecta, sobre todo en bebés a término.\n"
        "- Solo usa datos del certificado de nacido vivo de 2023; no incluye datos clínicos.\n"
        "- Es una herramienta académica y no reemplaza la valoración médica."
    )


def pantalla_prediccion(A):
    caso = formulario(A)
    if caso is None:
        return

    fila = preparar(caso, A)
    puntaje, nivel, pct = predecir_riesgo(fila, A)
    mostrar = {"Bajo": st.success, "Medio": st.warning, "Alto": st.error}[nivel]
    mostrar(f"Riesgo {nivel.lower()}")
    c1, c2 = st.columns(2)
    c1.metric("Bajo peso en casos similares", f"{pct:.1f} %".replace(".", ","))
    c2.metric("Promedio nacional", "11,0 %")
    puntaje = f"{puntaje:.3f}".replace(".", ",")
    st.caption(f"De cada 100 nacimientos con un puntaje parecido, {round(pct)} nacieron con bajo peso. "
               f"Puntaje del modelo: {puntaje}")

    st.subheader("Datos que más influyeron")
    st.caption("Cada barra compara el dato con el valor más común en los nacimientos del país.")
    mostrar_factores(factores(caso, fila, A))


def main():
    st.set_page_config(page_title="Bajo peso al nacer")
    st.title("Riesgo de bajo peso al nacer")
    st.caption("Proyecto Computación 3 - Universidad de los Llanos. Datos: EEVV 2023, DANE.")

    A = cargar_modelo()
    prediccion, modelo = st.tabs(["Predicción", "Sobre el modelo"])
    with prediccion:
        pantalla_prediccion(A)
    with modelo:
        pantalla_modelo(A)


if __name__ == "__main__":
    main()
