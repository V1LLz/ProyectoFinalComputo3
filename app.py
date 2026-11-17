from pathlib import Path

import joblib
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
    "T_GES_AGRU_CIE": "Semanas de gestación", "MUL_PARTO": "Tipo de embarazo",
    "EDAD_MADRE": "Edad de la madre", "EST_CIVM": "Estado conyugal",
    "NIV_EDUM": "Educación de la madre", "NIV_EDUP": "Educación del padre",
    "DEPARTAMENTO": "Departamento", "AREA_RES": "Área de residencia",
    "SEG_SOCIAL": "Seguridad social", "IDPERTET": "Pertenencia étnica", "SEXO": "Sexo del bebé",
    "NUMCONSUL": "Consultas prenatales", "N_HIJOSV": "Hijos nacidos vivos",
    "N_EMB": "Número de embarazos", "EDAD_PADRE": "Edad del padre",
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
    # Categóricas: coeficiente de la categoría elegida frente al promedio de su variable
    coef = pd.Series(A["modelo"].coef_[0], index=A["columnas"])
    aportes = {}
    for col in A["categoricas"]:
        grupo = coef[[c for c in A["columnas"] if c.startswith(col + "_")]]
        aportes[f"{NOMBRES[col]}: {caso[col]}"] = coef[f"{col}_{caso[col]}"] - grupo.mean()
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


def mostrar_factores(aportes, n=6):
    top = aportes[aportes.abs().sort_values(ascending=False).index[:n]]
    tabla = pd.DataFrame({
        "Dato": top.index,
        "Aporte": top.values,
        "Aumenta el riesgo": top.clip(lower=0).values,
        "Reduce el riesgo": top.clip(upper=0).values,
    })
    st.bar_chart(tabla, x="Dato", y=["Aumenta el riesgo", "Reduce el riesgo"],
                 color=["#E24A33", "#348ABD"], horizontal=True, sort="-Aporte",
                 x_label="Aporte al puntaje", y_label="")


def main():
    st.set_page_config(page_title="Bajo peso al nacer")
    st.title("Riesgo de bajo peso al nacer")
    st.caption("Proyecto Computación 3 - Universidad de los Llanos. Datos: EEVV 2023, DANE.")

    A = cargar_modelo()
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
    mostrar_factores(factores(caso, fila, A))


if __name__ == "__main__":
    main()
