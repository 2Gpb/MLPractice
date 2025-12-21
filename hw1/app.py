import pickle
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import seaborn as sns


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "linear_model_l2.pkl"
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        print("Model not found!")

    model = Ridge()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model

@st.cache_data
def load_file(file):
    return pd.read_csv(file)

st.set_page_config("Linear regression", page_icon="🚗")

def get_options(df, col):
    return sorted(df[col].dropna().astype(str).unique().tolist())


try:
    MODEL = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

try:
    TRAIN = load_file(TRAIN_PATH)
    TEST = load_file(TEST_PATH)
except Exception as e:
    st.error(f"Ошибка загрузки данных для визуализации EDA: {e}")
    st.stop()


st.title("Предсказание стоимости автомобиля")

tab_train, tab_test, tab_model, tab_pred = st.tabs(
    ["EDA: train", "EDA: test", "Model", "Predict"]
)

with tab_train:
    st.markdown("# EDA на трейне")

    st.markdown("### Размерность и пропуски")
    coll, col2, col3 = st.columns(3)
    with coll:
        st.metric("Количество столбцов", len(TRAIN.columns))
    with col2:
        st.metric("Количество строк", len(TRAIN))
    with col3:
        st.metric("Количество пропущенных значений", TRAIN.isnull().sum().sum())

    st.markdown("### Первые 30 строк датасета")
    st.dataframe(TRAIN.head(30))

    st.markdown("### Распределение целевой переменной")
    fig, ax = plt.subplots()
    ax.hist(TRAIN["selling_price"].dropna(), bins=30)
    ax.set_title("selling_price distribution")
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Основные статистики по числовым столбцам")
    st.write(TRAIN.describe(include="number"))

    st.markdown("### Основные статистики по категориальным столбцам")
    st.write(TRAIN.describe(include="object"))

    st.markdown("### Попарные распределения числовых признаков")
    if st.checkbox("Показать pairplot (может быть долго)", value=False, key="train"):
        st.pyplot(sns.pairplot(TRAIN).figure)

    st.markdown("### Корреляция Пирсона")
    fig, ax = plt.subplots()
    sns.heatmap(TRAIN.corr(numeric_only=True, method="pearson"), cmap="Blues", annot=True, ax=ax)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Корреляция Спирмена")
    fig, ax = plt.subplots()
    sns.heatmap(TRAIN.corr(numeric_only=True, method="spearman"), cmap="Blues", annot=True, ax=ax)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Корреляция Кендалла")
    fig, ax = plt.subplots()
    sns.heatmap(TRAIN.corr(numeric_only=True, method="kendall"), cmap="Blues", annot=True, ax=ax)
    st.pyplot(fig, clear_figure=True)

with tab_test:
    st.header("EDA на тесте")

    st.markdown("### Размерность и пропуски")
    coll, col2, col3 = st.columns(3)
    with coll:
        st.metric("Количество столбцов", len(TEST.columns))
    with col2:
        st.metric("Количество строк", len(TEST))
    with col3:
        st.metric("Количество пропущенных значений", TEST.isnull().sum().sum())

    st.markdown("### Первые 30 строк датасета")
    st.dataframe(TEST.head(30))

    st.markdown("### Основные статистики по числовым столбцам")
    st.write(TEST.describe(include="number"))

    st.markdown("### Основные статистики по категориальным столбцам")
    st.write(TEST.describe(include="object"))

    st.markdown("### Попарные распределения числовых признаков")
    if st.checkbox("Показать pairplot (может быть долго)", value=False, key="test"):
        st.pyplot(sns.pairplot(TEST).figure)

    st.markdown("### Корреляция Пирсона")
    fig, ax = plt.subplots()
    sns.heatmap(TEST.corr(numeric_only=True, method="pearson"), cmap="Blues", annot=True, ax=ax)
    st.pyplot(fig, clear_figure=True)

with tab_model:
    st.header("Визуализация весов модели")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(MODEL.named_steps["model"].coef_, bins=50)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Распределение коэффициентов модели")
    ax.set_xlabel("Значение веса")
    ax.set_ylabel("Кол-во весов в одном столбце")

    st.pyplot(fig, clear_figure=True)

with tab_pred:
    st.header("Предсказание стоимости по вашим данным")
    choice = st.radio("Выберите вариант предоставления данных", ["Загрузить файл", "Ввести руками"], key="pred_input_mode")
    if choice == "Загрузить файл":
        uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

        if uploaded_file is None:
            st.info("Загрузите CSV файл для начала работы")
            st.stop()

        df = pd.read_csv(uploaded_file)
        pred = MODEL.predict(df)

        prices = [f"{p:,.0f}".replace(",", " ") for p in pred]
        st.success("Предсказанные цены:")
        st.write(prices)
    elif choice == "Ввести руками":
        st.write("Введите признаки")

        cat_options = {c: get_options(TRAIN, c) for c in ["fuel", "seller_type", "transmission", "owner"] if c in TRAIN.columns}

        with st.form("manual_input"):
            c1, c2 = st.columns(2)

            with c1:
                st.subheader("Числовые")
                year = st.number_input("Year", min_value=1950, max_value=2030, value=2015)
                km_driven = st.number_input("Km_driven", min_value=0, value=50000)
                mileage = st.number_input("Mileage", min_value=0.0, value=18.0)
                engine = st.number_input("Engine", min_value=0.0, value=1200.0)
                max_power = st.number_input("Max_power", min_value=0.0, value=80.0)
                seats = st.number_input("Seats", min_value=0, value=14)

            with c2:
                st.subheader("Категориальные")
                fuel = st.selectbox("Fuel", cat_options.get("fuel", ["Petrol", "Diesel"]))
                seller_type = st.selectbox("Seller_type", cat_options.get("seller_type", ["Individual", "Dealer"]))
                transmission = st.selectbox("Transmission", cat_options.get("transmission", ["Manual", "Automatic"]))
                owner = st.selectbox("Owner", cat_options.get("owner", ["First Owner", "Second Owner"]))

            submitted = st.form_submit_button("Предсказать")

        if submitted:
            x = pd.DataFrame([{
                "year": year,
                "km_driven": km_driven,
                "mileage": mileage,
                "engine": engine,
                "max_power": max_power,
                "seats": seats,
                "fuel": fuel,
                "seller_type": seller_type,
                "transmission": transmission,
                "owner": owner,
            }]).reset_index(drop=True)

            st.dataframe(x, use_container_width=True)

            pred = MODEL.predict(x)[0]
            st.success(f"Предсказанная цена: {pred:,.0f}".replace(",", " "))