from __future__ import annotations

import re
from io import BytesIO
from urllib.error import URLError

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

st.set_page_config(page_title="Laptop Analytics Dashboard", page_icon="💻", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/drpatrickcerna-prog/laptop/main/laptop.csv"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: #f5f7fb; padding: 14px 16px; border-radius: 12px;}
[data-testid="stSidebar"] {background: #f7f8fc;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_remote_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)

@st.cache_data(show_spinner=False)
def load_uploaded_data(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(raw))

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip()
        df[c] = df[c].replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    if "Price" in df:
        df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    if "RAM" in df:
        df["RAM_GB"] = pd.to_numeric(df["RAM"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    if "Display" in df:
        df["Display_Inches"] = pd.to_numeric(df["Display"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    if "SSD" in df:
        df["SSD_GB"] = pd.to_numeric(df["SSD"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    if "HDD" in df:
        df["HDD_GB"] = pd.to_numeric(df["HDD"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    if "Ghz" in df:
        df["CPU_GHz"] = pd.to_numeric(df["Ghz"].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    return df

def money(value):
    return f"₱{value:,.0f}" if pd.notna(value) else "N/A"

st.title("💻 Laptop Market — Descriptive Analytics")
st.caption("Explore laptop prices, brands, processors, memory, storage, graphics, and display characteristics.")

with st.sidebar:
    st.header("Data source")
    uploaded = st.file_uploader("Upload a replacement CSV (optional)", type=["csv"])
    st.caption("By default, the app loads laptop.csv directly from the linked GitHub repository.")

try:
    source_df = load_uploaded_data(uploaded.getvalue()) if uploaded else load_remote_data()
    source_label = "Uploaded CSV" if uploaded else "GitHub repository"
except Exception as exc:
    st.error("The GitHub dataset could not be loaded. Check your internet connection or upload laptop.csv below.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

raw_rows, raw_cols = source_df.shape
df = clean_data(source_df)

with st.sidebar:
    st.divider()
    st.header("Interactive filters")
    filtered = df.copy()
    if "Brand" in filtered:
        brands = sorted(filtered["Brand"].dropna().unique().tolist())
        chosen_brands = st.multiselect("Brand", brands, default=brands)
        filtered = filtered[filtered["Brand"].isin(chosen_brands)]
    if "Processor_Brand" in filtered:
        processors = sorted(filtered["Processor_Brand"].dropna().unique().tolist())
        chosen_processors = st.multiselect("Processor brand", processors, default=processors)
        filtered = filtered[filtered["Processor_Brand"].isin(chosen_processors)]
    if "GPU_Brand" in filtered:
        gpus = sorted(filtered["GPU_Brand"].dropna().unique().tolist())
        chosen_gpus = st.multiselect("GPU brand", gpus, default=gpus)
        filtered = filtered[filtered["GPU_Brand"].isin(chosen_gpus)]
    if "Price" in filtered and filtered["Price"].notna().any():
        lo, hi = int(filtered["Price"].min()), int(filtered["Price"].max())
        price_range = st.slider("Price range (₱)", lo, hi, (lo, hi), step=max(1, (hi-lo)//100 or 1))
        filtered = filtered[filtered["Price"].between(*price_range)]
    if "RAM_GB" in filtered and filtered["RAM_GB"].notna().any():
        ram_values = sorted(filtered["RAM_GB"].dropna().unique().tolist())
        chosen_ram = st.multiselect("RAM (GB)", ram_values, default=ram_values)
        filtered = filtered[filtered["RAM_GB"].isin(chosen_ram)]
    st.divider()
    st.caption(f"Source: {source_label}")
    st.caption(f"Showing {len(filtered):,} of {len(df):,} records")

if filtered.empty:
    st.warning("No records match the selected filters. Adjust the filters in the sidebar.")
    st.stop()

# KPI summary
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Laptop records", f"{len(filtered):,}")
if "Brand" in filtered: k2.metric("Brands", f"{filtered['Brand'].nunique():,}")
if "Price" in filtered:
    k3.metric("Average price", money(filtered["Price"].mean()))
    k4.metric("Median price", money(filtered["Price"].median()))
    k5.metric("Price range", money(filtered["Price"].max() - filtered["Price"].min()))

overview, pricing, specifications, scatterplots, predictive, quality, records = st.tabs(["📊 Overview", "💰 Price analysis", "⚙️ Specifications", "🔎 Scatterplots", "🤖 Classification models", "🧹 Data quality", "📋 Dataset"])

with overview:
    left, right = st.columns(2)
    with left:
        if "Brand" in filtered:
            counts = filtered["Brand"].value_counts().rename_axis("Brand").reset_index(name="Laptop count")
            fig = px.bar(counts, x="Brand", y="Laptop count", title="Laptop listings by brand", color="Laptop count", text_auto=True)
            fig.update_layout(xaxis_tickangle=-35, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if "Processor_Brand" in filtered:
            counts = filtered["Processor_Brand"].value_counts().rename_axis("Processor brand").reset_index(name="Laptop count")
            fig = px.pie(counts, names="Processor brand", values="Laptop count", hole=.48, title="Processor brand share")
            st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        if "Price" in filtered:
            fig = px.histogram(filtered.dropna(subset=["Price"]), x="Price", nbins=30, title="Laptop price distribution", labels={"Price":"Price (₱)"})
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if "RAM_GB" in filtered:
            ram = filtered["RAM_GB"].value_counts().sort_index().rename_axis("RAM (GB)").reset_index(name="Laptop count")
            fig = px.bar(ram, x="RAM (GB)", y="Laptop count", title="RAM capacity distribution", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

with pricing:
    if "Price" not in filtered:
        st.info("No Price field found in this dataset.")
    else:
        numeric_price = filtered["Price"].dropna()
        stats = numeric_price.describe().to_frame("Price (₱)")
        stats.loc["median"] = numeric_price.median()
        stats.loc["mode"] = numeric_price.mode().iloc[0] if not numeric_price.mode().empty else float("nan")
        stats.loc["variance"] = numeric_price.var()
        stats.loc["skewness"] = numeric_price.skew()
        stats.loc["kurtosis"] = numeric_price.kurt()
        st.subheader("Descriptive statistics — price")
        st.dataframe(stats.style.format("₱{:,.2f}"), use_container_width=True)
        left, right = st.columns(2)
        with left:
            if "Brand" in filtered:
                brand_price = filtered.groupby("Brand", dropna=True)["Price"].agg(["mean", "median", "count"]).reset_index().sort_values("mean", ascending=False)
                fig = px.bar(brand_price, x="Brand", y="mean", hover_data=["median", "count"], title="Average laptop price by brand", labels={"mean":"Average price (₱)"}, text_auto=".2s")
                fig.update_layout(xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)
        with right:
            if "Processor_Brand" in filtered:
                fig = px.box(filtered.dropna(subset=["Price"]), x="Processor_Brand", y="Price", color="Processor_Brand", title="Price distribution by processor brand", labels={"Price":"Price (₱)"})
                st.plotly_chart(fig, use_container_width=True)
        if "Brand" in filtered:
            st.subheader("Brand price summary")
            st.dataframe(filtered.groupby("Brand")["Price"].agg(Count="count", Mean="mean", Median="median", Minimum="min", Maximum="max").sort_values("Mean", ascending=False).style.format({"Mean":"₱{:,.0f}","Median":"₱{:,.0f}","Minimum":"₱{:,.0f}","Maximum":"₱{:,.0f}"}), use_container_width=True)

with specifications:
    left, right = st.columns(2)
    with left:
        if "Processor_Name" in filtered:
            proc = filtered["Processor_Name"].value_counts().head(12).rename_axis("Processor").reset_index(name="Count")
            fig = px.bar(proc.sort_values("Count"), x="Count", y="Processor", orientation="h", title="Most common processor models")
            st.plotly_chart(fig, use_container_width=True)
        if "GPU_Brand" in filtered:
            gpu = filtered["GPU_Brand"].value_counts().rename_axis("GPU brand").reset_index(name="Count")
            fig = px.pie(gpu, names="GPU brand", values="Count", hole=.4, title="GPU brand distribution")
            st.plotly_chart(fig, use_container_width=True)
    with right:
        if "SSD_GB" in filtered:
            ssd = filtered["SSD_GB"].value_counts().sort_index().rename_axis("SSD capacity (GB)").reset_index(name="Count")
            fig = px.bar(ssd, x="SSD capacity (GB)", y="Count", title="SSD capacity distribution")
            st.plotly_chart(fig, use_container_width=True)
        if "Display_Inches" in filtered:
            display = filtered["Display_Inches"].value_counts().sort_index().rename_axis("Display (inches)").reset_index(name="Count")
            fig = px.bar(display, x="Display (inches)", y="Count", title="Display size distribution")
            st.plotly_chart(fig, use_container_width=True)
    if all(c in filtered for c in ["RAM_GB", "Price"]):
        st.subheader("Relationship between RAM and price")
        scatter = filtered.dropna(subset=["RAM_GB", "Price"])
        fig = px.scatter(scatter, x="RAM_GB", y="Price", color="Brand" if "Brand" in scatter else None, hover_name="Name" if "Name" in scatter else None, title="RAM capacity vs. laptop price", labels={"RAM_GB":"RAM (GB)","Price":"Price (₱)"})
        st.plotly_chart(fig, use_container_width=True)
    numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) >= 2:
        st.subheader("Numeric feature correlation")
        corr = filtered[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation matrix")
        st.plotly_chart(fig, use_container_width=True)


with scatterplots:
    st.subheader("Interactive scatterplot explorer")
    st.caption("Explore relationships between numeric laptop characteristics. Hover over points for record details.")
    numeric_options = filtered.select_dtypes(include="number").columns.tolist()
    if len(numeric_options) >= 2:
        c1, c2, c3 = st.columns(3)
        default_x = "RAM_GB" if "RAM_GB" in numeric_options else numeric_options[0]
        default_y = "Price" if "Price" in numeric_options else numeric_options[1]
        x_feature = c1.selectbox("X-axis", numeric_options, index=numeric_options.index(default_x), key="scatter_x")
        y_feature = c2.selectbox("Y-axis", numeric_options, index=numeric_options.index(default_y), key="scatter_y")
        color_options = ["None"] + filtered.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        color_feature = c3.selectbox("Color points by", color_options, index=1 if "Brand" in color_options else 0, key="scatter_color")
        scatter_data = filtered.dropna(subset=[x_feature, y_feature]).copy()
        if len(scatter_data):
            fig = px.scatter(
                scatter_data, x=x_feature, y=y_feature,
                color=None if color_feature == "None" else color_feature,
                hover_name="Name" if "Name" in scatter_data.columns else None,
                hover_data=[c for c in ["Brand", "Processor_Name", "RAM", "GPU", "Price"] if c in scatter_data.columns],
                title=f"{y_feature} vs. {x_feature}",
                opacity=0.72
            )
            fig.update_traces(marker={"size": 9})

            # Add an OLS-style trendline with scikit-learn instead of Plotly's
            # trendline="ols", which requires the optional statsmodels package.
            trend_data = scatter_data[[x_feature, y_feature]].dropna()
            if len(trend_data) >= 3 and trend_data[x_feature].nunique() >= 2:
                X_trend = trend_data[[x_feature]].astype(float)
                y_trend = trend_data[y_feature].astype(float)
                trend_model = LinearRegression()
                trend_model.fit(X_trend, y_trend)
                x_line = pd.DataFrame({
                    x_feature: [float(trend_data[x_feature].min()),
                                float(trend_data[x_feature].max())]
                })
                y_line = trend_model.predict(x_line)
                fig.add_trace(go.Scatter(
                    x=x_line[x_feature],
                    y=y_line,
                    mode="lines",
                    name="Linear trendline",
                    line={"color": "#E45756", "width": 3, "dash": "dash"}
                ))

            st.plotly_chart(fig, use_container_width=True)
            corr_value = scatter_data[[x_feature, y_feature]].corr().iloc[0, 1]
            st.metric("Pearson correlation", f"{corr_value:.3f}" if pd.notna(corr_value) else "N/A")
            st.caption("The trendline is calculated with scikit-learn; statsmodels is not required. Correlation describes linear association and does not establish causation.")
        else:
            st.info("No complete observations are available for the selected axes.")
    else:
        st.info("At least two numeric columns are needed to create a scatterplot.")

with predictive:
    st.subheader("Laptop classification — four machine-learning algorithms")
    st.write(
        "This module classifies laptop records into a selected category. "
        "By default, it derives three price segments (Budget, Mid-range, Premium) "
        "from the Price column using quantile-based bins. You may also select another "
        "categorical field as the prediction target."
    )

    from ml_classification import (
        make_price_segment_target,
        train_classification_models,
        make_classification_summary,
    )

    classification_df = df.copy()
    target_options = ["Price Segment (derived)"]
    categorical_candidates = []
    for col in classification_df.select_dtypes(include=["object", "string", "category"]).columns:
        unique_count = classification_df[col].nunique(dropna=True)
        if 2 <= unique_count <= 15:
            categorical_candidates.append(col)
    target_options += sorted(categorical_candidates)

    if "Price" not in classification_df.columns:
        target_options = [x for x in target_options if x != "Price Segment (derived)"]

    if not target_options:
        st.warning(
            "No suitable classification target was found. A target must contain "
            "at least two classes and should not have excessive unique values."
        )
    else:
        selected_target = st.selectbox(
            "Classification target",
            target_options,
            index=0,
            help=(
                "Price Segment is derived from Price using quantile bins. "
                "Other choices use an existing categorical column."
            ),
        )

        target_frame = classification_df.copy()
        if selected_target == "Price Segment (derived)":
            target_frame, target_col, target_note = make_price_segment_target(
                target_frame, price_column="Price"
            )
            if target_col is None:
                st.error(target_note)
                st.stop()
            st.caption(target_note)
            excluded_features = {"Price", "Name", target_col}
        else:
            target_col = selected_target
            target_frame[target_col] = target_frame[target_col].astype("string").str.strip()
            target_frame[target_col] = target_frame[target_col].replace(
                {"": pd.NA, "nan": pd.NA, "None": pd.NA}
            )
            excluded_features = {target_col, "Name"}

        target_frame = target_frame.dropna(subset=[target_col]).copy()
        target_frame[target_col] = target_frame[target_col].astype(str)

        class_counts = target_frame[target_col].value_counts()
        st.markdown("#### Target class distribution")
        class_distribution = (
            class_counts.rename_axis("Class")
            .reset_index(name="Records")
        )
        left, right = st.columns([1, 1])
        with left:
            st.dataframe(class_distribution, use_container_width=True, hide_index=True)
        with right:
            fig = px.bar(
                class_distribution,
                x="Class",
                y="Records",
                color="Class",
                title="Records per target class",
                text_auto=True,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Candidate predictors: remove target, price leakage for derived price segments,
        # constant columns, empty columns, and near-unique identifiers.
        candidate_features = []
        for col in target_frame.columns:
            if col in excluded_features:
                continue
            non_missing = target_frame[col].notna().sum()
            unique_count = target_frame[col].nunique(dropna=True)
            if non_missing == 0 or unique_count <= 1:
                continue
            if unique_count > max(50, int(len(target_frame) * 0.50)):
                continue
            candidate_features.append(col)

        if len(target_frame) < 20:
            st.warning("At least 20 labeled records are recommended before training classifiers.")
        elif not candidate_features:
            st.warning("No usable predictor variables were found.")
        else:
            with st.expander("Classification configuration", expanded=True):
                selected_features = st.multiselect(
                    "Predictor variables",
                    candidate_features,
                    default=candidate_features,
                    help=(
                        "For Price Segment classification, Price is excluded to prevent "
                        "target leakage. Name is excluded as a high-cardinality identifier."
                    ),
                )
                test_fraction = st.slider(
                    "Test data proportion", min_value=0.10, max_value=0.40,
                    value=0.20, step=0.05, key="classification_test_fraction"
                )
                random_seed = st.number_input(
                    "Random seed", min_value=1, max_value=9999,
                    value=42, step=1, key="classification_random_seed"
                )

            if not selected_features:
                st.info("Select at least one predictor variable.")
            elif class_counts.min() < 2:
                st.warning(
                    "Each target class must have at least two records so the application "
                    "can place examples in both training and test partitions. Merge rare "
                    "classes or collect more records."
                )
            else:
                st.caption(
                    f"Training dataset: {len(target_frame):,} records | "
                    f"Predictors: {len(selected_features)} | "
                    f"Classes: {target_frame[target_col].nunique()}"
                )
                if st.button(
                    "🚀 Train and evaluate all 4 classifiers",
                    type="primary",
                    key="run_classification",
                ):
                    with st.spinner("Training Gradient Boosting, Random Forest, SVM, and Naive Bayes..."):
                        try:
                            model_results, model_outputs = train_classification_models(
                                data=target_frame,
                                target=target_col,
                                features=selected_features,
                                test_fraction=float(test_fraction),
                                random_seed=int(random_seed),
                            )
                            st.session_state["laptop_classification_results"] = model_results
                            st.session_state["laptop_classification_outputs"] = model_outputs
                            st.session_state["laptop_classification_target"] = target_col
                            st.session_state["laptop_classification_features"] = selected_features
                            st.session_state["laptop_classification_seed"] = int(random_seed)
                            st.success("All four classifiers were trained and evaluated.")
                        except Exception as exc:
                            st.error(f"Classification could not be completed: {exc}")

                if "laptop_classification_results" in st.session_state:
                    results = st.session_state["laptop_classification_results"]
                    outputs = st.session_state["laptop_classification_outputs"]

                    # Avoid displaying stale results if the user changes target or predictors.
                    same_configuration = (
                        st.session_state.get("laptop_classification_target") == target_col
                        and st.session_state.get("laptop_classification_features") == selected_features
                        and st.session_state.get("laptop_classification_seed") == int(random_seed)
                    )
                    if same_configuration:
                        st.markdown("#### Model performance comparison")
                        best_row = results.sort_values(
                            ["Weighted F1", "Accuracy"], ascending=False
                        ).iloc[0]
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Best model (Weighted F1)", best_row["Model"])
                        m2.metric("Accuracy", f"{best_row['Accuracy']:.3f}")
                        m3.metric("Weighted F1", f"{best_row['Weighted F1']:.3f}")

                        metric_cols = [
                            "Accuracy", "Precision (weighted)",
                            "Recall (weighted)", "Weighted F1", "Macro F1"
                        ]
                        st.dataframe(
                            results.style.format({c: "{:.3f}" for c in metric_cols}),
                            use_container_width=True,
                            hide_index=True,
                        )

                        metric_long = results.melt(
                            id_vars="Model",
                            value_vars=["Accuracy", "Weighted F1", "Macro F1"],
                            var_name="Metric",
                            value_name="Score",
                        )
                        fig = px.bar(
                            metric_long, x="Model", y="Score",
                            color="Metric", barmode="group",
                            title="Classification performance by algorithm",
                            range_y=[0, 1],
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        selected_model = st.selectbox(
                            "Inspect model details",
                            results["Model"].tolist(),
                            key="inspect_classification_model",
                        )
                        selected_output = outputs[selected_model]
                        labels = selected_output["labels"]
                        cm = selected_output["confusion_matrix"]
                        cm_fig = px.imshow(
                            cm,
                            x=labels,
                            y=labels,
                            text_auto=True,
                            aspect="auto",
                            color_continuous_scale="Blues",
                            labels={"x": "Predicted class", "y": "Actual class", "color": "Records"},
                            title=f"Confusion matrix — {selected_model}",
                        )
                        st.plotly_chart(cm_fig, use_container_width=True)

                        report_df = pd.DataFrame(
                            selected_output["classification_report"]
                        ).transpose().reset_index().rename(columns={"index": "Class / metric"})
                        st.markdown("#### Per-class classification report")
                        st.dataframe(report_df, use_container_width=True, hide_index=True)

                        prediction_df = selected_output["prediction_details"]
                        st.markdown("#### Test-set predictions")
                        st.dataframe(prediction_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            "⬇️ Download test-set predictions (CSV)",
                            prediction_df.to_csv(index=False).encode("utf-8"),
                            "laptop_classification_predictions.csv",
                            "text/csv",
                        )
                        st.download_button(
                            "⬇️ Download classifier comparison (CSV)",
                            results.to_csv(index=False).encode("utf-8"),
                            "laptop_classification_metrics.csv",
                            "text/csv",
                        )
                    else:
                        st.info(
                            "The selected target or predictors have changed. "
                            "Train the classifiers again to refresh the results."
                        )

                st.info(
                    "Evaluation uses a stratified hold-out test set when feasible. "
                    "Accuracy and F1 scores are between 0 and 1; higher is better. "
                    "Macro F1 gives each class equal weight, while weighted F1 accounts "
                    "for class frequency. Results depend on the selected target, features, "
                    "data quality, and random split."
                )

with quality:
    st.subheader("Dataset profile")
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows in source", f"{raw_rows:,}")
    q2.metric("Original columns", f"{raw_cols:,}")
    q3.metric("Missing cells", f"{int(source_df.isna().sum().sum()):,}")
    missing = pd.DataFrame({"Column": source_df.columns, "Missing values": source_df.isna().sum().values, "Missing (%)": (source_df.isna().mean().values*100).round(2), "Data type": source_df.dtypes.astype(str).values}).sort_values("Missing (%)", ascending=False)
    st.dataframe(missing, use_container_width=True, hide_index=True)
    st.subheader("Duplicate records")
    st.write(f"Exact duplicate rows: **{int(source_df.duplicated().sum()):,}**")
    st.caption("Missing values and duplicates are reported, not silently removed. Derived numeric fields are parsed for analysis; original columns are retained.")

with records:
    st.subheader("Filtered laptop records")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data (CSV)", data=csv, file_name="filtered_laptop_dataset.csv", mime="text/csv")
    summary = filtered.describe(include="all").transpose().reset_index().rename(columns={"index":"Feature"})
    st.download_button("⬇️ Download descriptive summary (CSV)", data=summary.to_csv(index=False).encode("utf-8"), file_name="laptop_descriptive_summary.csv", mime="text/csv")

st.divider()
st.caption("Analytics platform: descriptive exploration, interactive scatterplots, and four-algorithm laptop classification.")
