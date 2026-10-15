"""Streamlit interface for a locally trained loan-approval model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from loan_approval import (
    DEFAULT_MODEL_PATH,
    explain_approval_sensitivity,
    predict_approval,
)


st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")
st.title("Loan Approval Prediction")
st.caption("An educational machine-learning demonstration based on your training data.")
st.warning(
    "This model output is not a credit decision and must not be used to approve or reject "
    "real applicants. Review fairness, legal, and compliance requirements before any "
    "real-world use."
)

model_path = Path(os.environ.get("LOAN_MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
if not model_path.is_file():
    st.error(
        f"Model artifact not found at `{model_path}`. Train a model first; see the "
        "README for the command."
    )
    st.stop()


@st.cache_resource
def load_model(path: str, modified_time: float) -> dict[str, Any]:
    del modified_time
    return joblib.load(path)


bundle = load_model(str(model_path.resolve()), model_path.stat().st_mtime)
st.sidebar.subheader("Model details")
st.sidebar.write(f"**Selected model:** {bundle['selected_model']}")
st.sidebar.write(f"**Training target:** `{bundle['target_column']}`")
metric_rows = bundle["test_metrics"]
best_metrics = next(
    row for row in metric_rows if row["Model"] == bundle["selected_model"]
)
st.sidebar.write(f"**Holdout accuracy:** {best_metrics['Accuracy']:.1%}")
st.sidebar.caption("Evaluation is based only on the dataset used to train this local model.")

st.subheader("Applicant information")
st.write("Enter values for the predictor fields used by the trained model.")
with st.form("applicant_form"):
    columns = st.columns(2)
    applicant: dict[str, Any] = {}
    for index, field in enumerate(bundle["feature_schema"]):
        name = field["name"]
        label = name.replace("_", " ")
        with columns[index % 2]:
            if field["kind"] == "numeric":
                numeric_options: dict[str, Any] = {
                    "value": float(field["default"]),
                    "step": field["step"],
                    "key": f"field_{index}",
                }
                if name in {"ApplicantIncome", "CoapplicantIncome", "LoanAmount"}:
                    numeric_options["min_value"] = 0.0
                applicant[name] = st.number_input(label, **numeric_options)
            elif field["choices"] and len(field["choices"]) <= 40:
                choices = field["choices"]
                selected = field["default"] if field["default"] in choices else choices[0]
                applicant[name] = st.selectbox(
                    label,
                    choices,
                    index=choices.index(selected),
                    key=f"field_{index}",
                )
            else:
                applicant[name] = st.text_input(
                    label,
                    value=field["default"],
                    key=f"field_{index}",
                )
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    prediction, probability = predict_approval(bundle, applicant)
    display_prediction = str(prediction)
    result = (
        f"Model output: **{display_prediction}** "
        f"(model-estimated approval score: {probability:.1%}; "
        "not a verified or calibrated probability)"
    )
    if prediction == bundle["positive_label"]:
        st.success(result)
    else:
        st.info(result)
    st.caption(
        "This educational model is not validated for real credit decisions. "
        "The score is not a calibrated probability."
    )
    with st.expander("What influenced this result?"):
        st.write(
            "Each row changes one entered field to that field's median (numeric) "
            "or most common value (category) in the training data, then compares "
            "the model score. A positive change means the entered value raised "
            "the score relative to that typical value."
        )
        st.caption(
            "This is a one-field-at-a-time sensitivity check, not a causal explanation. "
            "It does not account for interactions or related fields changing together."
        )
        st.dataframe(
            explain_approval_sensitivity(bundle, applicant),
            hide_index=True,
            width="stretch",
        )
