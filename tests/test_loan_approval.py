import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from loan_approval import (
    explain_approval_sensitivity,
    infer_approved_label,
    make_demo_data,
    predict_approval,
    train_and_evaluate,
)


class LoanApprovalTests(unittest.TestCase):
    def test_demo_data_is_deterministic_and_binary(self):
        first = make_demo_data(rows=100)
        second = make_demo_data(rows=100)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first["Loan_Status"].dropna()), {"Y", "N"})

    def test_approved_label_requires_clear_match(self):
        self.assertEqual(infer_approved_label(["N", "Y"]), "Y")
        with self.assertRaisesRegex(ValueError, "Could not identify"):
            infer_approved_label(["Declined", "Granted"])

    def test_training_saves_metrics_and_predictable_artifact(self):
        data = make_demo_data(rows=160)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "models" / "model.joblib"
            reports_dir = root / "reports"
            bundle = train_and_evaluate(
                data,
                model_path=model_path,
                reports_dir=reports_dir,
            )

            self.assertTrue(model_path.is_file())
            self.assertTrue((reports_dir / "model_metrics.csv").is_file())
            self.assertTrue((reports_dir / "evaluation.png").is_file())
            self.assertTrue((reports_dir / "feature_importance.png").is_file())
            self.assertEqual(len(bundle["test_metrics"]), 3)
            self.assertIn(bundle["selected_model"], {
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
            })

            saved = joblib.load(model_path)
            applicant = data.drop(columns=["Loan_Status", "Loan_ID"]).iloc[0].to_dict()
            prediction, probability = predict_approval(saved, applicant)
            self.assertIn(prediction, {"Y", "N"})
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)
            sensitivity = explain_approval_sensitivity(saved, applicant)
            self.assertEqual(len(sensitivity), len(saved["feature_schema"]))
            self.assertIn(
                "Score change vs typical (percentage points)", sensitivity.columns
            )

    def test_missing_target_is_reported(self):
        data = make_demo_data(rows=100).drop(columns=["Loan_Status"])
        with self.assertRaisesRegex(ValueError, "Target column"):
            train_and_evaluate(data)

    def test_training_normalizes_headers_and_target_values(self):
        data = make_demo_data(rows=100).rename(
            columns={"Loan_ID": " loan_id ", "Loan_Status": " loan_status "}
        )
        data[" loan_status "] = data[" loan_status "].map(
            lambda value: f" {value} " if isinstance(value, str) else value
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = train_and_evaluate(
                data,
                target_column="loan_status",
                model_path=Path(temp_dir) / "model.joblib",
                reports_dir=Path(temp_dir) / "reports",
            )
        self.assertNotIn("loan_id", bundle["input_columns"])
        self.assertEqual(bundle["positive_label"], "Y")


if __name__ == "__main__":
    unittest.main()
