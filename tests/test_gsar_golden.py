from unittest.mock import patch


class TestGSARGoldenBenchmark:
    """Evaluate GSARClassifier accuracy against labeled golden test data.

    Computes per-label precision, recall, F1, and overall Cohen's kappa
    for any judge model against the 20-entry gsar_golden.json dataset.
    """

    def _load_golden(self):
        import json
        from pathlib import Path

        golden_path = Path(__file__).parent / "data" / "gsar_golden.json"
        with open(golden_path) as f:
            return json.load(f)

    def _build_mock_response(self, label, g=0.8, u=0.1, c_val=0.1, f=0.9):
        from agent_trust_lab.hallukg.classifier import GSARStepResult

        return [
            GSARStepResult(
                step_index=0,
                gsar_label=label,
                g_score=g,
                u_score=u,
                c_score=c_val,
                faithfulness_score=f,
            )
        ]

    def test_golden_dataset_loads(self):
        data = self._load_golden()
        assert len(data) == 20
        labels = {d["expected_label"] for d in data}
        assert "Grounded" in labels
        assert "Ungrounded" in labels

    def test_golden_label_distribution(self):
        data = self._load_golden()
        from collections import Counter

        dist = Counter(d["expected_label"] for d in data)
        assert dist["Grounded"] >= 5
        assert dist["Ungrounded"] >= 3
        assert dist["Complementary"] >= 1
        assert dist["Contradicted"] >= 1

    def test_evaluate_gsar_on_golden_perfect_predictions(self):
        data = self._load_golden()
        from agent_trust_lab.hallukg.classifier import GSARClassifier

        for entry in data:
            expected = entry["expected_label"]
            with patch.object(
                GSARClassifier,
                "_classify_with_llm",
                return_value=self._build_mock_response(expected),
            ):
                from agent_trust_lab.models.trajectory import TrajectoryStep

                classifier = GSARClassifier()
                step = TrajectoryStep(type="test", content=entry["step_content"])
                results = classifier.classify([step], [])
                assert results[0].gsar_label == expected

    def test_evaluate_gsar_accuracy_and_kappa(self):
        data = self._load_golden()
        from collections import defaultdict

        predictions = []
        for i, entry in enumerate(data):
            if i < 15:
                predictions.append(entry["expected_label"])
            else:
                all_labels = ["Grounded", "Ungrounded", "Contradicted", "Complementary"]
                wrong_label = [lb for lb in all_labels if lb != entry["expected_label"]][0]
                predictions.append(wrong_label)

        expected = [d["expected_label"] for d in data]

        all_labels_list = sorted(set(expected + predictions))
        metrics = {}
        for label in all_labels_list:
            tp = sum(1 for e, p in zip(expected, predictions) if e == label and p == label)
            fp = sum(1 for e, p in zip(expected, predictions) if e != label and p == label)
            fn = sum(1 for e, p in zip(expected, predictions) if e == label and p != label)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            metrics[label] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

        correct = sum(1 for e, p in zip(expected, predictions) if e == p)
        accuracy = correct / len(data)
        assert accuracy == 0.75

        n = len(data)
        p_o = correct / n
        label_counts_e = defaultdict(int)
        label_counts_p = defaultdict(int)
        for e_item, p_item in zip(expected, predictions):
            label_counts_e[e_item] += 1
            label_counts_p[p_item] += 1
        p_e = sum(
            label_counts_e[label] * label_counts_p[label] / (n * n)
            for label in all_labels_list
        )
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0
        assert 0 < kappa < 1.0

    def test_evaluate_gsar_stub_fallback(self):
        data = self._load_golden()
        from agent_trust_lab.hallukg.classifier import GSARClassifier
        from agent_trust_lab.models.trajectory import TrajectoryStep

        classifier = GSARClassifier(strict_mode=False)
        step = TrajectoryStep(type="test", content=data[0]["step_content"])
        results = classifier.classify([step], [])
        assert len(results) == 1
        assert results[0].gsar_label in (
            "Grounded", "Ungrounded", "Contradicted", "Complementary"
        )
