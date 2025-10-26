from train import train_and_eval

def test_training_runs():
    metrics = train_and_eval()
    assert "recall" in metrics and metrics["recall"] > 0.5
