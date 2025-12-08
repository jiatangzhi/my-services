import os
import sys
import json
import torch
import torch.nn as nn
import pytest
import logging as log
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


try:
    from src.training.model import CreditScoringModel
except ImportError:
    log.error("Ensure the structure of your proyect is correct.")
    log.error("This test script is expected to be in a folder 'tests/' and the model in 'src/training/model.py'")
    sys.exit(1)
    
    
def setup_logging(level=log.INFO, log_file: str | None = None):
    handlers = [log.StreamHandler(sys.stdout)]  # print in console
    if log_file:
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))

    log.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    # Optional: lower the noise of other libs
    for noisy in ("mlflow", "urllib3", "matplotlib"):
        log.getLogger(noisy).setLevel(log.WARNING)


setup_logging()


# 1. config
@pytest.fixture(scope="module")
def model_config_fixture():
    """Provides base configuration for the model on the tests."""
    return {
        "num_features": 25,
        "hidden_layers": [128, 64],
        "dropout_rate": 0.2,
        "use_batch_norm": True,
        "activation_fn": "ReLU"
    }


@pytest.fixture(scope="module")
def expected_info_fixture():
    """Load the expected JSON structure from a file."""
    json_path = Path(__file__).parent / "expected_model_info.json"
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        pytest.fail(f"The expected JSON file couldn't be found in: {json_path}")
    except json.JSONDecodeError:
        pytest.fail(f"Error when decoding the JSON file: {json_path}")
        
        
# 2. tests
def test_model_instantiation(model_config_fixture):
    """
    Verify that the modelo can instanciate wihtout errors with a valid configuration.
    """
    log.info("TEST: Verifying the model instantiation.")
    try:
        model = CreditScoringModel(**model_config_fixture)
        assert model is not None, "The model should not be None."
        assert isinstance(model, nn.Module), "The model should be an instance of torch.nn.Module."
        log.info("✔ Succes! The model instantiated correctly.")
    except Exception as e:
        pytest.fail(f"The instantiation of the model failed with the error: {e}")


def test_model_architecture(model_config_fixture):
    """
    Valida que la arquitectura de la red (capas, dimensiones, orden) se construya correctamente.
    """
    log.info("TEST: Verificando la arquitectura de la red neuronal.")
    model = CreditScoringModel(**model_config_fixture)
    net = model.network
    
    # La arquitectura esperada es: (Linear -> BatchNorm -> ReLU -> Dropout) -> (Linear -> BatchNorm -> ReLU -> Dropout) -> Linear
    expected_layers_count = len(model_config_fixture["hidden_layers"]) * 4 + 1
    assert len(net) == expected_layers_count, f"Se esperaban {expected_layers_count} capas, pero se encontraron {len(net)}."
    
    # 1° hidden layer
    assert isinstance(net[0], nn.Linear) and net[0].in_features == 25 and net[0].out_features == 128
    assert isinstance(net[1], nn.BatchNorm1d) and net[1].num_features == 128
    assert isinstance(net[2], nn.ReLU)
    assert isinstance(net[3], nn.Dropout) and net[3].p == 0.2
    
    # 2° hidden layer
    assert isinstance(net[4], nn.Linear) and net[4].in_features == 128 and net[4].out_features == 64
    assert isinstance(net[5], nn.BatchNorm1d) and net[5].num_features == 64
    assert isinstance(net[6], nn.ReLU)
    assert isinstance(net[7], nn.Dropout) and net[7].p == 0.2
    
    # 3° output layer
    assert isinstance(net[8], nn.Linear) and net[8].in_features == 64 and net[8].out_features == 1
    
    log.info("✔ Success! The architecture and  dimensions of the layers are correct.")


def test_forward_pass(model_config_fixture):
    """
    Execute a "smoke test" to ensure that the forward pass works and return a tensor with the correct shape.
    """
    log.info("TEST: Verifying the forward pass.")
    model = CreditScoringModel(**model_config_fixture)
    model.eval()
    
    batch_size = 10
    input_tensor = torch.randn(batch_size, model_config_fixture["num_features"])
    
    with torch.no_grad():
        output = model(input_tensor)
    
    expected_shape = (batch_size, 1)
    assert output.shape == expected_shape, f"The tensor shape of the output is incorrect. Expected: {expected_shape}, Obtenido: {output.shape}"
    log.info(f"✔ Success! Forward pass completed and the output shape is correct: {output.shape}.")
