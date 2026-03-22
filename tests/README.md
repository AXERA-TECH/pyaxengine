# PyAXEngine Tests

## Running Tests

### All tests
```bash
pytest
```

### Unit tests only (no hardware required)
```bash
pytest -m "not hardware"
```

### Hardware tests only
```bash
pytest -m hardware
```

### With coverage
```bash
pytest --cov=axengine --cov-report=term-missing
```

## Test Markers

- `@pytest.mark.hardware`: Tests requiring AX hardware
- `@pytest.mark.unit`: Unit tests without hardware dependency
