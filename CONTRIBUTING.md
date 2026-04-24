# Contributing to Folsom AQI Backend

Thank you for your interest in contributing to the Folsom AQI backend! We welcome contributions from researchers and engineers alike.

## Developer Workflow

1. **Fork the repository**
2. **Clone locally:** `git clone https://github.com/your-username/folsom-aqi-backend.git`
3. **Install dependencies:** `make install`
4. **Create a branch:** `git checkout -b feature/your-feature-name`
5. **Make your changes**
6. **Lint and format:** Run `make lint` and `make format` (powered by Ruff)
7. **Run tests:** Ensure `make test` passes successfully
8. **Commit and Push:** `git commit -m "Description of your feature"` and `git push origin feature/your-feature-name`
9. **Open a Pull Request** against the `main` branch.

## Research Contributions

If you are contributing physical models, feature engineering heuristics, or new Lagrangian trajectory approximations:
- Please ensure that **no data leakage** occurs
- Document the physical rationale behind new features in the code comments, citing relevant meteorological or atmospheric chemistry papers if applicable.
- Update `features.py` logically, maintaining the `engineer_features()` pure-function paradigm.

## Model Training

Retraining the ensemble requires significant computational resources. Ensure your local machine has at least 8 cores and 16GB RAM before running `make train`.
