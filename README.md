# HealthPlan Intelligence

### Understand a health plan in about 30 seconds

HealthPlan Intelligence is a small machine-learning application that estimates
whether a health plan may create high out-of-pocket cost exposure. More
importantly, it explains the result in clear, everyday language.

This is my first focused AI/ML portfolio project. I built it to show that a
model should not only produce a number—it should help a real person understand
what that number means.

## What makes it intelligent

- Learns patterns from fictional health-plan examples instead of using one
  fixed `if/else` score.
- Handles both numbers and categories in one repeatable ML pipeline.
- Returns a probability rather than an unexplained yes/no answer.
- Adds a transparent, human-friendly explanation layer.
- Includes automated tests and a GitHub Actions workflow.

## Try it locally

You need Python 3.10 or newer.

```bash
git clone https://github.com/Santhoshipadma/healthplan-intelligence.git
cd healthplan-intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows activation command:

```powershell
.venv\Scripts\activate
```

Then open the local address shown by Streamlit, usually
`http://localhost:8501`.

## How the project works

1. `generate_demo_data()` creates safe, reproducible fictional plan records.
2. A preprocessing pipeline scales numeric fields and encodes categories.
3. A random-forest classifier learns the high-cost pattern.
4. The app predicts a probability for the plan entered by the user.
5. The explanation module converts important plan differences into plain
   English.

## Project structure

```text
healthplan-intelligence/
├── app.py                         # Friendly Streamlit interface
├── healthplan_intelligence/
│   ├── data.py                    # Creates safe demo data
│   ├── model.py                   # Trains and uses the model
│   └── explain.py                 # Human-friendly explanations
├── tests/test_project.py          # Small automated test suite
├── .github/workflows/tests.yml    # Runs tests after every GitHub push
├── requirements.txt
└── README.md
```

## Run the tests

```bash
pytest -q
```

## Example portfolio explanation

> I built a small end-to-end ML product that predicts high health-plan cost
> exposure and explains the result in plain language. I used a scikit-learn
> preprocessing pipeline, random-forest classification, Streamlit, pytest, and
> GitHub Actions. I also separated data, modeling, explanation, and interface
> code so the project is easy to understand and maintain.

## Responsible use

The dataset is synthetic and the output is only an educational demonstration.
The application is not medical, insurance, legal, or financial advice. A real
decision requires official plan documents and a qualified professional.

## Next small improvements

- Allow a user to compare two plans side by side.
- Train on an appropriately licensed public healthcare-plan dataset.
- Add model-drift monitoring when new plan data arrives.
- Deploy the app and include a short demo GIF in this README.

## Technology

Python · pandas · scikit-learn · Streamlit · pytest · GitHub Actions
