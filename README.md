# NYC Airbnb Room Type Predictor

A Streamlit app that predicts whether an NYC Airbnb listing is an entire home/apartment, private room, or shared room. The trained scikit-learn pipeline is stored in `Model_Pipeline.pkl`.

## Run locally

```text
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Streamlit opens the app at `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this repository, including `streamlit_app.py`, `requirements.txt`, and `Model_Pipeline.pkl`, to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io/) and choose **New app**.
3. Select the repository and branch, set the main file to `streamlit_app.py`, and deploy.
4. Keep the Python version aligned with `runtime.txt` if Streamlit Cloud asks for a version setting.

The app loads the model relative to its own file, so it does not depend on Streamlit Cloud's working directory and does not require the old Render API.

## Project notes

- `streamlit_app.py` is the deployed frontend and prediction entrypoint.
- The notebook contains the training and evaluation workflow.
