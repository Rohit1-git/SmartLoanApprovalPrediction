import pandas as pd
from django.contrib import messages
from django.shortcuts import redirect, render

from .ml_service import DATA_PATH, get_charts, get_metadata, predict_one, read_dataset, train_dataset


def home(request):
    return render(request, "dashboard/home.html", {"meta": get_metadata()})


def upload_dataset(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("dataset")
        if not uploaded_file:
            messages.error(request, "Please choose a CSV or XLSX dataset first.")
            return redirect("upload")
        try:
            frame = read_dataset(uploaded_file)
            metadata, _, _ = train_dataset(frame)
            messages.success(
                request,
                f"Dataset processed successfully. Model trained with {metadata['rows']} records.",
            )
            return redirect("analysis")
        except ValueError as exc:
            messages.error(request, str(exc))
        except Exception as exc:
            messages.error(request, f"Something went wrong while processing the dataset: {exc}")
    return render(request, "dashboard/upload.html", {"meta": get_metadata()})


def analysis(request):
    metadata = get_metadata()
    preview = []
    columns = []
    if metadata and DATA_PATH.exists():
        try:
            frame = pd.read_csv(DATA_PATH).head(10).fillna("")
            columns = frame.columns.tolist()
            preview = frame.astype(str).to_dict(orient="records")
        except Exception:
            messages.error(request, "The saved dataset preview could not be loaded. Please upload the dataset again.")
    return render(request, "dashboard/analysis.html", {
        "meta": metadata,
        "preview": preview,
        "columns": columns,
    })


def analytics(request):
    return render(request, "dashboard/analytics.html", {
        "meta": get_metadata(),
        "charts": get_charts(),
    })


def predict(request):
    metadata = get_metadata()
    result = None
    if request.method == "POST":
        try:
            result = predict_one(request.POST)
        except ValueError as exc:
            messages.error(request, str(exc))
        except Exception as exc:
            messages.error(request, f"Prediction could not be completed: {exc}")

    fields = []
    if metadata:
        for column in metadata["features"]:
            is_numeric = column in metadata["numeric_features"]
            fields.append({
                "name": column,
                "numeric": is_numeric,
                "choices": metadata.get("categorical_choices", {}).get(column, []),
                "value": request.POST.get(column, ""),
            })

    return render(request, "dashboard/predict.html", {
        "meta": metadata,
        "fields": fields,
        "result": result,
    })
