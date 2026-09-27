from fastapi import FastAPI

app = FastAPI(
    title="Predictive Maintenance API",
    version="0.1.0",
    description="Placeholder API for the predictive-maintenance project.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
