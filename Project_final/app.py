from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
import pickle
import os


app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load Local Model (Pipeline)
local_model = None
try:
    if os.path.exists("model_pipeline.pkl"):
        with open("model_pipeline.pkl", "rb") as f:
            local_model = pickle.load(f)
        print("Local model loaded successfully.")
    else:
        print("Warning: 'model_pipeline.pkl' not found. Local prediction is unavailable until a model is trained.")
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize Verifier (reverting to ClaimVerifier)
from verifier import ClaimVerifier
claim_verifier = ClaimVerifier()

# Request Models
class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Enter a statement to check.")
        return value.strip()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.post("/predict_local")
async def predict_local(request: TextRequest):
    """
    Endpoint for local linguistic analysis.
    Uses the refined Logistic Regression model (74% acc) based on Politifact logic.
    """
    if not local_model:
        return JSONResponse(status_code=503, content={"error": "Local model not available. Please train it first."})
    
    try:
        # Predict class
        prediction = local_model.predict([request.text])[0] # 1=Real, 0=Fake
        # Get probability
        probs = local_model.predict_proba([request.text])[0]
        confidence = round(probs[prediction] * 100, 2)
        
        result = "Real" if prediction == 1 else "Fake"
        
        # Politifact logic: 'true', 'mostly-true' mapped to 1 (Real)
        # 'false', 'barely-true', 'pants-fire' mapped to 0 (Fake)
        
        return {
            "verdict": result,
            "confidence": confidence,
            "type": "Pattern Analysis"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/verify_realtime")
def verify_realtime(request: TextRequest):
    """
    Endpoint for real-time verification using ClaimVerifier (GNews + Tavily + Groq).
    """
    try:
        result = claim_verifier.verify(request.text)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
