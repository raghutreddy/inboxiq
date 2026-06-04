# app.py - FastAPI backend for InboxIQ
# Turns the pipeline into a real API service

import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from pipeline import process_email
from cost_tracker import PipelineTracker

# ---- Create the FastAPI app ----
app = FastAPI(
    title="InboxIQ",
    description="AI Email Triage Agent — classifies emails, drafts replies, extracts action items",
    version="1.0.0"
)

# ---- Request/Response Models ----

class EmailRequest(BaseModel):
    """What the user sends to our API."""
    email_text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="The raw email text to triage",
        json_schema_extra={
            "example": "From: boss@company.com\nSubject: Need slides by 5 PM\n\nPlease send your slides by 5 PM today."
        }
    )

class ActionItem(BaseModel):
    """One extracted action item."""
    task: str
    owner: str
    deadline: str
    priority: str

class TriageResponse(BaseModel):
    """What our API sends back."""
    category: str
    urgency_score: int
    reasoning: str
    suggested_action: str
    reply_draft: Optional[str] = None
    action_items: list = []
    cost_usd: float
    processing_time_seconds: float
    model_used: str
    provider: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str


# ---- API Endpoints ----

@app.get("/", response_model=HealthResponse)
def health_check():
    """
    Health check — confirms the API is running.
    Every production API has this. Monitoring tools ping it continuously.
    """
    return HealthResponse(
        status="healthy",
        service="InboxIQ",
        version="1.0.0"
    )


@app.post("/triage", response_model=TriageResponse)
def triage_email(request: EmailRequest):
    """
    Main endpoint — send an email, get back classification + reply + actions.
    
    This is where the magic happens:
    1. Validates the input (Pydantic checks length, type)
    2. Runs the 3-stage pipeline (classify → reply → extract)
    3. Returns structured JSON with cost tracking
    """
    
    start_time = time.time()
    tracker = PipelineTracker()

    try:
        # Run the full pipeline
        result = process_email(request.email_text, tracker)

        # Extract classification details
        classification = result.get("classification", {})
        
        # Check for input validation errors from pipeline
        if classification.get("category") == "ERROR":
            raise HTTPException(
                status_code=400,
                detail=classification.get("error", "Invalid input")
            )

        # Extract action items safely
        actions_data = result.get("action_items", {})
        action_items = actions_data.get("action_items", []) if isinstance(actions_data, dict) else []

        # Get cost summary
        cost_summary = tracker.get_summary()
        processing_time = round(time.time() - start_time, 2)

        # Find which provider/model was used for classification
        calls = cost_summary.get("per_call_breakdown", [])
        provider = "unknown"
        model_used = "unknown"
        if calls:
            provider = calls[0].get("provider", "unknown")
            model_used = calls[0].get("model", "unknown")

        return TriageResponse(
            category=classification.get("category", "UNKNOWN"),
            urgency_score=classification.get("urgency_score", 0),
            reasoning=classification.get("reasoning", ""),
            suggested_action=classification.get("suggested_action", ""),
            reply_draft=result.get("reply_draft"),
            action_items=action_items,
            cost_usd=cost_summary.get("total_cost_usd", 0),
            processing_time_seconds=processing_time,
            model_used=model_used,
            provider=provider
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )


# ---- Run the server ----
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting InboxIQ API server...")
    print("📖 API docs: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)