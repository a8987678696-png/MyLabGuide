import json
import re
from typing import Any, Dict, List

import streamlit as st
from google import genai
from google.genai import types


st.set_page_config(
    page_title="Lab Report Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_FILE_SIZE_MB = 15

LANGUAGES = {
    "English": "English",
    "اردو": "Urdu",
}


def get_secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret without displaying it."""
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


def get_client() -> genai.Client:
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key is not configured. Add GEMINI_API_KEY to "
            "your Streamlit Cloud app Secrets."
        )
    return genai.Client(api_key=api_key)


def extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON, tolerating Markdown code fences around the response."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("The AI response was not valid JSON. Please try again.")
        result = json.loads(match.group(0))
    if not isinstance(result, dict):
        raise ValueError("The AI returned an unexpected response format.")
    return result


def make_analysis_prompt(language: str) -> str:
    return f"""
You are a careful medical-laboratory report explainer. Review the attached
laboratory report and return ONLY valid JSON, with no Markdown or extra text.

The patient's preferred explanation language is {language}. Provide BOTH
English and Urdu explanations in the JSON regardless of the selected language.

Extract each test only when it is legible and supported by the document.
For every test include:
- test_name: string
- measured_value: string or null
- unit: string or null
- reference_range: string or null (copy the report's range; do not invent one)
- status: exactly one of "Within range", "Below range", "Above range",
  "Cannot interpret", or "Not stated"
- explanation_en: brief patient-friendly explanation in English
- explanation_ur: brief patient-friendly explanation in Urdu
- evidence_notes: short note about ambiguity, missing information, or
  relevant evidence; otherwise an empty string.

Status rules:
1. Compare only with the reference range printed for that test on this report.
2. If the range is absent, unclear, age/sex-specific but patient context is
   unavailable, or the result cannot reliably be compared, use "Cannot interpret".
3. If the report explicitly marks a result high/low but no range is visible,
   you may use "Above range" or "Below range" only if the report itself clearly
   identifies it that way; explain that the report flag was used.
4. Do not apply generic normal ranges. Do not diagnose disease or infer a cause.
5. Preserve units and the exact reported values. Do not silently correct OCR.

Pre-analytical awareness:
- Only identify possible sample-related concerns when the report/document
  explicitly provides evidence (for example, a visible hemolysis comment,
  clotted/insufficient specimen note, rejected sample note, wrong tube note,
  labeling discrepancy, processing delay, or storage problem).
- Never infer hemolysis, clotting, insufficient volume, wrong tube, labeling
  error, delay, or poor storage merely from an abnormal result.
- Do not automatically reject a sample. A concern is not proof of an error.
- If no supporting evidence is present, return an empty concerns list and say
  that the report alone does not establish a pre-analytical error.

Return this exact top-level JSON structure:
{{
  "patient_friendly_summary_en": "string",
  "patient_friendly_summary_ur": "string",
  "tests": [
    {{
      "test_name": "string",
      "measured_value": "string or null",
      "unit": "string or null",
      "reference_range": "string or null",
      "status": "Within range | Below range | Above range | Cannot interpret | Not stated",
      "explanation_en": "string",
      "explanation_ur": "string",
      "evidence_notes": "string"
    }}
  ],
  "preanalytical_concerns": [
    {{
      "issue": "string",
      "supporting_evidence": "string",
      "caution": "string"
    }}
  ],
  "preanalytical_summary_en": "string",
  "preanalytical_summary_ur": "string",
  "limitations": ["string"]
}}

If the document is not a laboratory report, say so in both summaries,
return an empty tests list, and explain the limitation. If text is unreadable,
do not guess; mention it in limitations.
"""


def analyze_report(file_bytes: bytes, mime_type: str, language: str) -> Dict[str, Any]:
    client = get_client()
    model = get_secret("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    prompt = make_analysis_prompt(language)
    response = client.models.generate_content(
        model=model,
        contents=[
            prompt,
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    response_text = getattr(response, "text", None)
    if not response_text:
        raise ValueError("The AI did not return readable content. Please try again.")
    return extract_json(response_text)


def safe_text(value: Any, fallback: str = "Not available") -> str:
    if value is None:
        return fallback
    text_value = str(value).strip()
    return text_value if text_value else fallback


def status_icon(status: str) -> str:
    return {
        "Within range": "🟢",
        "Below range": "🔵",
        "Above range": "🟠",
        "Cannot interpret": "⚪",
        "Not stated": "⚪",
    }.get(status, "⚪")


def render_report(data: Dict[str, Any], language: str) -> None:
    tests = data.get("tests", [])
    if not isinstance(tests, list):
        tests = []

    st.subheader("Report overview")
    summary_key = (
        "patient_friendly_summary_ur"
        if language == "Urdu"
        else "patient_friendly_summary_en"
    )
    st.info(safe_text(data.get(summary_key), "No summary was returned."))

    total = len(tests)
    within = sum(1 for item in tests if item.get("status") == "Within range")
    below = sum(1 for item in tests if item.get("status") == "Below range")
    above = sum(1 for item in tests if item.get("status") == "Above range")
    cannot = sum(
        1 for item in tests
        if item.get("status") in ("Cannot interpret", "Not stated")
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Tests found", total)
    metric_cols[1].metric("Within range", within)
    metric_cols[2].metric("Below range", below)
    metric_cols[3].metric("Above range", above)
    metric_cols[4].metric("Need review", cannot)

    st.subheader("Extracted test results")
    if not tests:
        st.warning("No test results could be reliably extracted from this file.")
    else:
        table_rows = []
        for item in tests:
            if not isinstance(item, dict):
                continue
            status = safe_text(item.get("status"), "Cannot interpret")
            table_rows.append(
                {
                    "Test": safe_text(item.get("test_name")),
                    "Result": safe_text(item.get("measured_value")),
                    "Unit": safe_text(item.get("unit"), "—"),
                    "Reference range": safe_text(item.get("reference_range"), "Not shown"),
                    "Status": f"{status_icon(status)} {status}",
                }
            )
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.caption(
        "Status is based on the reference range printed on the uploaded report. "
        "A result marked 'Cannot interpret' should not be treated as normal or abnormal."
    )

    st.subheader("Test-by-test explanations")
    for index, item in enumerate(tests):
        if not isinstance(item, dict):
            continue
        test_name = safe_text(item.get("test_name"), f"Test {index + 1}")
        status = safe_text(item.get("status"), "Cannot interpret")
        with st.expander(f"{status_icon(status)} {test_name} — {status}", expanded=(index == 0)):
            cols = st.columns(3)
            cols[0].write("**Measured result**")
            cols[0].write(safe_text(item.get("measured_value")))
            cols[1].write("**Unit**")
            cols[1].write(safe_text(item.get("unit"), "Not stated"))
            cols[2].write("**Report reference range**")
            cols[2].write(safe_text(item.get("reference_range"), "Not provided"))

            explanation_key = "explanation_ur" if language == "Urdu" else "explanation_en"
            st.write(safe_text(item.get(explanation_key), "No explanation available."))
            if item.get("evidence_notes"):
                st.caption(f"Note: {item['evidence_notes']}")

    st.subheader("Pre-analytical sample awareness")
    concerns = data.get("preanalytical_concerns", [])
    summary_key = (
        "preanalytical_summary_ur"
        if language == "Urdu"
        else "preanalytical_summary_en"
    )
    st.write(safe_text(data.get(summary_key), "No sample-quality summary was returned."))
    if isinstance(concerns, list) and concerns:
        for concern in concerns:
            if not isinstance(concern, dict):
                continue
            st.warning(
                f"**{safe_text(concern.get('issue'))}**\n\n"
                f"Evidence in report: {safe_text(concern.get('supporting_evidence'))}\n\n"
                f"Caution: {safe_text(concern.get('caution'))}"
            )
    else:
        st.success(
            "No specific pre-analytical issue was established from the information "
            "visible in this report. This does not prove that sample collection or "
            "handling was error-free."
        )

    limitations = data.get("limitations", [])
    if isinstance(limitations, list) and limitations:
        st.subheader("Reading limitations")
        for limitation in limitations:
            st.write(f"• {limitation}")


def answer_question(question: str, report_data: Dict[str, Any], language: str) -> str:
    client = get_client()
    model = get_secret("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    context = json.dumps(report_data, ensure_ascii=False)
    prompt = f"""
You are a cautious assistant explaining a patient's laboratory report.
Answer in {language}. Use only the report data supplied below. Explain terms
in plain language. Do not diagnose, prescribe, recommend changing medication,
or invent reference ranges. If the report does not contain enough information,
say so and suggest asking a qualified healthcare professional. For urgent or
severe symptoms, advise seeking urgent medical care. Remind the user that this
is educational information, not a diagnosis.

REPORT DATA:
{context}

PATIENT QUESTION:
{question}
"""
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return safe_text(getattr(response, "text", None), "I could not generate an answer.")


st.title("🧪 Medical Laboratory Report Assistant")
st.write(
    "Understand the results printed on your laboratory report with clear, "
    "patient-friendly explanations."
)

with st.sidebar:
    st.header("Settings")
    language_choice = st.selectbox("Explanation language", ["English", "Urdu"])
    st.caption(f"AI model: {get_secret('GEMINI_MODEL', DEFAULT_MODEL) or DEFAULT_MODEL}")
    st.divider()
    st.subheader("Privacy")
    st.write(
        "The uploaded file is processed in memory for this session and is not "
        "saved by this app to a database or file storage. Avoid uploading reports "
        "you are not authorized to share. The AI provider processes the content "
        "to generate the explanation."
    )
    if st.button("Clear report and chat", use_container_width=True):
        for key in ("report_data", "report_name", "chat_history"):
            st.session_state.pop(key, None)
        st.rerun()

st.warning(
    "Educational use only. This application does not provide a medical diagnosis "
    "and does not replace advice from a doctor or qualified laboratory professional."
)

st.markdown("### 1. Upload your laboratory report")
uploaded_file = st.file_uploader(
    "Choose a report image or PDF",
    type=["jpg", "jpeg", "png", "pdf"],
    help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB. Ensure the report is clear and readable.",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    mime_by_extension = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "pdf": "application/pdf",
    }
    mime_type = mime_by_extension.get(extension)

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File is too large. Please upload a file smaller than {MAX_FILE_SIZE_MB} MB.")
    elif mime_type is None:
        st.error("Unsupported file type. Upload JPG, JPEG, PNG, or PDF.")
    else:
        st.caption(f"Selected file: {uploaded_file.name} ({file_size_mb:.2f} MB)")
        if st.button("Analyze report", type="primary", use_container_width=True):
            with st.spinner("Reading the report and preparing explanations..."):
                try:
                    result = analyze_report(file_bytes, mime_type, language_choice)
                    st.session_state["report_data"] = result
                    st.session_state["report_name"] = uploaded_file.name
                    st.session_state["chat_history"] = []
                    st.success("Analysis completed.")
                except Exception as exc:
                    # Show a useful error without printing secrets or uploaded content.
                    st.error(f"Could not analyze this report: {exc}")

if "report_data" in st.session_state:
    st.divider()
    st.caption(f"Current report: {st.session_state.get('report_name', 'Uploaded report')}")
    render_report(st.session_state["report_data"], language_choice)

    st.divider()
    st.subheader("2. Ask a question about your report")
    st.write("Ask what a test means, why a value is flagged, or what information is missing.")
    question = st.chat_input("Type your question about this report...")
    if question:
        history = st.session_state.setdefault("chat_history", [])
        history.append({"role": "user", "content": question})
        with st.spinner("Preparing an answer..."):
            try:
                reply = answer_question(
                    question,
                    st.session_state["report_data"],
                    language_choice,
                )
            except Exception as exc:
                reply = f"I could not answer right now: {exc}"
        history.append({"role": "assistant", "content": reply})

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.write(message["content"])

st.divider()
st.caption(
    "Important: Laboratory reference intervals can vary by laboratory, method, "
    "age, pregnancy status, and other factors. Discuss results and symptoms with "
    "your healthcare professional. Never delay urgent care because of this app."
)
