"""
grading_engine.py
Core logic for the AI-powered student answer grading pipeline.

Agent pipeline (6 agents):
  Grader → Evidence → Integrity → Bias → Review → Explainer

Supports two modes:
  1. single_answer  : grade one typed answer against optional context
  2. full_paper     : upload question paper + student answer sheet → auto-grade all Qs
"""

import json
import re
from typing import TypedDict, Optional

from groq import Groq
from langgraph.graph import StateGraph

# ---------------------------------------------------------------------------
# Agent model config
# ---------------------------------------------------------------------------

AGENTS = {
    "grader":    {"model": "llama-3.3-70b-versatile"},
    "evidence":  {"model": "qwen/qwen3-32b"},
    "integrity": {"model": "llama-3.3-70b-versatile"},   # NEW
    "bias":      {"model": "qwen/qwen3-32b"},
    "explainer": {"model": "llama-3.3-70b-versatile"},
    "extractor": {"model": "llama-3.3-70b-versatile"},
    "matcher":   {"model": "llama-3.3-70b-versatile"},
}

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def call_llm(prompt, model, client, temperature=0, json_mode=False):
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AssessmentState(TypedDict):
    answer:           str
    question_context: str
    grade:            str
    evidence:         str
    integrity_report: str   # NEW — AI/plagiarism detection result
    bias_report:      str
    explanation:      str
    human_review:     bool


# ---------------------------------------------------------------------------
# Agent 1 — Grader
# ---------------------------------------------------------------------------

def make_grading_agent(client):
    def grading_agent(state):
        ctx = ""
        if state.get("question_context", "").strip():
            ctx = f'\nReference / Marking Scheme:\n"""\n{state["question_context"]}\n"""\nUse this as the rubric.\n'
        prompt = f"""You are an educational assessment expert.
{ctx}
Student Answer:
{state['answer']}

Return:
1. Score out of 100
2. Brief rationale
"""
        state["grade"] = call_llm(prompt, AGENTS["grader"]["model"], client)
        return state
    return grading_agent


# ---------------------------------------------------------------------------
# Agent 2 — Evidence
# ---------------------------------------------------------------------------

def make_evidence_agent(client):
    def evidence_agent(state):
        ctx = ""
        if state.get("question_context", "").strip():
            ctx = f'\nReference material:\n"""\n{state["question_context"]}\n"""\n'
        prompt = f"""Extract evidence from the student's answer that supports the assigned score.
{ctx}
Student Answer:
{state['answer']}

Return ONLY valid JSON:
{{
  "supporting_evidence": ["Point 1", "Point 2"],
  "missing_information": ["Missing detail 1", "Missing detail 2"]
}}
"""
        state["evidence"] = call_llm(
            prompt, AGENTS["evidence"]["model"], client, json_mode=True
        )
        return state
    return evidence_agent


# ---------------------------------------------------------------------------
# Agent 3 — Integrity (NEW: AI-generated + plagiarism detection)
# ---------------------------------------------------------------------------

def make_integrity_agent(client):
    def integrity_agent(state):
        prompt = f"""You are an academic integrity specialist trained to detect:
1. AI-generated content (ChatGPT, Claude, Gemini, etc.)
2. Copied or plagiarised text (unnatural phrasing, inconsistent voice, suspiciously perfect structure)

Analyze the student's answer below and return a detailed integrity report.

Student Answer:
\"\"\"
{state['answer']}
\"\"\"

Scoring guide:
- ai_probability   : 0.0–1.0  (1.0 = almost certainly AI-generated)
- plagiarism_risk  : "Low" | "Medium" | "High"
- originality_score: 0–100 (100 = fully original human writing)

Look for these AI signals: overly formal transitions ("Furthermore", "In conclusion"),
perfect paragraph structure, lack of personal voice, generic examples,
hedging language ("It is important to note"), uniform sentence length.

Look for these plagiarism signals: sudden vocabulary shift, mixed tenses,
unusually polished phrasing inconsistent with the rest of the answer.

Return ONLY valid JSON:
{{
  "ai_probability": 0.12,
  "ai_verdict": "Human-written | Likely AI | Highly Likely AI",
  "plagiarism_risk": "Low | Medium | High",
  "originality_score": 88,
  "ai_signals": ["signal 1", "signal 2"],
  "plagiarism_signals": ["signal 1"],
  "recommendation": "Accept | Review | Reject",
  "summary": "One sentence summary of findings."
}}
"""
        state["integrity_report"] = call_llm(
            prompt, AGENTS["integrity"]["model"], client, json_mode=True
        )
        return state
    return integrity_agent


# ---------------------------------------------------------------------------
# Agent 4 — Bias Auditor
# ---------------------------------------------------------------------------

def make_bias_auditor_agent(client):
    def bias_auditor_agent(state):
        prompt = f"""You are an educational fairness auditor.

Student Answer: {state['answer']}
Assigned Grade: {state['grade']}
Supporting Evidence: {state['evidence']}

Return ONLY valid JSON:
{{
  "fairness": "Fair|Potential Issue",
  "confidence": 0.95,
  "concerns": ["..."]
}}
"""
        state["bias_report"] = call_llm(
            prompt, AGENTS["bias"]["model"], client, json_mode=True
        )
        return state
    return bias_auditor_agent


# ---------------------------------------------------------------------------
# Agent 5 — Review (rule-based, uses integrity + bias)
# ---------------------------------------------------------------------------

def make_review_agent():
    def review_agent(state):
        trigger_words    = ["bias", "concern", "unfair", "low confidence"]
        human_review_needed = False

        # Bias check
        try:
            report   = json.loads(state["bias_report"])
            fairness = report.get("fairness", "").lower()
            concerns = report.get("concerns", [])
        except json.JSONDecodeError:
            fairness = state["bias_report"].lower()
            concerns = []
            human_review_needed = True

        if any(w in fairness for w in trigger_words):
            human_review_needed = True
        for c in concerns:
            if any(w in c.lower() for w in trigger_words):
                human_review_needed = True
                break
        if "<think>" in state["bias_report"].lower():
            human_review_needed = True

        # Integrity check — flag if AI likely or plagiarism risk Medium/High
        try:
            ir = json.loads(state["integrity_report"])
            if ir.get("ai_probability", 0) >= 0.6:
                human_review_needed = True
            if ir.get("plagiarism_risk", "Low") in ("Medium", "High"):
                human_review_needed = True
            if ir.get("recommendation", "Accept") in ("Review", "Reject"):
                human_review_needed = True
        except Exception:
            pass

        state["human_review"] = human_review_needed
        return state
    return review_agent


# ---------------------------------------------------------------------------
# Agent 6 — Explainer
# ---------------------------------------------------------------------------

def make_explanation_agent(client):
    def explanation_agent(state):
        # Include integrity warning in explanation if flagged
        integrity_note = ""
        try:
            ir = json.loads(state["integrity_report"])
            if ir.get("ai_probability", 0) >= 0.6 or ir.get("plagiarism_risk", "Low") != "Low":
                integrity_note = f"\n\nNote: This answer was flagged by the Integrity Agent — {ir.get('summary', '')}"
        except Exception:
            pass

        prompt = f"""Generate a student-friendly explanation of their grade.

Grade: {state['grade']}
Evidence: {state['evidence']}
Fairness Audit: {state['bias_report']}
{integrity_note}

Be constructive, specific, and encouraging. Tell the student what they did well,
what they missed, and how they can improve next time.
"""
        state["explanation"] = call_llm(
            prompt, AGENTS["explainer"]["model"], client
        )
        return state
    return explanation_agent


# ---------------------------------------------------------------------------
# Graph builder (6-agent pipeline)
# ---------------------------------------------------------------------------

def build_graph(client):
    builder = StateGraph(AssessmentState)
    builder.add_node("grader",    make_grading_agent(client))
    builder.add_node("evidence",  make_evidence_agent(client))
    builder.add_node("integrity", make_integrity_agent(client))   # NEW
    builder.add_node("bias",      make_bias_auditor_agent(client))
    builder.add_node("review",    make_review_agent())
    builder.add_node("explainer", make_explanation_agent(client))

    builder.set_entry_point("grader")
    builder.add_edge("grader",    "evidence")
    builder.add_edge("evidence",  "integrity")   # NEW edge
    builder.add_edge("integrity", "bias")        # NEW edge
    builder.add_edge("bias",      "review")
    builder.add_edge("review",    "explainer")
    return builder.compile()


def run_assessment(answer, api_key, question_context=""):
    """Grade a single answer. Returns AssessmentState."""
    client = Groq(api_key=api_key)
    graph  = build_graph(client)
    return graph.invoke({
        "answer":           answer,
        "question_context": question_context,
        "grade":            "",
        "evidence":         "",
        "integrity_report": "",
        "bias_report":      "",
        "explanation":      "",
        "human_review":     False,
    })


# ===========================================================================
# Full paper grading (Option C)
# ===========================================================================

def extract_questions(paper_text, client):
    prompt = f"""You are given a question paper (and possibly a marking scheme).
Extract every question and its model answer / marking guide if present.

Question Paper:
\"\"\"
{paper_text[:6000]}
\"\"\"

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "number": "1",
      "question": "Full question text here",
      "model_answer": "Expected answer or marking points (empty string if not given)"
    }}
  ]
}}
"""
    raw = call_llm(prompt, AGENTS["extractor"]["model"], client, json_mode=True)
    try:
        return json.loads(raw).get("questions", [])
    except Exception:
        return []


def extract_student_answers(answer_sheet_text, questions, client):
    q_numbers = [q["number"] for q in questions]
    prompt = f"""You are given a student's answer sheet.
Extract each answer and match it to the correct question number.
Expected question numbers: {q_numbers}

Student Answer Sheet:
\"\"\"
{answer_sheet_text[:6000]}
\"\"\"

Return ONLY valid JSON:
{{
  "answers": [
    {{
      "number": "1",
      "student_answer": "Student's answer text here"
    }}
  ]
}}
If a question has no answer, use an empty string.
"""
    raw = call_llm(prompt, AGENTS["matcher"]["model"], client, json_mode=True)
    try:
        return json.loads(raw).get("answers", [])
    except Exception:
        return []


def extract_numeric_score(grade_text) -> Optional[int]:
    m = re.search(r"(\d{1,3})\s*/\s*100", grade_text)
    if m: return int(m.group(1))
    m = re.search(r"\b(\d{1,3})\b", grade_text)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100: return v
    return None


def run_full_paper(question_paper_text, answer_sheet_text, api_key, progress_callback=None):
    """
    Grade a full exam paper end-to-end.
    Returns dict with questions list, total_score, max_score, percentage,
    human_review_flags, and integrity_flags.
    """
    client = Groq(api_key=api_key)

    if progress_callback:
        progress_callback(0, 1, "📄 Extracting questions from paper…")
    questions = extract_questions(question_paper_text, client)
    if not questions:
        raise ValueError(
            "Could not extract any questions from the question paper. "
            "Please check the file format."
        )

    if progress_callback:
        progress_callback(0, 1, "📝 Extracting answers from student sheet…")
    student_answers_raw = extract_student_answers(answer_sheet_text, questions, client)
    sa_lookup = {a["number"]: a.get("student_answer", "") for a in student_answers_raw}

    total       = len(questions)
    graded      = []
    total_score = 0
    human_flags = 0
    integ_flags = 0   # NEW

    for i, q in enumerate(questions):
        num            = q["number"]
        question_text  = q["question"]
        model_answer   = q.get("model_answer", "")
        student_answer = sa_lookup.get(num, "")

        if progress_callback:
            progress_callback(i, total, f"🤖 Grading question {num} of {total}…")

        context = f"Question:\n{question_text}"
        if model_answer.strip():
            context += f"\n\nModel Answer / Marking Scheme:\n{model_answer}"

        if not student_answer.strip():
            result = {
                "answer":           "(No answer provided)",
                "question_context": context,
                "grade":            "Score: 0/100\nNo answer was provided.",
                "evidence":         json.dumps({"supporting_evidence": [], "missing_information": ["No answer submitted"]}),
                "integrity_report": json.dumps({"ai_probability": 0, "ai_verdict": "N/A", "plagiarism_risk": "Low", "originality_score": 100, "ai_signals": [], "plagiarism_signals": [], "recommendation": "Accept", "summary": "No answer submitted."}),
                "bias_report":      json.dumps({"fairness": "Fair", "confidence": 1.0, "concerns": []}),
                "explanation":      "No answer was submitted for this question.",
                "human_review":     False,
            }
        else:
            result = run_assessment(
                answer=student_answer,
                api_key=api_key,
                question_context=context,
            )

        score = extract_numeric_score(result["grade"])
        if score is not None:
            total_score += score
        if result["human_review"]:
            human_flags += 1

        # Count integrity flags
        try:
            ir = json.loads(result["integrity_report"])
            if ir.get("ai_probability", 0) >= 0.6 or ir.get("plagiarism_risk", "Low") != "Low":
                integ_flags += 1
        except Exception:
            pass

        graded.append({
            "number":         num,
            "question":       question_text,
            "model_answer":   model_answer,
            "student_answer": student_answer,
            "result":         result,
            "numeric_score":  score,
        })

    if progress_callback:
        progress_callback(total, total, "✅ All questions graded!")

    max_score  = total * 100
    percentage = round((total_score / max_score) * 100, 1) if max_score > 0 else 0.0

    return {
        "questions":          graded,
        "total_score":        total_score,
        "max_score":          max_score,
        "percentage":         percentage,
        "human_review_flags": human_flags,
        "integrity_flags":    integ_flags,
    }
