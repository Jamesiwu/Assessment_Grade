"""
app.py  —  AI Grading Assistant
Two modes:
  • Single Answer  : type one answer, optional rubric upload
  • Full Paper     : upload Q paper + student answer sheet → auto-grade all Qs

6-agent pipeline:
  Grader → Evidence → Integrity → Bias Auditor → Review → Explainer
"""

import csv
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from grading_engine import run_assessment, run_full_paper

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Grading Assistant", page_icon="🎓", layout="wide")

st.title("🎓 AI Grading Assistant")
st.caption(
    "Powered by LangGraph · Groq · LLaMA 3.3 & Qwen3  —  "
    "Grader → Evidence → **Integrity** → Bias Auditor → Review → Explainer"
)

# ---------------------------------------------------------------------------
# CSV log helpers
# ---------------------------------------------------------------------------
LOG_FILE = Path("grading_log.csv")
LOG_COLS = ["timestamp","mode","question_file","answer_file_or_preview",
            "total_questions","score","percentage","human_review_flags","integrity_flags"]

def ensure_log():
    if not LOG_FILE.exists():
        with open(LOG_FILE,"w",newline="",encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_COLS).writeheader()

def append_log(row):
    ensure_log()
    with open(LOG_FILE,"a",newline="",encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=LOG_COLS).writerow(row)

def read_log():
    ensure_log()
    with open(LOG_FILE,"r",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def extract_score_str(grade_text):
    m = re.search(r"(\d{1,3})\s*/\s*100", grade_text)
    if m: return m.group(1)+"/100"
    m = re.search(r"\b(\d{1,3})\b", grade_text)
    return m.group(1) if m else "—"

# ---------------------------------------------------------------------------
# File text extractor
# ---------------------------------------------------------------------------
def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        try:
            import pypdf
            return "\n".join(
                p.extract_text() or ""
                for p in pypdf.PdfReader(io.BytesIO(uploaded_file.read())).pages
            )
        except Exception as e:
            st.warning(f"PDF read error: {e}"); return ""
    elif name.endswith(".docx"):
        try:
            import docx
            return "\n".join(
                p.text for p in docx.Document(io.BytesIO(uploaded_file.read())).paragraphs
            )
        except Exception as e:
            st.warning(f"DOCX read error: {e}"); return ""
    return ""

# ---------------------------------------------------------------------------
# Integrity panel — reusable component
# ---------------------------------------------------------------------------
def show_integrity_panel(integrity_report_str: str):
    """Render the integrity report as a styled Streamlit panel."""
    try:
        ir = json.loads(integrity_report_str)
    except Exception:
        st.markdown(integrity_report_str)
        return

    ai_prob   = ir.get("ai_probability", 0)
    ai_verdict= ir.get("ai_verdict", "Unknown")
    plag_risk = ir.get("plagiarism_risk", "Low")
    orig_score= ir.get("originality_score", 100)
    ai_sigs   = ir.get("ai_signals", [])
    plag_sigs = ir.get("plagiarism_signals", [])
    rec       = ir.get("recommendation", "Accept")
    summary   = ir.get("summary", "")

    # Verdict colors
    ai_color   = "green" if ai_prob < 0.4 else "orange" if ai_prob < 0.7 else "red"
    plag_color = "green" if plag_risk == "Low" else "orange" if plag_risk == "Medium" else "red"
    rec_color  = "green" if rec == "Accept" else "orange" if rec == "Review" else "red"

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🤖 AI Probability",   f"{int(ai_prob * 100)}%")
    c2.metric("📋 Plagiarism Risk",  plag_risk)
    c3.metric("✍️ Originality",      f"{orig_score}/100")
    c4.metric("📌 Recommendation",   rec)

    # Progress bars
    st.progress(ai_prob,               text=f"AI likelihood: :{ai_color}[{ai_verdict}]")
    st.progress(orig_score / 100,      text=f"Originality score: {orig_score}/100")

    # Recommendation badge
    st.markdown(f"**Recommendation:** :{rec_color}[{rec}]")
    if summary:
        st.caption(summary)

    # Signal details
    col_a, col_b = st.columns(2)
    with col_a:
        if ai_sigs:
            st.markdown("**🤖 AI Signals Detected**")
            for s in ai_sigs:
                st.markdown(f"- {s}")
        else:
            st.markdown("**🤖 AI Signals:** None detected ✅")
    with col_b:
        if plag_sigs and plag_sigs != [""]:
            st.markdown("**📋 Plagiarism Signals**")
            for s in plag_sigs:
                st.markdown(f"- {s}")
        else:
            st.markdown("**📋 Plagiarism Signals:** None detected ✅")

    # Warning banners
    if ai_prob >= 0.6:
        st.error("🚨 High AI-content probability — human review required before accepting this submission.")
    elif ai_prob >= 0.4:
        st.warning("⚠️ Moderate AI-content signals — consider reviewing this submission.")

    if plag_risk == "High":
        st.error("🚨 High plagiarism risk — verify originality before accepting.")
    elif plag_risk == "Medium":
        st.warning("⚠️ Medium plagiarism risk — some suspicious patterns detected.")

# ---------------------------------------------------------------------------
# API key helper
# ---------------------------------------------------------------------------
def get_api_key():
    try:    return st.secrets.get("GROQ_API_KEY", "")
    except: return os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key", value=get_api_key(), type="password",
        help="Get your free key at console.groq.com",
    )
    st.markdown("---")
    st.markdown(
        "**6-Agent Pipeline**\n\n"
        "`Grader` → `Evidence` → `Integrity` → `Bias` → `Review` → `Explainer`"
    )
    st.markdown("---")
    st.markdown("### 📋 Recent Assessments")
    rows = read_log()
    if rows:
        for r in reversed(rows[-5:]):
            flags = int(r.get("human_review_flags","0") or 0)
            iflags= int(r.get("integrity_flags","0") or 0)
            icon  = "🚨" if flags or iflags else "✅"
            st.markdown(
                f"{icon} **{r.get('score','—')}** · {r['timestamp'][:16]}\n\n"
                f"_{r.get('answer_file_or_preview','')[:35]}…_"
            )
    else:
        st.caption("No assessments yet.")

# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------
mode = st.radio(
    "Select grading mode",
    ["📝 Single Answer", "📄 Full Paper (Upload Q + A files)"],
    horizontal=True,
)
st.markdown("---")

# ===========================================================================
# MODE 1 — Single Answer
# ===========================================================================
if mode == "📝 Single Answer":

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.subheader("📂 Question / Marking Scheme (optional)")
        rubric_file = st.file_uploader(
            "Upload question paper or answer key",
            type=["pdf","docx","txt"], key="rubric",
        )
        question_context = ""
        if rubric_file:
            with st.spinner("Reading file…"):
                question_context = extract_text(rubric_file)
            if question_context.strip():
                st.success(f"✅ {len(question_context):,} chars from **{rubric_file.name}**")
                with st.expander("Preview"):
                    st.text(question_context[:800]+"…")
            else:
                st.warning("Could not extract text.")

    with col_r:
        st.subheader("✏️ Student Answer")
        student_answer = st.text_area(
            "answer", height=220, label_visibility="collapsed",
            placeholder="Paste or type the student's answer here…",
        )

    run_btn = st.button("▶ Run Assessment", type="primary", use_container_width=True)

    if run_btn:
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar."); st.stop()
        if not student_answer.strip():
            st.warning("Please enter a student answer."); st.stop()

        with st.spinner("Running 6-agent pipeline… 15–25 s"):
            try:
                result = run_assessment(student_answer.strip(), api_key, question_context)
            except Exception as e:
                st.error(f"Pipeline error: {e}"); st.stop()

        st.success("Assessment complete!")

        # 3-tab layout for results
        tab_grade, tab_integrity, tab_evidence, tab_explain, tab_raw = st.tabs(
            ["📊 Grade", "🔍 Integrity", "📋 Evidence", "💬 Explanation", "🛠 Raw JSON"]
        )

        with tab_grade:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Grade & Rationale")
                st.markdown(result["grade"])
            with c2:
                st.markdown("#### Fairness Audit")
                try:
                    bias = json.loads(result["bias_report"])
                    fairness   = bias.get("fairness","Unknown")
                    confidence = bias.get("confidence", None)
                    concerns   = bias.get("concerns",[])
                    color = "green" if "fair" in fairness.lower() else "orange"
                    st.markdown(f"**Fairness:** :{color}[{fairness}]")
                    if confidence:
                        st.progress(float(confidence), text=f"Confidence: {confidence:.0%}")
                    if concerns and concerns != ["..."]:
                        for c in concerns: st.markdown(f"- {c}")
                except Exception:
                    st.markdown(result["bias_report"])
            if result["human_review"]:
                st.warning("🚨 **Human Review Recommended**")
            else:
                st.success("✅ No human review required.")

        with tab_integrity:
            st.markdown("#### 🔍 Academic Integrity Report")
            show_integrity_panel(result["integrity_report"])

        with tab_evidence:
            st.markdown("#### Evidence Breakdown")
            try:
                ev = json.loads(result["evidence"])
                e1, e2 = st.columns(2)
                with e1:
                    st.markdown("**✅ Supporting Evidence**")
                    for p in ev.get("supporting_evidence",[]): st.markdown(f"- {p}")
                with e2:
                    st.markdown("**❌ Missing Information**")
                    for p in ev.get("missing_information",[]): st.markdown(f"- {p}")
            except Exception:
                st.markdown(result["evidence"])

        with tab_explain:
            st.markdown("#### Student-Friendly Explanation")
            st.markdown(result["explanation"])

        with tab_raw:
            st.json(result)

        # Log
        try:
            ir = json.loads(result["integrity_report"])
            i_flag = 1 if ir.get("ai_probability",0) >= 0.6 or ir.get("plagiarism_risk","Low") != "Low" else 0
        except Exception:
            i_flag = 0

        append_log({
            "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode":                   "single",
            "question_file":          rubric_file.name if rubric_file else "—",
            "answer_file_or_preview": student_answer.strip()[:60].replace("\n"," "),
            "total_questions":        1,
            "score":                  extract_score_str(result["grade"]),
            "percentage":             "—",
            "human_review_flags":     int(result["human_review"]),
            "integrity_flags":        i_flag,
        })
        st.sidebar.success("💾 Saved to log")

# ===========================================================================
# MODE 2 — Full Paper
# ===========================================================================
else:
    st.subheader("📄 Full Paper Grading")
    st.info(
        "Upload two files:\n\n"
        "**① Question Paper** — questions + optional model answers / marking scheme\n\n"
        "**② Student Answer Sheet** — the student's submitted answers\n\n"
        "The system extracts, matches, and grades every Q→A pair through all 6 agents."
    )

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("**① Question Paper / Marking Scheme**")
        qp_file = st.file_uploader("Upload question paper", type=["pdf","docx","txt"], key="qp")
        qp_text = ""
        if qp_file:
            with st.spinner("Reading…"):
                qp_text = extract_text(qp_file)
            if qp_text.strip():
                st.success(f"✅ {len(qp_text):,} chars — **{qp_file.name}**")
                with st.expander("Preview"): st.text(qp_text[:600]+"…")
            else:
                st.warning("Could not extract text."); qp_text = ""

    with col_r:
        st.markdown("**② Student Answer Sheet**")
        sa_file = st.file_uploader("Upload student answer sheet", type=["pdf","docx","txt"], key="sa")
        sa_text = ""
        if sa_file:
            with st.spinner("Reading…"):
                sa_text = extract_text(sa_file)
            if sa_text.strip():
                st.success(f"✅ {len(sa_text):,} chars — **{sa_file.name}**")
                with st.expander("Preview"): st.text(sa_text[:600]+"…")
            else:
                st.warning("Could not extract text."); sa_text = ""

    grade_btn = st.button(
        "▶ Grade Full Paper", type="primary", use_container_width=True,
        disabled=(not qp_text or not sa_text),
    )

    if grade_btn:
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar."); st.stop()

        prog_bar  = st.progress(0, text="Starting…")
        prog_text = st.empty()

        def update_progress(current, total, message):
            prog_bar.progress(int((current / max(total,1)) * 100), text=message)
            prog_text.markdown(f"_{message}_")

        try:
            report = run_full_paper(
                question_paper_text=qp_text,
                answer_sheet_text=sa_text,
                api_key=api_key,
                progress_callback=update_progress,
            )
        except ValueError as e:
            st.error(str(e)); st.stop()
        except Exception as e:
            st.error(f"Pipeline error: {e}"); st.stop()

        prog_bar.empty(); prog_text.empty()
        st.success("🎉 Full paper graded!")

        # ── Summary scorecard ──────────────────────────────────────────
        pct    = report["percentage"]
        total  = report["total_score"]
        maxs   = report["max_score"]
        flags  = report["human_review_flags"]
        iflags = report["integrity_flags"]
        nq     = len(report["questions"])

        grade_letter = ("A" if pct>=90 else "B" if pct>=80 else "C" if pct>=70 else "D" if pct>=60 else "F")

        sc1,sc2,sc3,sc4,sc5 = st.columns(5)
        sc1.metric("Total Score",        f"{total} / {maxs}")
        sc2.metric("Percentage",         f"{pct}%")
        sc3.metric("Grade",              grade_letter)
        sc4.metric("🚨 Review Flags",    flags)
        sc5.metric("🔍 Integrity Flags", iflags)

        st.progress(int(pct), text=f"Overall: {pct}%")

        # ── Per-question results ───────────────────────────────────────
        st.markdown("---")
        st.subheader("📋 Per-Question Results")

        for item in report["questions"]:
            num    = item["number"]
            score  = item["numeric_score"]
            result = item["result"]

            # Integrity quick-check
            try:
                ir       = json.loads(result["integrity_report"])
                ai_prob  = ir.get("ai_probability", 0)
                plag     = ir.get("plagiarism_risk","Low")
                i_icon   = "🔍⚠️" if (ai_prob >= 0.4 or plag != "Low") else "🔍✅"
            except Exception:
                i_icon = "🔍"

            flag  = "🚨" if result["human_review"] else "✅"
            color = "green" if (score or 0)>=70 else "orange" if (score or 0)>=50 else "red"

            with st.expander(
                f"{flag} {i_icon}  Question {num}  —  **:{color}[{score}/100]**",
                expanded=False,
            ):
                t1,t2,t3,t4,t5 = st.tabs(["📝 Q & A","📊 Grade","🔍 Integrity","📋 Evidence","💬 Explanation"])

                with t1:
                    c1,c2 = st.columns(2)
                    with c1:
                        st.markdown("**Question**"); st.markdown(item["question"])
                        if item["model_answer"].strip():
                            st.markdown("**Model Answer**"); st.markdown(item["model_answer"])
                    with c2:
                        st.markdown("**Student's Answer**")
                        st.markdown(item["student_answer"] if item["student_answer"].strip() else "_No answer provided_")

                with t2:
                    st.markdown(result["grade"])
                    try:
                        bias = json.loads(result["bias_report"])
                        fairness = bias.get("fairness","Unknown")
                        conf     = bias.get("confidence", None)
                        bc = "green" if "fair" in fairness.lower() else "orange"
                        st.markdown(f"**Fairness:** :{bc}[{fairness}]")
                        if conf: st.progress(float(conf), text=f"Confidence: {conf:.0%}")
                    except Exception: pass
                    if result["human_review"]:
                        st.warning("🚨 Human review recommended.")

                with t3:
                    show_integrity_panel(result["integrity_report"])

                with t4:
                    try:
                        ev = json.loads(result["evidence"])
                        e1,e2 = st.columns(2)
                        with e1:
                            st.markdown("**✅ Supporting**")
                            for p in ev.get("supporting_evidence",[]): st.markdown(f"- {p}")
                        with e2:
                            st.markdown("**❌ Missing**")
                            for p in ev.get("missing_information",[]): st.markdown(f"- {p}")
                    except Exception:
                        st.markdown(result["evidence"])

                with t5:
                    st.markdown(result["explanation"])

        # ── Score summary table ────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Score Summary Table")

        import pandas as pd
        table_rows = []
        for item in report["questions"]:
            r = item["result"]
            try:
                fairness = json.loads(r["bias_report"]).get("fairness","—")
            except: fairness = "—"
            try:
                ir2     = json.loads(r["integrity_report"])
                ai_pct  = f"{int(ir2.get('ai_probability',0)*100)}%"
                plag2   = ir2.get("plagiarism_risk","—")
                rec2    = ir2.get("recommendation","—")
            except:
                ai_pct = plag2 = rec2 = "—"

            table_rows.append({
                "Q#":            item["number"],
                "Score":         f"{item['numeric_score']}/100" if item["numeric_score"] is not None else "—",
                "Fairness":      fairness,
                "AI Prob":       ai_pct,
                "Plagiarism":    plag2,
                "Integrity Rec": rec2,
                "Human Review":  "🚨 Yes" if r["human_review"] else "✅ No",
                "Answer Given":  "Yes" if item["student_answer"].strip() else "No",
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        report_export = {k:v for k,v in report.items() if k != "questions"}
        st.download_button(
            "⬇️ Download Summary Report (JSON)",
            data=json.dumps(report_export, indent=2),
            file_name="grading_report.json",
            mime="application/json",
        )

        append_log({
            "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode":                   "full_paper",
            "question_file":          qp_file.name,
            "answer_file_or_preview": sa_file.name,
            "total_questions":        nq,
            "score":                  f"{total}/{maxs}",
            "percentage":             f"{pct}%",
            "human_review_flags":     flags,
            "integrity_flags":        iflags,
        })
        st.sidebar.success("💾 Saved to log")

# ---------------------------------------------------------------------------
# History table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📋 Assessment History")
rows = read_log()
if rows:
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Full Log (CSV)",
        data=LOG_FILE.read_bytes(),
        file_name="grading_log.csv",
        mime="text/csv",
    )
else:
    st.info("No assessments logged yet.")
