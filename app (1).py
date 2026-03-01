import streamlit as st
import json
from graphviz import Digraph
from mlx import PMAgent

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Project Manager Agent",
    page_icon="🚀",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS (ULTRA PRO MAX MODE)
# ─────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: white;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Cards */
.glass {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px);
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 0 30px rgba(0,0,0,0.4);
    transition: 0.3s ease;
}

.glass:hover {
    transform: translateY(-6px);
    box-shadow: 0 0 40px rgba(0,255,255,0.4);
}

/* Titles */
.title {
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(90deg, #00F5A0, #00D9F5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Severity badges */
.badge-high {
    background-color: #ff4b4b;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 600;
}

.badge-medium {
    background-color: #ff9f43;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 600;
}

.badge-low {
    background-color: #1dd1a1;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="title">⚡ AI Project Manager Control Room</div>', unsafe_allow_html=True)
st.markdown("### Intelligent Agile Governance Engine")

st.divider()

# ─────────────────────────────────────────────
# INITIALIZE AGENT
# ─────────────────────────────────────────────
agent = PMAgent()
agent.load_backlog("data.csv", "csv")

# ─────────────────────────────────────────────
# FEATURE INPUT
# ─────────────────────────────────────────────
feature = st.text_input("🧠 Enter a Feature to Decompose")

if st.button("🚀 Generate Execution Plan"):

    tickets = agent.break_down_feature(feature)
    blockers = agent.detect_blockers()
    summary = agent.generate_summary()

    st.divider()

    # ─────────────────────────────────────────
    # METRICS SECTION
    # ─────────────────────────────────────────
    # col1, col2, col3, col4 = st.columns(4)

    # col1.metric("📦 Total Issues", summary.total_issues)
    # col2.metric("🚧 In Progress", summary.in_progress)
    # col3.metric("⛔ Blocked", summary.blocked)
    # col4.metric("⚠️ At Risk", len(summary.at_risk))

    st.divider()

    # ─────────────────────────────────────────
    # GENERATED TICKETS
    # ─────────────────────────────────────────
    st.subheader("🧩 Generated Tickets")

    for t in tickets:
        with st.container():
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown(f"### {t.title}")
            st.write(f"**Story Points:** {t.estimated_story_points}")
            st.write(f"**Assigned Team:** {t.assigned_team}")
            st.write("**Acceptance Criteria:**")
            for ac in t.acceptance_criteria:
                st.write(f"- {ac}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ─────────────────────────────────────────
    # DEPENDENCY GRAPH
    # ─────────────────────────────────────────
    st.subheader("🔗 Dependency Graph")

    dot = Digraph()
    for t in tickets:
        dot.node(t.ticket_id, t.title)
        for dep in t.dependencies:
            dot.edge(dep, t.ticket_id)

    st.graphviz_chart(dot)

    st.divider()

    # ─────────────────────────────────────────
    # BLOCKERS SECTION
    # ─────────────────────────────────────────
    st.subheader("🚨 Active Blockers")

    for b in blockers:
        if b.severity == "high":
            badge = "badge-high"
        elif b.severity == "medium":
            badge = "badge-medium"
        else:
            badge = "badge-low"

        st.markdown(
            f'<div class="glass">'
            f'<b>{b.issue_id}</b> — {b.description} '
            f'<span class="{badge}">{b.severity.upper()}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    # ─────────────────────────────────────────
    # DAILY SUMMARY
    # ─────────────────────────────────────────
    st.subheader("📊 Executive Daily Summary")

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.write(summary.key_updates[0])
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ─────────────────────────────────────────
    # EXPORT JSON
    # ─────────────────────────────────────────
    st.subheader("📤 JSON Export (Evaluation Format)")

    export_data = agent.export_results(tickets=tickets, blockers=blockers)
    st.json(export_data)