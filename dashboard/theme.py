# ─────────────────────────────────────────
#  THEME  —  Bloomberg-style dark terminal
# ─────────────────────────────────────────

BG       = "#0a0e1a"
PANEL    = "#0d1117"
CARD     = "#111827"
BORDER   = "#1f2937"
ACCENT   = "#00d4aa"   # teal  – primary
ACCENT2  = "#3b82f6"   # blue  – secondary
ACCENT3  = "#f59e0b"   # amber – warning / highlight
RED      = "#ef4444"
GREEN    = "#22c55e"
TEXT     = "#e2e8f0"
MUTED    = "#6b7280"
FONT     = "JetBrains Mono, Consolas, monospace"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, family=FONT, size=11),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
    margin=dict(l=45, r=20, t=38, b=40),
)
