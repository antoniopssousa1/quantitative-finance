# ─────────────────────────────────────────
#  REUSABLE UI COMPONENTS
# ─────────────────────────────────────────

from dash import dcc, html
import dash_bootstrap_components as dbc
from dashboard.theme import (
    ACCENT, ACCENT2, ACCENT3, BORDER, CARD, MUTED, PANEL, TEXT, FONT, RED, GREEN
)


def card(title: str, *children):
    """Dark card with a teal header label."""
    return dbc.Card([
        dbc.CardHeader(
            html.Span(title, style={
                "color": ACCENT, "fontFamily": FONT, "fontSize": "10px",
                "letterSpacing": "2px", "textTransform": "uppercase",
            }),
            style={"background": CARD, "borderBottom": f"1px solid {BORDER}", "padding": "7px 14px"}
        ),
        dbc.CardBody(list(children), style={"background": CARD, "padding": "12px"}),
    ], style={"border": f"1px solid {BORDER}", "borderRadius": "6px", "marginBottom": "12px"})


def stat(label: str, value: str, color: str = ACCENT):
    """Small KPI tile."""
    return html.Div([
        html.Div(label, style={
            "color": MUTED, "fontSize": "9px", "letterSpacing": "1px",
            "textTransform": "uppercase", "fontFamily": FONT,
        }),
        html.Div(value, style={
            "color": color, "fontSize": "19px", "fontWeight": "bold",
            "fontFamily": FONT, "marginTop": "2px",
        }),
    ], style={
        "background": PANEL, "border": f"1px solid {BORDER}",
        "borderRadius": "4px", "padding": "9px 13px",
        "flex": "1", "minWidth": "110px",
    })


def stat_row(*items):
    """Flex row of stat tiles."""
    return html.Div(list(items), style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "10px"})


def input_field(label: str, id: str, value, type: str = "number", step=None, min=None, max=None):
    """Labelled input field."""
    return html.Div([
        html.Label(label, style={
            "color": MUTED, "fontSize": "9px", "fontFamily": FONT,
            "letterSpacing": "1px", "display": "block", "marginBottom": "3px",
        }),
        dcc.Input(
            id=id, type=type, value=value, step=step, min=min, max=max,
            debounce=False,
            style={
                "background": PANEL, "border": f"1px solid {BORDER}",
                "color": TEXT, "fontFamily": FONT, "fontSize": "13px",
                "padding": "5px 8px", "borderRadius": "3px", "width": "100%",
            },
        ),
    ], style={"marginBottom": "8px"})


def section_label(text: str, color: str = ACCENT):
    return html.Div(text, style={
        "color": color, "fontSize": "9px", "letterSpacing": "2px",
        "textTransform": "uppercase", "fontFamily": FONT, "marginBottom": "7px",
    })


def divider():
    return html.Hr(style={"borderColor": BORDER, "margin": "12px 0"})


def run_button(id: str, label: str = "RUN"):
    return dbc.Button(label, id=id, color="success", size="sm", style={
        "width": "100%", "fontFamily": FONT, "letterSpacing": "2px", "marginTop": "4px",
    })


def graph(id: str, height: int = 340):
    return dcc.Graph(id=id, style={"height": f"{height}px"}, config={"displayModeBar": False})
