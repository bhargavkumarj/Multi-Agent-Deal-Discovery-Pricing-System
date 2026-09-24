"""Monitoring UI: what the agents are doing, and what they have found."""

import logging
import queue
import threading

import gradio as gr
import plotly.graph_objects as go

from core import logs
from core.config import settings
from core.framework import DealDiscovery

CATEGORY_COLOURS = {
    "Appliances": "#00b0b0",
    "Automotive": "#4488ff",
    "Cell_Phones_and_Accessories": "#a5673f",
    "Electronics": "#ff8c00",
    "Musical_Instruments": "#d6c000",
    "Office_Products": "#2e9e5b",
    "Tools_and_Home_Improvement": "#9b59b6",
    "Toys_and_Games": "#d3453e",
}


class LogStream(logging.Handler):
    def __init__(self):
        super().__init__()
        self.queue: queue.Queue[str] = queue.Queue()
        self.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        self.queue.put(self.format(record))


def panel(lines: list[str]) -> str:
    body = "<br>".join(lines[-30:])
    return (
        "<div style='height:380px;overflow-y:auto;border:1px solid #333;"
        f"background:#16161d;padding:10px;font-family:ui-monospace,monospace;font-size:12px'>{body}</div>"
    )


def rows(memory):
    return [
        [
            opportunity.deal.product_description[:220],
            f"${opportunity.deal.price:,.2f}",
            f"${opportunity.estimate:,.2f}",
            f"${opportunity.discount:,.2f}",
            "yes" if memory.was_alerted(opportunity) else "",
            opportunity.deal.url,
        ]
        for opportunity in memory.best(30)
    ]


class Dashboard:
    def __init__(self):
        self.framework = DealDiscovery()
        self.stream = LogStream()
        logging.getLogger().addHandler(self.stream)

    def catalogue_plot(self):
        documents, reduced, categories = self.framework.plot_data()
        figure = go.Figure(
            data=[
                go.Scatter3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=[CATEGORY_COLOURS.get(c, "#888") for c in categories],
                        opacity=0.7,
                    ),
                    text=[f"{c}<br>{d[:60]}" for c, d in zip(categories, documents)],
                    hoverinfo="text",
                )
            ]
        )
        figure.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(xaxis_title="", yaxis_title="", zaxis_title=""),
            template="plotly_dark",
        )
        return figure

    def cycle(self, lines):
        result: list = []
        worker = threading.Thread(target=lambda: result.append(self.framework.cycle()))
        worker.start()

        while worker.is_alive() or not self.stream.queue.empty():
            try:
                lines = lines + [logs.to_html(self.stream.queue.get(timeout=0.2))]
            except queue.Empty:
                continue
            yield lines, panel(lines), rows(self.framework.memory)

        worker.join()
        yield lines, panel(lines), rows(self.framework.memory)

    def build(self) -> gr.Blocks:
        with gr.Blocks(title="Deal Discovery", fill_width=True, theme=gr.themes.Base()) as ui:
            lines = gr.State([])
            gr.Markdown(
                "## Deal Discovery\n"
                "Seven agents watch deal feeds, price each item with an ensemble and "
                f"alert on anything more than ${settings.discount_threshold:,.0f} below "
                "its estimated value."
            )
            table = gr.Dataframe(
                headers=["Deal", "Price", "Estimate", "Discount", "Alerted", "URL"],
                column_widths=[6, 1, 1, 1, 1, 3],
                wrap=True,
                row_count=8,
                max_height=340,
            )
            with gr.Row():
                log_view = gr.HTML()
                gr.Plot(value=self.catalogue_plot(), show_label=False)
            with gr.Row():
                scan = gr.Button("Run a cycle now", variant="primary")

            ui.load(self.cycle, inputs=[lines], outputs=[lines, log_view, table])
            scan.click(self.cycle, inputs=[lines], outputs=[lines, log_view, table])
            gr.Timer(value=600).tick(self.cycle, inputs=[lines], outputs=[lines, log_view, table])
        return ui


if __name__ == "__main__":
    Dashboard().build().launch(inbrowser=True)
