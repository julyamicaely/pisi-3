"""
Pacote de componentes reutilizáveis para o dashboard.
Importa utils e cards para facilitar uso.
"""

from .utils import (
    make_card,
    make_metric_card,
    build_metric_grid,
    make_section_header,
    make_tabs,
    make_loading_wrapper,
    make_alert,
    make_empty_state,
    make_page_header,
    make_info_tooltip,
)

from .cards import (
    build_confusion_matrix,
    build_feature_importance,
    build_metrics_table,
    build_classification_report_card,
    build_scatter_plot,
    build_histogram,
    build_box_plot,
    build_pie_chart,
    build_line_chart,
)

__all__ = [
    # Utils
    "make_card",
    "make_metric_card",
    "build_metric_grid",
    "make_section_header",
    "make_tabs",
    "make_loading_wrapper",
    "make_alert",
    "make_empty_state",
    "make_page_header",
    "make_info_tooltip",
    # Cards
    "build_confusion_matrix",
    "build_feature_importance",
    "build_metrics_table",
    "build_classification_report_card",
    "build_scatter_plot",
    "build_histogram",
    "build_box_plot",
    "build_pie_chart",
    "build_line_chart",
]
