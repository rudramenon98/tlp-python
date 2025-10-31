import os
import sys
from itertools import chain
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd
from ai_doc_parser.training.classifier_trainer import FEATURE_COLUMNS
from ai_doc_parser.training.labeller import TextClass
from PyQt5.QtCore import QEvent, QLineF, QRectF, QSettings, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QCloseEvent,
    QColor,
    QFont,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QShowEvent,
    QWheelEvent,
)
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ClickableStatusLabel(QLabel):
    """A clickable QLabel that copies file paths to clipboard when clicked."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; padding: 2px; }"
            "QLabel:hover { background-color: #e0e0e0; border: 1px solid #999; }"
        )
        self.file_path = ""
        self.original_text = ""
        self.reset_timer = QTimer()
        self.reset_timer.setSingleShot(True)
        self.reset_timer.timeout.connect(self._reset_display)

    def set_file_path(self, file_path: str) -> None:
        """Set the file path to copy when clicked."""
        self.file_path = file_path

    def _reset_display(self) -> None:
        """Reset the display to original text and styling."""
        self.setText(self.original_text)
        self.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; padding: 2px; }"
            "QLabel:hover { background-color: #e0e0e0; border: 1px solid #999; }"
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse click to copy file path to clipboard."""
        if event.button() == Qt.LeftButton and self.file_path:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.file_path)

            # Show temporary visual feedback
            self.original_text = self.text()
            self.setText(f"Copied: {os.path.basename(self.file_path)}")
            self.setStyleSheet(
                "QLabel { background-color: #d4edda; border: 1px solid #c3e6cb; padding: 2px; }"
            )

            # Reset after 1.5 seconds
            self.reset_timer.start(1500)

        super().mousePressEvent(event)


def get_class_color(
    row: pd.Series,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Get color based on class information.

    Args:
        row: pandas Series containing class information

    Returns:
        Tuple containing (fill_color, border_color) as RGBA tuples
    """
    # Priority order: predicted_class > ClassLabel > ClassName
    class_value = None

    # Check predicted_class first
    if "FinalClass" in row and pd.notna(row["FinalClass"]):
        class_value = row["FinalClass"]
    elif "PredictedClass" in row and pd.notna(row["PredictedClass"]):
        class_value = row["PredictedClass"]
    elif "ClassLabel" in row and pd.notna(row["ClassLabel"]):
        class_value = row["ClassLabel"]
    elif "SourceClass" in row and pd.notna(row["SourceClass"]):
        class_value = str(row["SourceClass"]).lower()
    elif "LabelledClass" in row and pd.notna(row["LabelledClass"]):
        class_value = row["LabelledClass"]
    elif "header_footer_type" in row and pd.notna(row["header_footer_type"]):
        class_value = row["header_footer_type"]

    # Define color mapping based on class values

    alpha = 90
    alpha_cont = 25
    colors_map = {
        TextClass.PARAGRAPH: (0, 255, 0, alpha),
        TextClass.PARAGRAPH_CONT: (0, 255, 0, alpha_cont),
        TextClass.HEADING: (77, 240, 232, alpha),
        TextClass.HEADING_CONT: (77, 240, 232, alpha_cont),
        TextClass.TABLE: (214, 106, 222, alpha),
        TextClass.TOC: (106, 143, 222, alpha),
        TextClass.BULLET_LIST: (210, 242, 92, alpha),
        TextClass.BULLET_LIST_CONT: (210, 242, 92, alpha_cont),
        TextClass.ENUM_LIST: (252, 128, 45, alpha),
        TextClass.ENUM_LIST_CONT: (252, 128, 45, alpha_cont),
        TextClass.FOOTER: (125, 4, 4, alpha),
        TextClass.HEADER: (113, 4, 125, alpha),
    }

    class_color = colors_map.get(class_value, (255, 0, 0, 30))

    # make border color a little darker and have alpha 100
    border_color = (
        int(class_color[0] * 0.8),
        int(class_color[1] * 0.8),
        int(class_color[2] * 0.8),
        100,
    )
    return class_color, border_color


def get_annotation_text(row: pd.Series) -> str:
    """Get formatted annotation text for display.

    Args:
        row: pandas Series containing annotation data

    Returns:
        Formatted string containing annotation details
    """
    labelled_class_name = (
        TextClass(row.get("LabelledClass", None)).name
        if row.get("LabelledClass", None) is not None
        else "n/a"
    )
    row["LabelledClassName"] = labelled_class_name

    feature_str = "\n".join(sorted([f"__" + col + "__" for col in FEATURE_COLUMNS]))
    # Define the list of keys to display
    details_str = f"""
    PDF Features:
    __pdf_idx__
    __xml_idx__
    __FinalClassName__
    __PredictedClassName__
    __LabelledClassName__
    __ExtractedClassName__
    __header_footer_type__


    Computed Features:
    __left_indent__
    __right_space__
    __space_above__
    __space_below__
    __font_size__
    {feature_str}

    Predictions:
    __PredictionProbs__
    __MaxPredictionProb__

    Text:
    __XML_text__
    -------------------------------------
    __text__
    -------------------------------------
    __source_matched_text__
    -------------------------------------
    """

    output_str = ""
    lines = [line.strip() for line in details_str.split("\n") if line.strip()]
    for key in lines:
        if key.startswith("__") and key.endswith("__"):
            key = key[2:-2]
            value = row.get(key, "n/a")
            if "text" in key:
                line_str = f"\n{key}:\n {value}\n"
            else:
                line_str = f"{key:<35}: {value}"
        else:
            line_str = key
        output_str += line_str + "\n"
    return output_str


class AnnotationRectItem(QGraphicsRectItem):
    """Custom graphics item for clickable annotation rectangles."""

    annotation_text: str
    original_border_color: Tuple[int, int, int, int]
    original_fill_color: Tuple[int, int, int, int]

    def __init__(
        self,
        rect: QRectF,
        annotation_text: str,
        border_color: Tuple[int, int, int, int],
        fill_color: Tuple[int, int, int, int],
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(rect, parent)
        self.annotation_text = annotation_text
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable)

        # Store the original colors for restoring later
        self.original_border_color = border_color
        self.original_fill_color = fill_color

        # Set up visual appearance based on class colors
        self.setPen(QPen(QColor(*border_color), 4))
        self.setBrush(QBrush(QColor(*fill_color)))

    def hoverEnterEvent(self, event: QEvent) -> None:
        """Change appearance on hover.

        Args:
            event: Hover enter event
        """
        self.setPen(QPen(QColor(255, 255, 0, 200), 3))  # Yellow border
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QEvent) -> None:
        """Restore appearance when hover ends.

        Args:
            event: Hover leave event
        """
        self.setPen(
            QPen(QColor(*self.original_border_color), 2)
        )  # Back to original border color
        super().hoverLeaveEvent(event)


class PDFViewer(QGraphicsView):
    """Custom graphics view for displaying PDF pages with annotations."""

    annotation_clicked = pyqtSignal(str)  # Signal to emit annotation text when clicked

    scene: QGraphicsScene
    current_pixmap_item: Optional[QGraphicsPixmapItem]
    annotation_items: List[AnnotationRectItem]
    lines_rectangles_items: List[QGraphicsItem]
    selected_item_index: int
    zoom_factor: float
    min_zoom: float
    max_zoom: float

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

        # Current page data
        self.current_pixmap_item = None
        self.annotation_items = []
        self.lines_rectangles_items = []  # Store line and rectangle items
        self.selected_item_index = -1  # Track the currently selected item

        # Zoom settings
        self.zoom_factor = 0.5
        self.min_zoom = 0.1
        self.max_zoom = 5.0

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse click events to check for annotation items.

        Args:
            event: Mouse press event
        """
        if event.button() == Qt.LeftButton:
            # Get the click position in scene coordinates
            click_pos = self.mapToScene(event.pos())

            # Check if any annotation item was clicked
            for item in self.annotation_items:
                if item.rect().contains(click_pos):
                    self.annotation_clicked.emit(item.annotation_text)
                    break

        super().mousePressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle wheel events for zooming.

        Args:
            event: Wheel event
        """
        if event.modifiers() & Qt.ControlModifier:
            # Zoom with Ctrl+scroll
            zoom_in_factor = 1.25
            zoom_out_factor = 1 / zoom_in_factor

            if event.angleDelta().y() > 0:
                # Zoom in
                self.zoom_factor *= zoom_in_factor
            else:
                # Zoom out
                self.zoom_factor *= zoom_out_factor

            # Clamp zoom factor
            self.zoom_factor = max(self.min_zoom, min(self.max_zoom, self.zoom_factor))

            # Apply zoom by resetting transform and applying the total zoom factor
            self.resetTransform()
            self.scale(self.zoom_factor, self.zoom_factor)

            event.accept()
        else:
            # Normal scrolling when Ctrl is not pressed
            super().wheelEvent(event)

    def display_page(
        self, pixmap: QPixmap, annotations: Optional[pd.DataFrame] = None
    ) -> None:
        """Display a PDF page with optional annotations.

        Args:
            pixmap: QPixmap containing the page image
            annotations: Optional DataFrame containing annotation data
        """
        # Store current zoom factor
        current_zoom = self.zoom_factor

        # Clear previous items
        self.scene.clear()
        self.annotation_items.clear()
        self.lines_rectangles_items.clear()

        # Add the page image
        self.current_pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.current_pixmap_item)

        # Add annotation rectangles if provided
        if annotations is not None and not annotations.empty:
            self._add_annotations(annotations)

        # Only reset zoom if this is the first page load (zoom_factor is still 1.0)
        if current_zoom == 1.0:
            self.zoom_factor = 1.0
            self.resetTransform()
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            # Preserve zoom level for subsequent page loads
            self.zoom_factor = current_zoom
            self.resetTransform()
            self.scale(self.zoom_factor, self.zoom_factor)

    def _add_annotations(self, annotations: pd.DataFrame) -> None:
        """Add annotation rectangles to the scene.

        Args:
            annotations: DataFrame containing annotation data
        """
        scale = 2
        for _, row in annotations.iterrows():
            # Extract bounding box coordinates
            crop_x0 = row.get("crop_x0", 0) * scale
            crop_x1 = row.get("crop_x1", 0) * scale
            crop_y0 = row.get("crop_y0", 0) * scale
            crop_y1 = row.get("crop_y1", 0) * scale
            page_width = row.get("page_width", 0) * scale

            top_margin = 180
            row.get("major_font_size", 12)
            x0 = row.get("line_x0", 0) * scale + crop_x0
            x1 = row.get("line_x1", 0) * scale + crop_x0
            y0 = row.get("line_y0", 0) * scale + crop_y0
            y1 = row.get("line_y1", 0) * scale + crop_y0
            # Create rectangle
            rect = QRectF(x0, y0, x1 - x0, y1 - y0)

            # draw crop rectangle
            crop_rect = QRectF(crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0)
            self.scene.addRect(crop_rect, QPen(QColor(0, 0, 255, 180), 2))

            # Get annotation text
            annotation_text = get_annotation_text(row)

            # Get class-based colors
            fill_color, border_color = get_class_color(row)

            # Create clickable annotation item with class-based colors
            annotation_item = AnnotationRectItem(
                rect, annotation_text, border_color, fill_color
            )
            self.scene.addItem(annotation_item)
            self.annotation_items.append(annotation_item)

        if "top_margin" in annotations.columns:
            top_margin = annotations["top_margin"].values[0]
            top_margin = top_margin * scale + crop_y0
            # draw a horizontal line at the top margin
            line = QLineF(0, top_margin, page_width, top_margin)
            line_item = self.scene.addLine(line, QPen(QColor(0, 0, 255, 180), 2))

        if "bottom_margin" in annotations.columns:
            bottom_margin = annotations["bottom_margin"].values[0]
            bottom_margin = bottom_margin * scale + crop_y0
            # draw a horizontal line at the bottom margin
            line = QLineF(0, bottom_margin, page_width, bottom_margin)
            line_item = self.scene.addLine(line, QPen(QColor(0, 0, 255, 180), 2))

        # get table coordinates, horizontal lines, and vertical lines
        def get_lines_list(annotations: pd.DataFrame, key: str) -> list:
            lines = list(annotations.get(key, []))
            lines = [t for t in lines if t != "[]"]
            lines = list(chain.from_iterable([eval(t) for t in lines]))
            return lines

        table_coordinates = get_lines_list(annotations, "table_coordinates")
        horizontal_lines = get_lines_list(annotations, "horizontal_lines")
        vertical_lines = get_lines_list(annotations, "vertical_lines")

        for table_coordinate in table_coordinates:
            rect = QRectF(
                table_coordinate[0] * scale + crop_x0,
                table_coordinate[1] * scale + crop_y0,
                (table_coordinate[2] - table_coordinate[0]) * scale + crop_x0,
                (table_coordinate[3] - table_coordinate[1]) * scale + crop_y0,
            )
            rect_item = self.scene.addRect(rect, QPen(QColor(0, 0, 255, 180), 2))
            self.lines_rectangles_items.append(rect_item)

        for horizontal_line in horizontal_lines:
            line = QLineF(
                horizontal_line[0] * scale + crop_x0,
                horizontal_line[1] * scale + crop_y0,
                horizontal_line[2] * scale + crop_x0,
                horizontal_line[3] * scale + crop_y0,
            )
            line_item = self.scene.addLine(line, QPen(QColor(0, 0, 255, 180), 2))
            self.lines_rectangles_items.append(line_item)

        for vertical_line in vertical_lines:
            line = QLineF(
                vertical_line[0] * scale + crop_x0,
                vertical_line[1] * scale + crop_y0,
                vertical_line[2] * scale + crop_x0,
                vertical_line[3] * scale + crop_y0,
            )
            line_item = self.scene.addLine(line, QPen(QColor(0, 0, 255, 180), 2))
            self.lines_rectangles_items.append(line_item)

    def highlight_item(self, item_index: int) -> None:
        """Highlight a specific annotation item in blue.

        Args:
            item_index: Index of the item to highlight
        """
        # Reset all items to their original class-based appearance
        for item in self.annotation_items:
            item.setPen(QPen(QColor(*item.original_border_color), 2))
            item.setBrush(QBrush(QColor(*item.original_fill_color)))

        # Highlight the selected item in blue
        if 0 <= item_index < len(self.annotation_items):
            self.selected_item_index = item_index
            selected_item = self.annotation_items[item_index]
            selected_item.setPen(QPen(QColor(0, 0, 255, 200), 5))  # Blue border
            selected_item.setBrush(QBrush(QColor(*selected_item.original_fill_color)))

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle resize events to maintain proper scaling.

        Args:
            event: Resize event
        """
        super().resizeEvent(event)
        if self.current_pixmap_item:
            # Only fit to view if we're at the default zoom level
            if self.zoom_factor == 1.0:
                self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            else:
                # Preserve zoom level during resize
                self.resetTransform()
                self.scale(self.zoom_factor, self.zoom_factor)

    def toggle_lines_rectangles_visibility(self, visible: bool) -> None:
        """Toggle the visibility of lines and rectangles.

        Args:
            visible: Whether to show lines and rectangles
        """
        for item in self.lines_rectangles_items:
            item.setVisible(visible)

    def has_lines_rectangles(self) -> bool:
        """Check if the current page has any lines or rectangles.

        Returns:
            True if the page has lines or rectangles, False otherwise
        """
        return len(self.lines_rectangles_items) > 0


class FeatureVisualizer(QMainWindow):
    """Main application window for PDF feature visualization."""

    pdf_document: Optional[fitz.Document]
    current_page: int
    total_pages: int
    annotations_df: Optional[pd.DataFrame]
    updating_spinbox: bool
    settings: QSettings
    show_lines_rectangles: bool
    lines_rectangles_items: List[QGraphicsItem]
    default_pdf_path: str
    default_csv_path: str
    pdf_viewer: PDFViewer
    page_spinbox: QSpinBox
    total_pages_label: QLabel
    line_spinbox: QSpinBox
    total_lines_label: QLabel
    show_lines_rectangles_checkbox: QCheckBox
    annotation_text: QTextEdit
    load_pdf_button: QPushButton
    annotations_combobox: QComboBox
    refresh_button: QPushButton

    def __init__(self) -> None:
        super().__init__()
        self.pdf_document = None
        self.current_page = 0
        self.total_pages = 0
        self.annotations_df = None
        self.updating_spinbox = False  # Flag to prevent recursive calls

        # Initialize QSettings
        self.settings = QSettings("Enginius", "PDFFeatureVisualizer")

        # Lines and rectangles visibility
        self.show_lines_rectangles = True
        self.lines_rectangles_items = []  # Store line and rectangle items for toggling

        # Default file paths
        self.default_pdf_path = (
            r"C:\Users\r123m\Documents\enginius\data\xml_cfr\CFR-2024-title14-vol2.pdf"
        )
        self.default_csv_path = r"C:\Users\r123m\Documents\enginius\data\xml_cfr\labelled\CFR-2024-title14-vol2_labelled.csv"

        self.init_ui()

        # Load saved state or default files
        self.load_saved_state()

    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("PDF Feature Visualizer")
        self.setGeometry(100, 100, 1200, 800)

        # Enable keyboard focus for the window
        self.setFocusPolicy(Qt.StrongFocus)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: PDF viewer
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # PDF viewer
        self.pdf_viewer = PDFViewer()
        left_layout.addWidget(self.pdf_viewer)

        # Navigation controls
        nav_layout = QHBoxLayout()

        # Page number spinbox
        page_label = QLabel("Page:")
        # center the text vertically, and right align the text
        page_label.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        nav_layout.addWidget(page_label)

        self.page_spinbox = QSpinBox()
        self.page_spinbox.setMinimum(1)
        self.page_spinbox.setMaximum(1)
        self.page_spinbox.valueChanged.connect(self.go_to_page)
        nav_layout.addWidget(self.page_spinbox)

        total_label = QLabel("of 0")
        nav_layout.addWidget(total_label)
        self.total_pages_label = total_label

        # Line index spinbox
        line_label = QLabel("Line:")
        line_label.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        nav_layout.addWidget(line_label)

        self.line_spinbox = QSpinBox()
        self.line_spinbox.setMinimum(0)
        self.line_spinbox.setMaximum(0)
        self.line_spinbox.valueChanged.connect(self.go_to_line)
        nav_layout.addWidget(self.line_spinbox)

        line_total_label = QLabel("of 0")
        nav_layout.addWidget(line_total_label)
        self.total_lines_label = line_total_label

        # Add some spacing
        nav_layout.addStretch()

        # Lines and rectangles toggle
        self.show_lines_rectangles_checkbox = QCheckBox("Show Lines & Rectangles")
        self.show_lines_rectangles_checkbox.setChecked(self.show_lines_rectangles)
        self.show_lines_rectangles_checkbox.toggled.connect(
            self.toggle_lines_rectangles
        )
        nav_layout.addWidget(self.show_lines_rectangles_checkbox)

        left_layout.addLayout(nav_layout)

        # Right panel: Annotation details
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Status information at the top
        status_layout = QVBoxLayout()

        # PDF status
        pdf_status_layout = QHBoxLayout()
        pdf_status_label = QLabel("PDF:")
        pdf_status_label.setFont(QFont("Arial", 10, QFont.Bold))
        pdf_status_layout.addWidget(pdf_status_label)

        self.pdf_status_text = ClickableStatusLabel("No PDF loaded")
        pdf_status_layout.addWidget(self.pdf_status_text)
        status_layout.addLayout(pdf_status_layout)

        # Annotations status
        annotations_status_layout = QHBoxLayout()
        annotations_status_label = QLabel("Annotations:")
        annotations_status_label.setFont(QFont("Arial", 10, QFont.Bold))
        annotations_status_layout.addWidget(annotations_status_label)

        self.annotations_status_text = ClickableStatusLabel("No annotations loaded")
        annotations_status_layout.addWidget(self.annotations_status_text)
        status_layout.addLayout(annotations_status_layout)

        right_layout.addLayout(status_layout)

        # Annotation text display
        annotation_label = QLabel("Annotation Details:")
        annotation_label.setFont(QFont("Arial", 12, QFont.Bold))
        right_layout.addWidget(annotation_label)

        self.annotation_text = QTextEdit()
        self.annotation_text.setReadOnly(True)
        # Enable HTML support for rich text formatting
        # self.annotation_text.setAcceptRichText(True)
        # Set default font for HTML content
        font = QFont("Courier New", 10)
        self.annotation_text.setFont(font)
        right_layout.addWidget(self.annotation_text)

        # File controls
        file_layout = QHBoxLayout()

        self.load_pdf_button = QPushButton("Load PDF")
        self.load_pdf_button.clicked.connect(self.load_pdf)
        file_layout.addWidget(self.load_pdf_button)

        # Annotations combobox
        annotations_label = QLabel("Annotations:")
        file_layout.addWidget(annotations_label)

        self.annotations_combobox = QComboBox()
        self.annotations_combobox.setMinimumWidth(200)
        self.annotations_combobox.currentTextChanged.connect(
            self.on_annotation_selected
        )
        file_layout.addWidget(self.annotations_combobox)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_current_files)
        file_layout.addWidget(self.refresh_button)

        right_layout.addLayout(file_layout)
        # tell right_layout to give most space to the 2nd widget

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 400])  # Initial split sizes

        # Connect annotation click signals
        self._connect_annotation_signals()

        # Initialize UI state
        self.update_navigation_controls()

        # Set focus to the window so it can receive keyboard events
        self.setFocus()

        # Restore window geometry if available
        saved_geometry = self.settings.value("geometry")
        if saved_geometry:
            self.restoreGeometry(saved_geometry)

    def showEvent(self, event: QShowEvent) -> None:
        """Handle window show event to ensure keyboard focus.

        Args:
            event: Show event
        """
        super().showEvent(event)
        self.setFocus()

    def load_saved_state(self) -> None:
        """Load saved state or default files."""
        # Try to load saved PDF path
        saved_pdf_path = self.settings.value("pdf_path", "")
        saved_csv_path = self.settings.value("csv_path", "")

        # Load PDF (saved path or default)
        pdf_loaded = False
        if saved_pdf_path and os.path.exists(saved_pdf_path):
            try:
                self.pdf_document = fitz.open(saved_pdf_path)
                self.total_pages = len(self.pdf_document)
                self.current_page = self.settings.value("current_page", 0, type=int)
                self.setWindowTitle(
                    f"PDF Feature Visualizer - {os.path.basename(saved_pdf_path)}"
                )
                print(f"Loaded saved PDF: {saved_pdf_path}")
                pdf_loaded = True
                # Populate annotations combobox
                self.populate_annotations_combobox(saved_pdf_path)
            except Exception as e:
                print(f"Failed to load saved PDF: {e}")

        # If saved PDF failed, try default
        if not pdf_loaded and os.path.exists(self.default_pdf_path):
            try:
                self.pdf_document = fitz.open(self.default_pdf_path)
                self.total_pages = len(self.pdf_document)
                self.current_page = 0
                self.setWindowTitle(
                    f"PDF Feature Visualizer - {os.path.basename(self.default_pdf_path)}"
                )
                print(f"Loaded default PDF: {self.default_pdf_path}")
                pdf_loaded = True
                # Populate annotations combobox
                self.populate_annotations_combobox(self.default_pdf_path)
            except Exception as e:
                print(f"Failed to load default PDF: {e}")

        # Load CSV (saved path or default)
        csv_loaded = False
        if saved_csv_path and os.path.exists(saved_csv_path):
            try:
                self.annotations_df = pd.read_csv(saved_csv_path)
                print(f"Loaded saved CSV: {saved_csv_path}")
                csv_loaded = True
            except Exception as e:
                print(f"Failed to load saved CSV: {e}")

        # If saved CSV failed, try default
        if not csv_loaded and os.path.exists(self.default_csv_path):
            try:
                self.annotations_df = pd.read_csv(self.default_csv_path)
                print(f"Loaded default CSV: {self.default_csv_path}")
                csv_loaded = True
            except Exception as e:
                print(f"Failed to load default CSV: {e}")

                # Display the current page if PDF was loaded
        if self.pdf_document:
            self.display_current_page()
            self.update_navigation_controls()

            # Restore line selection if annotations are loaded
            if self.annotations_df is not None and not self.annotations_df.empty:
                saved_line = self.settings.value("current_line", 0, type=int)
                page_annotations = self._get_current_page_annotations()
                if page_annotations is not None and not page_annotations.empty:
                    if 0 <= saved_line < len(page_annotations):
                        self.updating_spinbox = True
                        self.line_spinbox.setValue(saved_line)
                        self.updating_spinbox = False
                        self.pdf_viewer.highlight_item(saved_line)
                        # Update annotation text
                        row = page_annotations.iloc[saved_line]
                        annotation_text = get_annotation_text(row)
                        self.annotation_text.setText(annotation_text)

            # Restore combobox selection to match the loaded annotations
            if self.annotations_df is not None:
                try:
                    # Try to get the CSV path from metadata first
                    csv_path = self.annotations_df._metadata.get("file_path", "")
                    if not csv_path:
                        # If no metadata, try to get it from settings
                        csv_path = self.settings.value("csv_path", "")

                    if csv_path and os.path.exists(csv_path):
                        # Find the index of the loaded file in the combobox
                        for i in range(self.annotations_combobox.count()):
                            if self.annotations_combobox.itemData(i) == csv_path:
                                self.annotations_combobox.setCurrentIndex(i)
                                print(
                                    f"Restored combobox selection to: {self.annotations_combobox.currentText()}"
                                )
                                break
                except Exception as e:
                    print(f"Could not restore combobox selection: {e}")

        # Update status display after loading saved state
        self.update_status_display()

        # Load saved toggle state
        self.show_lines_rectangles = self.settings.value(
            "show_lines_rectangles", True, type=bool
        )
        if hasattr(self, "show_lines_rectangles_checkbox"):
            self.show_lines_rectangles_checkbox.setChecked(self.show_lines_rectangles)

    def save_state(self) -> None:
        """Save current application state."""
        if self.pdf_document:
            # Save PDF path
            pdf_path = self.pdf_document.name
            if pdf_path:
                self.settings.setValue("pdf_path", pdf_path)

            # Save current page
            self.settings.setValue("current_page", self.current_page)

        # Save current annotation file selection
        if self.annotations_combobox.currentData():
            current_csv_path = self.annotations_combobox.currentData()
            if current_csv_path and current_csv_path != "No annotations":
                self.settings.setValue("csv_path", current_csv_path)
        elif self.annotations_df is not None:
            # Save CSV path (if it has a path attribute)
            try:
                csv_path = self.annotations_df._metadata.get("file_path", "")
                if csv_path:
                    self.settings.setValue("csv_path", csv_path)
            except:
                pass

        # Save current line
        current_line = self.line_spinbox.value()
        self.settings.setValue("current_line", current_line)

        # Save lines and rectangles visibility state
        self.settings.setValue("show_lines_rectangles", self.show_lines_rectangles)

        # Save window geometry
        self.settings.setValue("geometry", self.saveGeometry())

    def update_status_display(self) -> None:
        """Update the status display text boxes."""
        # Update PDF status
        if self.pdf_document and self.pdf_document.name:
            pdf_name = os.path.basename(self.pdf_document.name)
            self.pdf_status_text.setText(f"Loaded: {pdf_name}")
            self.pdf_status_text.set_file_path(self.pdf_document.name)
        else:
            self.pdf_status_text.setText("No PDF loaded")
            self.pdf_status_text.set_file_path("")

        # Update annotations status
        if self.annotations_df is not None and not self.annotations_df.empty:
            try:
                csv_path = self.annotations_df._metadata.get("file_path", "")
                if csv_path:
                    csv_name = os.path.basename(csv_path)
                    self.annotations_status_text.setText(f"Loaded: {csv_name}")
                    self.annotations_status_text.set_file_path(csv_path)
                else:
                    self.annotations_status_text.setText("Annotations loaded (no path)")
                    self.annotations_status_text.set_file_path("")
            except:
                self.annotations_status_text.setText("Annotations loaded (no path)")
                self.annotations_status_text.set_file_path("")
        else:
            self.annotations_status_text.setText("No annotations loaded")
            self.annotations_status_text.set_file_path("")

    def load_default_files(self) -> None:
        """Load default PDF and CSV files if they exist."""
        # Load default PDF
        if os.path.exists(self.default_pdf_path):
            try:
                self.pdf_document = fitz.open(self.default_pdf_path)
                self.total_pages = len(self.pdf_document)
                self.current_page = 0
                self.setWindowTitle(
                    f"PDF Feature Visualizer - {os.path.basename(self.default_pdf_path)}"
                )
                print(f"Loaded default PDF: {self.default_pdf_path}")
            except Exception as e:
                print(f"Failed to load default PDF: {e}")

        # Load default CSV
        if os.path.exists(self.default_csv_path):
            try:
                self.annotations_df = pd.read_csv(self.default_csv_path)
                print(f"Loaded default CSV: {self.default_csv_path}")
            except Exception as e:
                print(f"Failed to load default CSV: {e}")

        # Display the first page if PDF was loaded
        if self.pdf_document:
            self.display_current_page()
            self.update_navigation_controls()

    def _connect_annotation_signals(self) -> None:
        """Connect annotation click signals to the text display."""
        # Connect the PDF viewer's annotation clicked signal
        self.pdf_viewer.annotation_clicked.connect(self.on_annotation_clicked)

    def load_pdf(self) -> None:
        """Load a PDF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open PDF File", "", "PDF Files (*.pdf)"
        )

        if file_path:
            try:
                self.pdf_document = fitz.open(file_path)
                self.total_pages = len(self.pdf_document)
                self.current_page = 0
                self.display_current_page()
                self.update_navigation_controls()

                self.setWindowTitle(
                    f"PDF Feature Visualizer - {os.path.basename(file_path)}"
                )

                # Populate annotations combobox with matching CSV files
                self.populate_annotations_combobox(file_path)

                # Update status display
                self.update_status_display()

                # Save state after loading PDF
                self.save_state()
            except Exception as e:
                print(f"Failed to load PDF: {e}")
                self.update_status_display()

    def find_matching_csv_files(self, pdf_path: str) -> List[Tuple[str, str]]:
        """Find CSV files with the same stem name as the PDF in subdirectories.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of tuples containing (subdirectory_name, csv_file_path)
        """
        pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_parent_dir = os.path.dirname(pdf_path)

        matching_files = []

        try:
            # Look through immediate subdirectories of the PDF's parent directory
            for item in os.listdir(pdf_parent_dir):
                subdir_path = os.path.join(pdf_parent_dir, item)

                # Check if it's a directory
                if os.path.isdir(subdir_path):
                    # Look for CSV files in this subdirectory
                    for file in os.listdir(subdir_path):
                        if file == f"{pdf_stem}.csv":
                            csv_path = os.path.join(subdir_path, file)
                            matching_files.append((item, csv_path))
        except Exception as e:
            print(f"Error searching for matching CSV files: {e}")

        return matching_files

    def populate_annotations_combobox(self, pdf_path: str) -> None:
        """Populate the annotations combobox with matching CSV files.

        Args:
            pdf_path: Path to the PDF file
        """
        # Clear existing items
        self.annotations_combobox.clear()

        # Add a default "No annotations" option
        self.annotations_combobox.addItem("No annotations")

        # Find matching CSV files
        matching_files = self.find_matching_csv_files(pdf_path)

        if matching_files:
            # Add each matching file, using subdirectory name as display text
            for subdir_name, csv_path in matching_files:
                self.annotations_combobox.addItem(subdir_name, csv_path)

            print(f"Found {len(matching_files)} matching annotation files:")
            for subdir_name, csv_path in matching_files:
                print(f"  {subdir_name}: {csv_path}")
        else:
            print("No matching annotation files found")

    def on_annotation_selected(self, selected_text: str) -> None:
        """Handle annotation selection from combobox.

        Args:
            selected_text: The selected text (subdirectory name)
        """
        if selected_text == "No annotations":
            # Clear annotations
            self.annotations_df = None
            self.display_current_page()
            # Update status display
            self.update_status_display()
            # Save state to remember the "No annotations" selection
            self.save_state()
            return

        # Get the CSV file path from the combobox data
        csv_path = self.annotations_combobox.currentData()
        if csv_path and os.path.exists(csv_path):
            try:
                self.annotations_df = pd.read_csv(csv_path)
                # Store the file path in metadata for later saving
                self.annotations_df._metadata = {"file_path": csv_path}
                self.display_current_page()  # Refresh display with annotations

                # Update status display
                self.update_status_display()

                # Save state after loading annotations
                self.save_state()
            except Exception as e:
                print(f"Failed to load annotations: {e}")
                self.update_status_display()
        else:
            print(f"Annotation file not found: {csv_path}")
            self.update_status_display()

    def refresh_current_files(self) -> None:
        """Refresh the current PDF and annotations by reloading them."""
        if not self.pdf_document:
            QMessageBox.warning(self, "No PDF Loaded", "Please load a PDF file first.")
            return

        try:
            # Get current PDF path
            pdf_path = self.pdf_document.name
            if not pdf_path:
                QMessageBox.warning(
                    self, "No PDF Path", "Cannot refresh: PDF path not available."
                )
                return

            # Remember the currently selected annotation file before refreshing
            current_csv_path = None
            if self.annotations_df is not None:
                try:
                    # Try to get the current CSV path from metadata first
                    current_csv_path = self.annotations_df._metadata.get(
                        "file_path", ""
                    )
                except:
                    pass

                # If no metadata path, try to get it from settings
                if not current_csv_path:
                    current_csv_path = self.settings.value("csv_path", "")

            # Reload PDF
            self.pdf_document.close()
            self.pdf_document = fitz.open(pdf_path)
            self.total_pages = len(self.pdf_document)

            # Repopulate annotations combobox
            self.populate_annotations_combobox(pdf_path)

            # Restore the previously selected annotation file if it still exists
            if current_csv_path and os.path.exists(current_csv_path):
                # Find the index of the previously selected file in the combobox
                for i in range(self.annotations_combobox.count()):
                    if self.annotations_combobox.itemData(i) == current_csv_path:
                        self.annotations_combobox.setCurrentIndex(i)
                        break

                # Reload the annotations
                try:
                    self.annotations_df = pd.read_csv(current_csv_path)
                    self.annotations_df._metadata = {"file_path": current_csv_path}
                    print(f"Refreshed annotations from: {current_csv_path}")
                except Exception as e:
                    print(f"Error refreshing annotations: {e}")
            else:
                # If the previous file doesn't exist, try to load from settings
                saved_csv_path = self.settings.value("csv_path", "")
                if saved_csv_path and os.path.exists(saved_csv_path):
                    try:
                        self.annotations_df = pd.read_csv(saved_csv_path)
                        self.annotations_df._metadata = {"file_path": saved_csv_path}
                        print(
                            f"Refreshed annotations from saved path: {saved_csv_path}"
                        )

                        # Update combobox selection to match
                        for i in range(self.annotations_combobox.count()):
                            if self.annotations_combobox.itemData(i) == saved_csv_path:
                                self.annotations_combobox.setCurrentIndex(i)
                                break
                    except Exception as e:
                        print(f"Error refreshing annotations from saved path: {e}")

            # Refresh the display
            self.display_current_page()
            self.update_navigation_controls()

            # Update status display
            self.update_status_display()

        except Exception as e:
            print(f"Refresh error: {e}")
            self.update_status_display()

    def display_current_page(self) -> None:
        """Display the current page with annotations."""
        if not self.pdf_document:
            return

        # Get the current page
        page = self.pdf_document[self.current_page]

        # Render the page to a pixmap
        mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better quality
        pix = page.get_pixmap(matrix=mat)

        # Convert to QPixmap
        img_data = pix.tobytes("ppm")
        pixmap = QPixmap()
        pixmap.loadFromData(img_data)

        # Get annotations for current page if available
        page_annotations = None
        if self.annotations_df is not None and not self.annotations_df.empty:
            # Filter annotations for current page
            if "PageNumber" in self.annotations_df.columns:
                page_annotations = self.annotations_df[
                    self.annotations_df["PageNumber"] == self.current_page
                ]
            else:
                # If no page column, use all annotations
                page_annotations = self.annotations_df

        # Display the page with annotations
        self.pdf_viewer.display_page(pixmap, page_annotations)

        # Apply the current toggle state
        self.pdf_viewer.toggle_lines_rectangles_visibility(self.show_lines_rectangles)

        # Update toggle button state
        self.update_toggle_state()

    def on_annotation_clicked(self, text: str) -> None:
        """Handle annotation click events.

        Args:
            text: Annotation text that was clicked
        """
        # Store current scroll position
        scrollbar = self.annotation_text.verticalScrollBar()
        current_position = scrollbar.value()

        # Update the text
        self.annotation_text.setText(text)

        # Restore scroll position
        scrollbar.setValue(current_position)

        # Find which annotation was clicked and update the line spinbox
        page_annotations = self._get_current_page_annotations()
        if page_annotations is not None and not page_annotations.empty:
            # Find the annotation that matches the clicked text
            for i, (_, row) in enumerate(page_annotations.iterrows()):
                if get_annotation_text(row) == text:
                    # Update the line spinbox to reflect the clicked item
                    self.updating_spinbox = True
                    self.line_spinbox.setValue(i)
                    self.updating_spinbox = False

                    # Highlight the clicked item
                    self.pdf_viewer.highlight_item(i)
                    break

    def previous_page(self) -> None:
        """Navigate to the previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self.display_current_page()
            self.update_navigation_controls()

    def next_page(self) -> None:
        """Navigate to the next page."""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.display_current_page()
            self.update_navigation_controls()

    def go_to_page(self, page_number: int) -> None:
        """Navigate to a specific page number.

        Args:
            page_number: Page number to navigate to (1-based)
        """
        if self.pdf_document is not None and not self.updating_spinbox:
            # Convert from 1-based to 0-based indexing
            target_page = page_number - 1
            if 0 <= target_page < self.total_pages:
                self.current_page = target_page
                self.display_current_page()
                self.update_navigation_controls()

                # Highlight the first annotation if any exist
                page_annotations = self._get_current_page_annotations()
                if page_annotations is not None and not page_annotations.empty:
                    self.go_to_line(0)

                # Save state after page change
                self.save_state()

    def go_to_line(self, line_index: int) -> None:
        """Navigate to a specific line index on the current page.

        Args:
            line_index: Line index to navigate to (0-based)
        """
        if self.pdf_document is not None and not self.updating_spinbox:
            # Get current page annotations
            page_annotations = self._get_current_page_annotations()
            if page_annotations is not None and not page_annotations.empty:
                if 0 <= line_index < len(page_annotations):
                    # Get the annotation text for the selected line
                    row = page_annotations.iloc[line_index]
                    annotation_text = get_annotation_text(row)
                    self.annotation_text.setText(annotation_text)

                    # Highlight the selected annotation in blue
                    self.pdf_viewer.highlight_item(line_index)

                    # Save state after line change
                    self.save_state()

    def _get_current_page_annotations(self) -> Optional[pd.DataFrame]:
        """Get annotations for the current page.

        Returns:
            DataFrame containing annotations for current page, or None if no annotations
        """
        if self.annotations_df is not None and not self.annotations_df.empty:
            # Filter annotations for current page
            if "PageNumber" in self.annotations_df.columns:
                return self.annotations_df[
                    self.annotations_df["PageNumber"] == self.current_page
                ]
            else:
                # If no page column, use all annotations
                return self.annotations_df
        return None

    def update_navigation_controls(self) -> None:
        """Update the state of navigation controls."""
        has_pdf = self.pdf_document is not None

        if has_pdf:
            # Update page spinbox range and value
            self.updating_spinbox = True
            self.page_spinbox.setMaximum(self.total_pages)
            self.page_spinbox.setValue(self.current_page + 1)  # Convert to 1-based
            self.updating_spinbox = False
            self.total_pages_label.setText(f"of {self.total_pages}")

            # Update line spinbox range and value
            page_annotations = self._get_current_page_annotations()
            if page_annotations is not None and not page_annotations.empty:
                total_lines = len(page_annotations)
                self.updating_spinbox = True
                self.line_spinbox.setMaximum(total_lines - 1)  # 0-based indexing
                self.line_spinbox.setValue(0)  # Reset to first line
                self.updating_spinbox = False
                self.total_lines_label.setText(f"of {total_lines - 1}")
            else:
                self.line_spinbox.setMaximum(0)
                self.line_spinbox.setValue(0)
                self.total_lines_label.setText("of 0")
        else:
            # Reset controls when no PDF is loaded
            self.page_spinbox.setMaximum(1)
            self.page_spinbox.setValue(1)
            self.total_pages_label.setText("of 0")
            self.line_spinbox.setMaximum(0)
            self.line_spinbox.setValue(0)
            self.total_lines_label.setText("of 0")

        # Update toggle button state
        self.update_toggle_state()

    def toggle_lines_rectangles(self, checked: bool) -> None:
        """Toggle the visibility of lines and rectangles.

        Args:
            checked: Whether lines and rectangles should be visible
        """
        self.show_lines_rectangles = checked
        self.pdf_viewer.toggle_lines_rectangles_visibility(checked)

    def update_toggle_state(self) -> None:
        """Update the toggle button state based on whether the current page has lines and rectangles."""
        has_lines_rectangles = self.pdf_viewer.has_lines_rectangles()
        self.show_lines_rectangles_checkbox.setEnabled(has_lines_rectangles)

        # If there are no lines/rectangles, uncheck the toggle
        if not has_lines_rectangles:
            self.show_lines_rectangles_checkbox.setChecked(False)
            self.show_lines_rectangles = False
        else:
            # Restore the saved state if there are lines/rectangles
            self.show_lines_rectangles_checkbox.setChecked(self.show_lines_rectangles)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle keyboard shortcuts for navigation.

        Args:
            event: Key press event
        """
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Left:
                # Ctrl + Left Arrow: Previous page
                if self.current_page > 0:
                    self.current_page -= 1
                    self.display_current_page()
                    self.update_navigation_controls()
                    # Highlight the first annotation if any exist
                    page_annotations = self._get_current_page_annotations()
                    if page_annotations is not None and not page_annotations.empty:
                        self.pdf_viewer.highlight_item(0)
                    # Save state after page change
                    self.save_state()
                event.accept()
                return
            elif event.key() == Qt.Key_Right:
                # Ctrl + Right Arrow: Next page
                if self.current_page < self.total_pages - 1:
                    self.current_page += 1
                    self.display_current_page()
                    self.update_navigation_controls()
                    # Highlight the first annotation if any exist
                    page_annotations = self._get_current_page_annotations()
                    if page_annotations is not None and not page_annotations.empty:
                        self.pdf_viewer.highlight_item(0)
                    # Save state after page change
                    self.save_state()
                event.accept()
                return
            elif event.key() == Qt.Key_Up:
                # Ctrl + Up Arrow: Previous line
                current_line = self.line_spinbox.value()
                if current_line > 0:
                    self.line_spinbox.setValue(current_line - 1)
                    # Save state after line change
                    self.save_state()
                event.accept()
                return
            elif event.key() == Qt.Key_Down:
                # Ctrl + Down Arrow: Next line
                current_line = self.line_spinbox.value()
                max_line = self.line_spinbox.maximum()
                if current_line < max_line:
                    self.line_spinbox.setValue(current_line + 1)
                    # Save state after line change
                    self.save_state()
                event.accept()
                return

        # Call parent class for other key events
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle application close event.

        Args:
            event: Close event
        """
        # Save state before closing
        self.save_state()

        if self.pdf_document:
            self.pdf_document.close()
        event.accept()


def main() -> None:
    """Main application entry point."""
    # Setup Qt environment for remote execution
    try:
        app = QApplication(sys.argv)

        # Set application style
        app.setStyle("Fusion")

        # Create and show the main window
        window = FeatureVisualizer()
        window.show()

        # Start the application
        sys.exit(app.exec_())

    except Exception as e:
        print(f"Failed to start application: {e}")
        print("Available Qt platforms:")
        print("  - Try running with: export QT_QPA_PLATFORM=offscreen")
        print("  - Or use X11 forwarding: ssh -X user@remote")
        print("  - Or install xvfb: sudo apt-get install xvfb")
        sys.exit(1)


if __name__ == "__main__":
    main()
