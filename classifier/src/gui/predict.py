
import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# PROJECT PATH
# ============================================================

SRC_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(SRC_DIR)
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from data_preparation import (
    load_dataset,
    remove_duplicates,
    stratified_split
)

from naive_bayes.naive_bayes import (
    MultinomialNaiveBayes
)

from evaluation.accuracy import accuracy
from evaluation.report import macro_f1


# ============================================================
# CUSTOMER SUPPORT ROUTER
# ============================================================

class CustomerSupportRouter:
    """
    Handles model training, evaluation, and prediction.

    The GUI communicates with this class instead of
    directly managing the Naive Bayes model.
    """

    def __init__(self):

        # ----------------------------------------------------
        # 1. Load dataset
        # ----------------------------------------------------

        data = load_dataset()

        # ----------------------------------------------------
        # 2. Remove duplicate messages
        # ----------------------------------------------------

        data = remove_duplicates(
            data
        )

        # ----------------------------------------------------
        # 3. Split dataset
        # ----------------------------------------------------

        train_data, test_data = (
            stratified_split(data)
        )

        # ----------------------------------------------------
        # 4. Create classifier
        # ----------------------------------------------------

        self.classifier = (
            MultinomialNaiveBayes()
        )

        # ----------------------------------------------------
        # 5. Train model
        # ----------------------------------------------------

        self.classifier.fit(
            train_data["instruction"].tolist(),
            train_data["intent"].tolist()
        )

        # ----------------------------------------------------
        # 6. Predict test data
        # ----------------------------------------------------

        test_texts = (
            test_data["instruction"].tolist()
        )

        test_labels = (
            test_data["intent"].tolist()
        )

        test_predictions = (
            self.classifier.predict(
                test_texts
            )
        )

        # ----------------------------------------------------
        # 7. Evaluate model
        # ----------------------------------------------------

        self.test_accuracy = accuracy(
            test_labels,
            test_predictions
        )

        self.test_f1 = macro_f1(
            test_labels,
            test_predictions
        )

        # ----------------------------------------------------
        # 8. Store dataset information
        # ----------------------------------------------------

        self.training_size = len(
            train_data
        )

        self.test_size = len(
            test_data
        )

        self.number_of_intents = len(
            self.classifier.classes
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, text):
        """
        Predict the intent of one customer request.
        """

        if not text.strip():

            raise ValueError(
                "Please enter a customer request."
            )

        return self.classifier.predict(
            [text]
        )[0]


# ============================================================
# GRAPHICAL USER INTERFACE
# ============================================================

class CustomerSupportGUI:
    """
    Tkinter interface for the customer-support router.
    """

    def __init__(
        self,
        root,
        router
    ):

        self.root = root

        self.router = router

        # ----------------------------------------------------
        # Window configuration
        # ----------------------------------------------------

        self.root.title(
            "Customer Support Ticket Router"
        )

        self.root.geometry(
            "750x700"
        )

        self.root.minsize(
            500,
            550
        )

        # ----------------------------------------------------
        # Create interface
        # ----------------------------------------------------

        self.create_styles()

        self.create_interface()

        # ----------------------------------------------------
        # Responsive layout
        # ----------------------------------------------------

        self.root.bind(
            "<Configure>",
            self.on_resize
        )

    # ========================================================
    # STYLES
    # ========================================================

    def create_styles(self):

        style = ttk.Style()

        style.configure(
            "Title.TLabel",
            font=(
                "Segoe UI",
                22,
                "bold"
            )
        )

        style.configure(
            "Subtitle.TLabel",
            font=(
                "Segoe UI",
                10
            )
        )

        style.configure(
            "Classify.TButton",
            font=(
                "Segoe UI",
                11,
                "bold"
            ),
            padding=10
        )

        style.configure(
            "Metric.TLabel",
            font=(
                "Segoe UI",
                12,
                "bold"
            )
        )

    # ========================================================
    # MAIN INTERFACE
    # ========================================================

    def create_interface(self):

        main = ttk.Frame(
            self.root,
            padding=30
        )

        main.pack(
            fill="both",
            expand=True
        )

        self.create_header(main)

        self.create_input_section(main)

        self.create_result_section(main)

        self.create_evaluation_section(main)

    # ========================================================
    # HEADER
    # ========================================================

    def create_header(self, parent):

        title = ttk.Label(
            parent,
            text=(
                "Customer Support Ticket Router"
            ),
            style="Title.TLabel"
        )

        title.pack(
            anchor="w"
        )

        subtitle = ttk.Label(
            parent,
            text=(
                "Classify customer requests "
                "into support intents."
            ),
            style="Subtitle.TLabel"
        )

        subtitle.pack(
            anchor="w",
            pady=(5, 20)
        )

    # ========================================================
    # INPUT SECTION
    # ========================================================

    def create_input_section(self, parent):

        input_label = ttk.Label(
            parent,
            text="Customer Request",
            font=(
                "Segoe UI",
                11,
                "bold"
            )
        )

        input_label.pack(
            anchor="w",
            pady=(0, 8)
        )

        self.text_input = tk.Text(
            parent,
            height=4,
            font=(
                "Segoe UI",
                11
            ),
            wrap="word",
            padx=12,
            pady=12,
            relief="solid",
            borderwidth=1
        )

        self.text_input.pack(
            fill="x"
        )

        self.classify_button = ttk.Button(
            parent,
            text="Classify Ticket",
            style="Classify.TButton",
            command=self.classify_ticket
        )

        self.classify_button.pack(
            pady=15
        )

    # ========================================================
    # RESULT SECTION
    # ========================================================

    def create_result_section(self, parent):

        result_frame = ttk.LabelFrame(
            parent,
            text="Classification Result",
            padding=20
        )

        result_frame.pack(
            fill="both",
            expand=True
        )

        result_title = ttk.Label(
            result_frame,
            text="Predicted Intent"
        )

        result_title.pack(
            anchor="center"
        )

        self.result = tk.Label(
            result_frame,
            text="—",
            font=(
                "Segoe UI",
                18,
                "bold"
            ),
            anchor="center",
            justify="center",
            wraplength=600
        )

        self.result.pack(
            fill="x",
            pady=(12, 15)
        )

        self.status = ttk.Label(
            result_frame,
            text=(
                "Enter a customer request and "
                "click Classify Ticket."
            ),
            anchor="center",
            justify="center"
        )

        self.status.pack(
            fill="x"
        )

    # ========================================================
    # EVALUATION SECTION
    # ========================================================

    def create_evaluation_section(self, parent):

        evaluation_frame = ttk.LabelFrame(
            parent,
            text="Model Evaluation",
            padding=15
        )

        evaluation_frame.pack(
            fill="x",
            pady=(20, 0)
        )

        metrics = ttk.Frame(
            evaluation_frame
        )

        metrics.pack(
            fill="x"
        )

        self.create_metric(
            metrics,
            "Training Samples",
            self.router.training_size
        )

        self.create_metric(
            metrics,
            "Test Samples",
            self.router.test_size
        )

        self.create_metric(
            metrics,
            "Intents",
            self.router.number_of_intents
        )

        self.create_metric(
            metrics,
            "Accuracy",
            f"{self.router.test_accuracy * 100:.2f}%"
        )

        self.create_metric(
            metrics,
            "Macro F1",
            f"{self.router.test_f1:.4f}"
        )

    # ========================================================
    # METRIC
    # ========================================================

    def create_metric(
        self,
        parent,
        title,
        value
    ):

        metric = ttk.Frame(
            parent
        )

        metric.pack(
            side="left",
            expand=True
        )

        ttk.Label(
            metric,
            text=title
        ).pack()

        ttk.Label(
            metric,
            text=str(value),
            style="Metric.TLabel"
        ).pack()

    # ========================================================
    # RESPONSIVE LAYOUT
    # ========================================================

    def on_resize(self, event):

        if event.widget != self.root:
            return

        new_width = max(
            300,
            event.width - 100
        )

        self.result.config(
            wraplength=new_width
        )

    # ========================================================
    # CLASSIFY TICKET
    # ========================================================

    def classify_ticket(self):

        text = self.text_input.get(
            "1.0",
            tk.END
        ).strip()

        try:

            prediction = (
                self.router.predict(text)
            )

            self.result.config(
                text=prediction
            )

            self.status.config(
                text=(
                    "Ticket successfully classified."
                )
            )

        except ValueError as error:

            messagebox.showwarning(
                "Missing Request",
                str(error)
            )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():

    print(
        "Loading customer support classifier..."
    )

    router = CustomerSupportRouter()

    print(
        "Classifier ready."
    )

    print(
        f"Training samples: "
        f"{router.training_size}"
    )

    print(
        f"Test samples: "
        f"{router.test_size}"
    )

    print(
        f"Number of intents: "
        f"{router.number_of_intents}"
    )

    print(
        f"Accuracy: "
        f"{router.test_accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{router.test_f1:.4f}"
    )

    root = tk.Tk()

    CustomerSupportGUI(
        root,
        router
    )

    root.mainloop()


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()

