import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def create_combined_pdf(figures_list: list[Figure], output_path: str) -> None:
    with PdfPages(output_path) as pdf:
        for fig in figures_list:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
