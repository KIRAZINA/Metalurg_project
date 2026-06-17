from pathlib import Path

import matplotlib.pyplot as plt

from test_metal.io.pdf import create_combined_pdf


class TestCreateCombinedPDF:
    def test_creates_pdf_with_figures(self, tmp_path):
        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2], [3, 4])
        fig2, ax2 = plt.subplots()
        ax2.plot([5, 6], [7, 8])
        output_path = str(tmp_path / "output.pdf")
        create_combined_pdf([fig1, fig2], output_path)
        assert Path(output_path).exists()

    def test_empty_figures_list(self, tmp_path):
        output_path = str(tmp_path / "empty.pdf")
        create_combined_pdf([], output_path)
        assert Path(output_path).parent.exists()

    def test_single_figure(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        output_path = str(tmp_path / "single.pdf")
        create_combined_pdf([fig], output_path)
        assert Path(output_path).exists()
