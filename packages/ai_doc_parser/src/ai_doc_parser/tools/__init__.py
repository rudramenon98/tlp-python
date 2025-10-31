import cProfile
import logging
import os
import pstats
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

def plot_pstats_file(pstats_file_path: Path, png_file_path: Path) -> plt.Figure:
    """
    Visualizes profiling statistics by converting them to a call graph image.

    Args:
        pstats_file_path (Path): Path to the .pstats file containing profiling data.
        png_file_path (Path): Path where the output PNG image will be saved.

    Returns:
        plt.Figure: Matplotlib figure containing the rendered profiling graph.
    """
    pstats_file_path = str(pstats_file_path)
    png_file_path = str(png_file_path)

    stats = pstats.Stats(pstats_file_path)
    total_time = stats.sort_stats("cumulative").total_tt
    stats.strip_dirs().sort_stats(-1).print_stats()
    stats.sort_stats("cumtime").print_callers(20)

    os.system(
        f"gprof2dot -f pstats {pstats_file_path} | dot -Tpng -o {png_file_path}"
    )
    log.debug("Saved profiling graph to %s", png_file_path)

    output = cv2.imread(png_file_path)

    # draw total time on the top left of the image
    text = f"Total Time: {total_time:.2f} s"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_x = 10
    text_y = 20
    cv2.putText(output, text, (text_x, text_y), font, font_scale, (0, 0, 0),
        font_thickness, cv2.LINE_AA)
    cv2.imwrite(png_file_path, output)

    fig, ax = plt.subplots()
    ax.imshow(output)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Vtag Timing Profile. TotTime: {total_time}")
    return fig


class TimingProfiler:
    """
    Context manager for profiling blocks of Python code and visualizing the results.

    Attributes:
        png_file_path (str): Path to save the output PNG image.
        pstats_file_path (str): Path to save the .pstats file (optional).
        save_files (bool): Whether to retain profiling output files after execution.
        fig (plt.Figure): Matplotlib figure created from profiling output.
        total_time (float): Total execution time recorded in profiling (if extracted).
    """

    def __init__(self, png_file_path: str,
            pstats_file_path: str  = None,
            save_files: bool = False):
        self.png_file_path = Path(png_file_path).with_suffix(".png").resolve()
        self.pstats_file_path = pstats_file_path or png_file_path
        self.pstats_file_path = Path(self.pstats_file_path).with_suffix(
            ".pstats").resolve()
        self.profiler = cProfile.Profile()
        self.fig = None
        self.total_time = None
        self.save_files = save_files

    def __enter__(self):
        self.profiler.enable()
        return self

    def plot(self):
        self.fig = plot_pstats_file(self.pstats_file_path, self.png_file_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        self.profiler.dump_stats(self.pstats_file_path)

        if not self.save_files:
            # delete any saved files
            os.remove(self.png_file_path)
            os.remove(self.pstats_file_path)
            log.info("Deleted profiling files: %s, %s", self.png_file_path,
                self.pstats_file_path)
        else:
            # save the files
            log.info("Saved profiling files: %s, %s", self.png_file_path,
                self.pstats_file_path)


def main():
    """
    Example usage of TimingProfiler to profile a block of code containing sample functions.
    """
    import time
    def slow_func():
        time.sleep(0.2)

    def fast_func():
        time.sleep(0.05)

    def combined_func():
        time.sleep(0.001)
        slow_func()
        fast_func()

    with TimingProfiler("test.png", save_files=False) as profiler:
        for i in range(5):
            slow_func()
            fast_func()
            combined_func()
    plt.show()


if __name__ == "__main__":
    main()
