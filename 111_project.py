import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------- Helper functions ---------- #

def parse_manual_data(text: str) -> np.ndarray | None:
    """
    Parse user-entered text into a 1D numpy array of floats.
    Accepts numbers separated by spaces, commas, or newlines.
    Returns None if parsing fails.
    """
    if not text.strip():
        return np.array([])

    # Replace commas with spaces, then split on whitespace
    cleaned = text.replace(",", " ")
    parts = cleaned.split()
    try:
        data = np.array([float(p) for p in parts], dtype=float)
        return data
    except ValueError:
        return None


def compute_fit_errors(data: np.ndarray, dist, bins: int) -> tuple[float, float]:
    """
    Approximate the quality of fit by comparing the histogram
    (as a density estimate) to the distribution's pdf at the
    bin centres.

    Returns (mean_abs_error, max_abs_error).
    """
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pdf_vals = dist.pdf(bin_centres)

    errors = np.abs(pdf_vals - hist)
    mae = float(np.mean(errors))
    max_err = float(np.max(errors))
    return mae, max_err


def build_param_labels(dist_obj, num_params: int) -> list[str]:
    """
    Try to label parameters as 'shape 1', 'shape 2', 'loc', 'scale'
    based on SciPy's convention: shape params first, then loc, scale.
    """
    # Many SciPy continuous dists use: shape(s), loc, scale
    # Some have no shape params (numargs == 0)
    num_shape = getattr(dist_obj, "numargs", 0)
    labels = []

    for i in range(num_params):
        if i < num_shape:
            labels.append(f"shape {i + 1}")
        elif i == num_shape:
            labels.append("loc")
        else:
            labels.append("scale")

    return labels


def build_manual_sliders(params_fit: tuple, dist_obj, data: np.ndarray) -> list[float]:
    """
    Create sliders for each parameter, starting from the auto-fit values.
    Returns a list of parameter values chosen by the user.
    """
    params_fit = list(params_fit)
    num_params = len(params_fit)
    labels = build_param_labels(dist_obj, num_params)

    data_min = float(np.min(data))
    data_max = float(np.max(data))
    span = max(data_max - data_min, 1.0)

    manual_params = []
    for i, (p, label) in enumerate(zip(params_fit, labels)):
        p = float(p)
        # Basic heuristic for slider ranges:
        # - loc: around data range
        # - scale/shape: positive and based on data span
        if "loc" in label:
            slider_min = data_min - span
            slider_max = data_max + span
        elif "scale" in label or "shape" in label:
            # Ensure positive range
            base = span if span > 0 else max(abs(p), 1.0)
            slider_min = max(1e-6, base * 0.1)
            slider_max = base * 5.0
            # If fitted value is way outside, expand range to include it
            slider_min = min(slider_min, p * 0.5) if p > 0 else slider_min
            slider_max = max(slider_max, p * 2.0) if p > 0 else slider_max
        else:
            # Generic fallback: symmetric around p
            delta = max(abs(p), 1.0)
            slider_min = p - 2 * delta
            slider_max = p + 2 * delta

        # Ensure slider_min < slider_max
        if slider_min >= slider_max:
            slider_min, slider_max = slider_min - 1.0, slider_min + 1.0

        val = st.slider(
            label,
            min_value=float(slider_min),
            max_value=float(slider_max),
            value=float(p),
        )
        manual_params.append(val)

    return manual_params


# ---------- Main app ---------- #

def main():
    st.set_page_config(
        page_title="Distribution Fitting App",
        layout="wide",
    )

    st.title("📊 Histogram & Distribution Fitting Tool")
    st.write(
        "This app lets you upload or enter data, fit a variety of statistical "
        "distributions using `scipy.stats`, and visualise the histogram plus "
        "the fitted probability density function (PDF). You can also manually "
        "adjust the parameters using sliders."
    )

    # Define the available distributions
    DISTRIBUTIONS = {
        "Normal (norm)": stats.norm,
        "Gamma": stats.gamma,
        "Exponential": stats.expon,
        "Weibull (weibull_min)": stats.weibull_min,
        "Lognormal": stats.lognorm,
        "Chi-square": stats.chi2,
        "Beta": stats.beta,
        "Uniform": stats.uniform,
        "Triangular (triang)": stats.triang,
        "Pareto": stats.pareto,
    }

    # ----- SIDEBAR: Data input and options ----- #
    st.sidebar.header("Data Input")

    data_mode = st.sidebar.radio(
        "Choose how to provide data:",
        ("Manual entry", "Upload CSV"),
    )

    data = None

    if data_mode == "Manual entry":
        text = st.sidebar.text_area(
            "Enter numbers (separated by spaces, commas, or new lines):",
            height=150,
            placeholder="Example:\n1.2 3.4 2.1 5.6 4.3",
        )
        if text.strip():
            parsed = parse_manual_data(text)
            if parsed is None:
                st.sidebar.error("Could not parse the data. Please check formatting.")
            else:
                data = parsed

    else:  # Upload CSV
        uploaded = st.sidebar.file_uploader(
            "Upload a CSV file", type=["csv"]
        )
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if df.empty:
                    st.sidebar.error("The uploaded CSV file is empty.")
                else:
                    col = st.sidebar.selectbox(
                        "Choose column to use as data:",
                        df.columns,
                    )
                    series = df[col].dropna()
                    if series.empty:
                        st.sidebar.error("Selected column has no numeric data.")
                    else:
                        data = series.to_numpy(dtype=float)
                        st.sidebar.write(f"Loaded {len(data)} values from column `{col}`.")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {e}")

    st.sidebar.header("Fitting Options")

    dist_name = st.sidebar.selectbox(
        "Choose distribution:",
        list(DISTRIBUTIONS.keys()),
    )
    dist_obj = DISTRIBUTIONS[dist_name]

    fit_mode = st.sidebar.radio(
        "Fitting mode:",
        ("Automatic fit", "Manual fit"),
    )

    bins = st.sidebar.slider(
        "Number of histogram bins:",
        min_value=5,
        max_value=100,
        value=30,
    )

    # ----- MAIN AREA ----- #

    if data is None or data.size < 2:
        st.warning("Please enter or upload at least two numeric data points to begin.")
        return

    # Ensure 1D
    data = np.ravel(data.astype(float))

    # Automatic parameter fitting using scipy.stats
    try:
        params_fit = dist_obj.fit(data)
    except Exception as e:
        st.error(f"Could not fit the distribution `{dist_name}` to the data: {e}")
        return

    num_params = len(params_fit)
    param_labels = build_param_labels(dist_obj, num_params)

    # Create x-range for plotting the PDF
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    span = max(data_max - data_min, 1.0)
    x_min = data_min - 0.1 * span
    x_max = data_max + 0.1 * span
    x = np.linspace(x_min, x_max, 400)

    # Decide which parameters to use for plotting
    if fit_mode == "Automatic fit":
        current_params = params_fit
        st.subheader("Automatic Fit")
    else:
        st.subheader("Manual Fit")
        st.write("Use the sliders below to adjust the distribution parameters.")
        current_params = build_manual_sliders(params_fit, dist_obj, data)
        current_params = tuple(current_params)

    # Create frozen distribution with chosen parameters
    try:
        dist_current = dist_obj(*current_params)
    except Exception as e:
        st.error(f"Could not create distribution with chosen parameters: {e}")
        return

    pdf_vals = dist_current.pdf(x)

    # Layout: left = plot, right = parameters + errors
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(
            data,
            bins=bins,
            density=True,
            alpha=0.6,
            color="tab:blue",
            edgecolor="black",
            label="Data histogram",
        )
        ax.plot(
            x,
            pdf_vals,
            "r-",
            linewidth=2,
            label=f"{dist_name} PDF",
        )
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)

    with col2:
        st.markdown("### Parameters")
        param_table = {
            "Parameter": param_labels,
            "Value": [float(p) for p in current_params],
        }
        st.table(pd.DataFrame(param_table))

        mae, max_err = compute_fit_errors(data, dist_current, bins)
        st.markdown("### Fit Quality")
        st.write(f"**Mean absolute error** (hist vs PDF): `{mae:.4f}`")
        st.write(f"**Max absolute error** (hist vs PDF): `{max_err:.4f}`")

        st.markdown("### Data Summary")
        st.write(f"Number of data points: `{len(data)}`")
        st.write(f"Min: `{data_min:.4f}`  |  Max: `{data_max:.4f}`")
        st.write(f"Mean: `{np.mean(data):.4f}`  |  Std: `{np.std(data):.4f}`")


if __name__ == "__main__":
    main()