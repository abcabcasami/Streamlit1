import glob
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st


# =========================================================
# Streamlit ページ設定
# =========================================================
st.set_page_config(
    page_title="楕円の座標変換可視化",
    layout="wide"
)

st.title("楕円の座標変換の可視化")
st.write(
    r"""
このアプリでは、相関係数 $\rho$、標準偏差 $\sigma_1,\sigma_2$、
平均 $\mu_1,\mu_2$ を変化させながら、楕円がどのように変換されるかを確認できます。
"""
)


# =========================================================
# 日本語フォント設定
# =========================================================
def setup_japanese_font():
    font_candidates = glob.glob("/usr/share/fonts/**/*NotoSansCJK*.ttc", recursive=True)
    font_candidates += glob.glob("/usr/share/fonts/**/*NotoSansCJK*.otf", recursive=True)
    font_candidates += glob.glob("/usr/share/fonts/**/*NotoSansJP*.ttf", recursive=True)
    font_candidates += glob.glob("/usr/share/fonts/**/*NotoSansJP*.otf", recursive=True)

    if len(font_candidates) > 0:
        font_path = font_candidates[0]
        fm.fontManager.addfont(font_path)
        jp_font_name = fm.FontProperties(fname=font_path).get_name()

        mpl.rcParams["font.family"] = jp_font_name
        mpl.rcParams["font.sans-serif"] = [jp_font_name]
        mpl.rcParams["axes.unicode_minus"] = False

        return jp_font_name
    else:
        mpl.rcParams["axes.unicode_minus"] = False
        return None


jp_font_name = setup_japanese_font()

if jp_font_name is not None:
    st.sidebar.success(f"使用フォント: {jp_font_name}")
else:
    st.sidebar.warning(
        "日本語フォントが見つかりません。日本語が文字化けする場合があります。"
    )


# =========================================================
# 固定パラメータ
# =========================================================
c_values = [0.5, 1.0, 2.0, 3.0]
theta = np.linspace(0, 2 * np.pi, 500)

P = (1 / np.sqrt(2)) * np.array([
    [1,  1],
    [1, -1]
])


# =========================================================
# 補助関数
# =========================================================
def ellipse_in_yprime(c, rho, theta):
    """
    直交変換後の座標 y' = (y1', y2') における楕円

        (y1')^2 / (c(1+rho)) + (y2')^2 / (c(1-rho)) = 1

    を媒介変数表示で生成する。
    """
    a = np.sqrt(c * (1 + rho))
    b = np.sqrt(c * (1 - rho))

    y1p = a * np.cos(theta)
    y2p = b * np.sin(theta)

    Yp = np.vstack([y1p, y2p])
    return Yp, a, b


def yprime_to_y(Yp, P):
    """
    y' -> y
    y = P y'
    """
    return P @ Yp


def y_to_x(Y, sigma1, sigma2, mu1, mu2):
    """
    y -> x
    x_i = sigma_i y_i + mu_i
    """
    D = np.diag([sigma1, sigma2])
    mu = np.array([mu1, mu2])
    return D @ Y + mu.reshape(2, 1)


def setup_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)


# =========================================================
# 描画関数
# =========================================================
def draw_ellipses(rho, sigma1, sigma2, mu1, mu2):
    c_ref = c_values[-1]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    ax1, ax2, ax3 = axes

    # =====================================================
    # 図1：直交変換前の楕円
    # =====================================================
    for c in c_values:
        Yp, a, b = ellipse_in_yprime(c, rho, theta)
        Y = yprime_to_y(Yp, P)
        ax1.plot(Y[0], Y[1], label=f"c = {c}")

    Yp_ref, a_ref, b_ref = ellipse_in_yprime(c_ref, rho, theta)

    dir1 = P @ np.array([1.0, 0.0])
    dir2 = P @ np.array([0.0, 1.0])

    axis1_pos = a_ref * dir1
    axis1_neg = -a_ref * dir1
    axis2_pos = b_ref * dir2
    axis2_neg = -b_ref * dir2

    ax1.plot(
        [axis1_neg[0], axis1_pos[0]],
        [axis1_neg[1], axis1_pos[1]],
        linestyle="--",
        linewidth=2,
        label="y1-prime 軸の像"
    )

    ax1.plot(
        [axis2_neg[0], axis2_pos[0]],
        [axis2_neg[1], axis2_pos[1]],
        linestyle="--",
        linewidth=2,
        label="y2-prime 軸の像"
    )

    L = 3.5
    ax1.plot(
        [-L, L],
        [-L, L],
        linestyle=":",
        linewidth=1.5,
        label="45度方向"
    )

    ax1.plot(
        [-L, L],
        [L, -L],
        linestyle=":",
        linewidth=1.5,
        label="135度方向"
    )

    setup_axes(
        ax1,
        title=r"図1：直交変換前の楕円",
        xlabel="y1",
        ylabel="y2"
    )

    lim1 = np.sqrt(c_ref * (1 + abs(rho))) * 1.8
    ax1.set_xlim(-lim1, lim1)
    ax1.set_ylim(-lim1, lim1)
    ax1.legend(fontsize=9)

    # =====================================================
    # 図2：直交変換後の楕円
    # =====================================================
    for c in c_values:
        Yp, a, b = ellipse_in_yprime(c, rho, theta)
        ax2.plot(Yp[0], Yp[1], label=f"c = {c}")

    Yp_ref, a_ref, b_ref = ellipse_in_yprime(c_ref, rho, theta)

    ax2.plot(
        [-a_ref, a_ref],
        [0, 0],
        linestyle="--",
        linewidth=2,
        label="y1-prime 軸"
    )

    ax2.plot(
        [0, 0],
        [-b_ref, b_ref],
        linestyle="--",
        linewidth=2,
        label="y2-prime 軸"
    )

    setup_axes(
        ax2,
        title=r"図2：直交変換後の楕円",
        xlabel="y1-prime",
        ylabel="y2-prime"
    )

    lim2 = np.sqrt(c_ref * (1 + abs(rho))) * 1.6
    ax2.set_xlim(-lim2, lim2)
    ax2.set_ylim(-lim2, lim2)
    ax2.legend(fontsize=9)

    # =====================================================
    # 図3：x_i = sigma_i y_i + mu_i による拡大・平行移動後
    # =====================================================
    all_x1 = []
    all_x2 = []

    for c in c_values:
        Yp, a, b = ellipse_in_yprime(c, rho, theta)
        Y = yprime_to_y(Yp, P)
        X = y_to_x(Y, sigma1, sigma2, mu1, mu2)

        ax3.plot(X[0], X[1], label=f"c = {c}")
        all_x1.extend(X[0])
        all_x2.extend(X[1])

    ax3.scatter(mu1, mu2, s=70, label="中心 mu")

    ax3.text(
        mu1 + 0.15,
        mu2 + 0.15,
        "mu = (mu1, mu2)",
        fontsize=11
    )

    ax3.arrow(
        mu1,
        mu2,
        sigma1,
        0,
        head_width=0.12,
        length_includes_head=True
    )

    ax3.arrow(
        mu1,
        mu2,
        0,
        sigma2,
        head_width=0.12,
        length_includes_head=True
    )

    ax3.text(
        mu1 + sigma1 + 0.1,
        mu2,
        "sigma1 方向",
        fontsize=10
    )

    ax3.text(
        mu1,
        mu2 + sigma2 + 0.1,
        "sigma2 方向",
        fontsize=10
    )

    setup_axes(
        ax3,
        title="図3：x_i = sigma_i y_i + mu_i による変換後の楕円",
        xlabel="x1",
        ylabel="x2"
    )

    all_x1 = np.array(all_x1)
    all_x2 = np.array(all_x2)

    margin = 1.5
    ax3.set_xlim(all_x1.min() - margin, all_x1.max() + margin)
    ax3.set_ylim(all_x2.min() - margin, all_x2.max() + margin)
    ax3.legend(fontsize=9)

    fig.suptitle(
        f"rho={rho:.2f}, sigma1={sigma1:.2f}, sigma2={sigma2:.2f}, "
        f"mu1={mu1:.2f}, mu2={mu2:.2f}",
        fontsize=16
    )

    plt.tight_layout()

    return fig


# =========================================================
# Streamlit サイドバー
# =========================================================
st.sidebar.header("パラメータ設定")

rho = st.sidebar.slider(
    label="rho",
    min_value=-0.95,
    max_value=0.95,
    value=0.60,
    step=0.05
)

sigma1 = st.sidebar.slider(
    label="sigma1",
    min_value=0.2,
    max_value=5.0,
    value=2.0,
    step=0.1
)

sigma2 = st.sidebar.slider(
    label="sigma2",
    min_value=0.2,
    max_value=5.0,
    value=1.2,
    step=0.1
)

mu1 = st.sidebar.slider(
    label="mu1",
    min_value=-5.0,
    max_value=5.0,
    value=3.0,
    step=0.1
)

mu2 = st.sidebar.slider(
    label="mu2",
    min_value=-5.0,
    max_value=5.0,
    value=-1.0,
    step=0.1
)


# =========================================================
# グラフ表示
# =========================================================
fig = draw_ellipses(rho, sigma1, sigma2, mu1, mu2)
st.pyplot(fig)
plt.close(fig)


# =========================================================
# 数値情報
# =========================================================
st.subheader("現在のパラメータ")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("rho", f"{rho:.3f}")
col2.metric("sigma1", f"{sigma1:.3f}")
col3.metric("sigma2", f"{sigma2:.3f}")
col4.metric("mu1", f"{mu1:.3f}")
col5.metric("mu2", f"{mu2:.3f}")

st.subheader("長軸・短軸の向き")

if rho > 0:
    st.info("rho > 0 なので、図1では長軸は 45° 方向、短軸は 135° 方向になります。")
elif rho < 0:
    st.info("rho < 0 なので、図1では長軸は 135° 方向、短軸は 45° 方向になります。")
else:
    st.info("rho = 0 なので、図1の楕円は回転していません。")


# =========================================================
# 数式説明
# =========================================================
with st.expander("数式の説明を見る"):
    st.markdown(
        r"""
まず、直交変換後の座標を

$$
y' = 
\begin{pmatrix}
y_1' \\
y_2'
\end{pmatrix}
$$

とすると、楕円は

$$
\frac{(y_1')^2}{c(1+\rho)}
+
\frac{(y_2')^2}{c(1-\rho)}
=
1
$$

と表されます。

このとき、半軸の長さは

$$
a = \sqrt{c(1+\rho)}
$$

$$
b = \sqrt{c(1-\rho)}
$$

です。

その後、直交行列

$$
P=
\frac{1}{\sqrt{2}}
\begin{pmatrix}
1 & 1 \\
1 & -1
\end{pmatrix}
$$

によって

$$
y = Py'
$$

と変換しています。

最後に、

$$
x_i = \sigma_i y_i + \mu_i
$$

によって、各軸方向に $\sigma_i$ 倍され、中心が $\mu$ に移動します。
"""
    )
