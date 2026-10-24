import os
import numpy as np
import matplotlib.pyplot as plt

# the five embeddings of Person A
angles = [5, 15, 25, 35, 45]

# the margins
m_sphere = 2
m_cos = 0.20
m_arc = np.radians(10)

# the four cases
methods = ["none", "sphere", "cos", "arc"]
titles = ["No margin: cos(theta)",
          "SphereFace: cos(2 theta)",
          "CosFace: cos(theta) - 0.20",
          "ArcFace: cos(theta + 10 deg)"]


def draw_panel(ax, method, title):
    # unit circle
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="lightgray")

    # class direction W_A = (1, 0)
    ax.arrow(0, 0, 1, 0, head_width=0.03, length_includes_head=True, color="red")
    ax.text(1.05, -0.08, "W_A", color="red", fontsize=12)

    # one color for each embedding
    colors = ["purple", "blue", "green", "orange", "brown"]

    for i in range(5):
        a = angles[i]
        c = colors[i]
        theta = np.radians(a)

        # the real embedding (solid dot on the circle)
        x = np.cos(theta)
        y = np.sin(theta)
        ax.plot([0, x], [0, y], color=c, linewidth=1.5)
        ax.scatter(x, y, color=c, s=60)
        ax.text(1.1 * x, 1.1 * y, str(a) + " deg", color=c, fontsize=9)

        # the effective angle (what the loss sees)
        if method == "sphere":
            eff = m_sphere * theta
        elif method == "cos":
            eff = np.arccos(np.cos(theta) - m_cos)
        elif method == "arc":
            eff = theta + m_arc
        else:
            eff = theta

        # draw the effective angle and the margin (only if there is a margin)
        if method != "none":
            # hollow dot = effective position
            ax.scatter(0.8 * np.cos(eff), 0.8 * np.sin(eff),
                       facecolors="none", edgecolors=c, s=60)

            # dashed arc from theta to the effective angle = the margin
            arc = np.linspace(theta, eff, 30)
            r = 0.55 + 0.05 * i
            ax.plot(r * np.cos(arc), r * np.sin(arc), "--", color=c, linewidth=1)

            print(method, "| theta =", a, "| effective angle =",
                  round(np.degrees(eff), 2), "| margin =",
                  round(np.degrees(eff - theta), 2))

    # small arc to show theta
    arc = np.linspace(0, np.radians(45), 30)
    ax.plot(0.3 * np.cos(arc), 0.3 * np.sin(arc), color="black", linewidth=1)
    ax.text(0.33, 0.08, "theta", fontsize=12)

    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.set_xlim(-0.2, 1.3)
    ax.set_ylim(-0.2, 1.3)
    ax.grid(alpha=0.3)


def run():
    # make the folder for the plots
    os.makedirs("plots/part3", exist_ok=True)

    print("=== PART 7 ===")

    # one figure with the four methods
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for i in range(4):
        draw_panel(axes[i], methods[i], titles[i])
    fig.suptitle("Solid dots = real embeddings (angle theta). "
                 "Hollow dots and dashed arcs = effective angle seen by the loss")
    fig.tight_layout()
    fig.savefig("plots/part3/part3_all_methods.png", dpi=150)
    plt.close(fig)

    # one figure for each method
    for i in range(4):
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_panel(ax, methods[i], titles[i])
        fig.savefig("plots/part3/part3_" + methods[i] + ".png", dpi=150)
        plt.close(fig)

    print()
    print("Plots saved in: plots/part3")