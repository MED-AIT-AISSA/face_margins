import os
import numpy as np
import matplotlib.pyplot as plt


def run():
    # make the folder for the plots
    os.makedirs("plots/part2", exist_ok=True)

    theta = np.radians(30)

    # margins
    m_sphere = 2
    m_cos = 0.20
    m_arc = np.radians(10)

    # ---------------- PART 3: SphereFace ----------------
    print("=== PART 3: SphereFace (m = 2) ===")
    print("cos(30)     =", round(np.cos(theta), 4))
    print("cos(2 x 30) =", round(np.cos(m_sphere * theta), 4))

    # ---------------- PART 4: CosFace ----------------
    print()
    print("=== PART 4: CosFace (m = 0.20) ===")
    print("cos(30)        =", round(np.cos(theta), 4))
    print("cos(30) - 0.20 =", round(np.cos(theta) - m_cos, 4))
    print("The angle is still 30 degrees")

    # ---------------- PART 5: ArcFace ----------------
    print()
    print("=== PART 5: ArcFace (m = 10 degrees) ===")
    print("cos(30)      =", round(np.cos(theta), 4))
    print("cos(30 + 10) =", round(np.cos(theta + m_arc), 4))
    print("Effective angle = 40 degrees")
    print("To keep the same score, the network needs theta = 20 degrees")

    # ---------------- PART 6: table ----------------
    print()
    print("=== PART 6: comparison table ===")
    print("Standard Softmax | cos(theta)  | cos(theta)       | nothing")
    print("SphereFace       | cos(theta)  | cos(m x theta)   | the angle (multiplicative)")
    print("CosFace          | cos(theta)  | cos(theta) - m   | the cosine (additive)")
    print("ArcFace          | cos(theta)  | cos(theta + m)   | the angle (additive)")

    # ---------------- PLOT 1: target score ----------------
    # angles from 0 to 90 degrees
    th = np.linspace(0, np.radians(90), 500)
    deg = np.degrees(th)

    softmax = np.cos(th)
    sphere = np.cos(m_sphere * th)
    cosface = np.cos(th) - m_cos
    arcface = np.cos(th + m_arc)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(deg, softmax, "k", lw=2.5, label="Softmax: cos(theta)")
    ax.plot(deg, sphere, label="SphereFace: cos(2 theta)")
    ax.plot(deg, cosface, label="CosFace: cos(theta) - 0.20")
    ax.plot(deg, arcface, label="ArcFace: cos(theta + 10 deg)")
    ax.axvline(30, color="gray", linestyle=":")
    ax.set_xlabel("theta (degrees)")
    ax.set_ylabel("score of the correct class")
    ax.set_title("Score of the correct class for each method")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig("plots/part2/part2_logit_curves.png", dpi=150)
    plt.close(fig)

    # ---------------- PLOT 2: effective angle and margin ----------------
    # effective angle = the angle that the loss "sees"
    eff_softmax = deg
    eff_sphere = np.degrees(m_sphere * th)
    eff_cosface = np.degrees(np.arccos(np.cos(th) - m_cos))
    eff_arcface = np.degrees(th + m_arc)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # left: effective angle
    axes[0].plot(deg, eff_softmax, label="Softmax")
    axes[0].plot(deg, eff_sphere, label="SphereFace (m=2)")
    axes[0].plot(deg, eff_cosface, label="CosFace (m=0.20)")
    axes[0].plot(deg, eff_arcface, label="ArcFace (m=10 deg)")
    axes[0].set_title("Effective angle seen by the loss")

    # right: margin = effective angle - real angle
    axes[1].plot(deg, eff_softmax - deg, label="Softmax")
    axes[1].plot(deg, eff_sphere - deg, label="SphereFace (m=2)")
    axes[1].plot(deg, eff_cosface - deg, label="CosFace (m=0.20)")
    axes[1].plot(deg, eff_arcface - deg, label="ArcFace (m=10 deg)")
    axes[1].set_title("Margin (effective angle - real angle)")

    for ax in axes:
        ax.set_xlabel("real theta (degrees)")
        ax.set_ylabel("degrees")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.savefig("plots/part2/part2_effective_angle_and_margin.png", dpi=150)
    plt.close(fig)

    print()
    print("Plots saved in: plots/part2")