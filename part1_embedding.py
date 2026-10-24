import os
import numpy as np
import matplotlib.pyplot as plt


def run():
    # make the folder for the plots
    os.makedirs("plots/part1", exist_ok=True)

    # ---------------- PART 1 ----------------
    print("=== PART 1 ===")

    x = np.array([0.8, 0.6])
    W_A = np.array([1.0, 0.0])
    W_B = np.array([0.0, 1.0])

    # norms
    print("norm of x   =", np.linalg.norm(x))
    print("norm of W_A =", np.linalg.norm(W_A))
    print("norm of W_B =", np.linalg.norm(W_B))

    # cosine similarities
    cos_A = np.dot(x, W_A)
    cos_B = np.dot(x, W_B)
    print("cos(theta_A) =", cos_A)
    print("cos(theta_B) =", cos_B)

    # angles in degrees
    theta_A = np.degrees(np.arccos(cos_A))
    theta_B = np.degrees(np.arccos(cos_B))
    print("theta_A =", round(theta_A, 2), "degrees")
    print("theta_B =", round(theta_B, 2), "degrees")

    # decision
    if theta_A < theta_B:
        print("The embedding belongs to Class A")
    else:
        print("The embedding belongs to Class B")

    # plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # unit circle
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="lightgray")

    # the three vectors
    ax.arrow(0, 0, x[0], x[1], head_width=0.03, length_includes_head=True, color="blue")
    ax.arrow(0, 0, W_A[0], W_A[1], head_width=0.03, length_includes_head=True, color="red")
    ax.arrow(0, 0, W_B[0], W_B[1], head_width=0.03, length_includes_head=True, color="green")

    ax.text(0.85, 0.65, "x", color="blue", fontsize=12)
    ax.text(1.05, 0.0, "W_A", color="red", fontsize=12)
    ax.text(0.05, 1.05, "W_B", color="green", fontsize=12)

    ax.text(0.4, 0.06, "theta_A = " + str(round(theta_A, 1)) + " deg", color="red")
    ax.text(0.05, 0.55, "theta_B = " + str(round(theta_B, 1)) + " deg", color="green")

    ax.set_title("Part 1: embedding x and class directions")
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.grid(alpha=0.3)

    fig.savefig("plots/part1/part1_embedding_2d.png", dpi=150)
    plt.close(fig)

    # ---------------- PART 2 ----------------
    print()
    print("**** PART 2 ****")

    cases = [(30, 45), (39, 40)]   # (theta_A, theta_B)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(2):
        ax = axes[i]
        tA = cases[i][0]
        tB = cases[i][1]

        # vectors: W_A on the x axis, W_B placed tA + tB degrees away
        wa = np.array([1.0, 0.0])
        wb = np.array([np.cos(np.radians(tA + tB)), np.sin(np.radians(tA + tB))])
        xe = np.array([np.cos(np.radians(tA)), np.sin(np.radians(tA))])

        # unit circle
        ax.plot(np.cos(t), np.sin(t), color="lightgray")

        ax.arrow(0, 0, xe[0], xe[1], head_width=0.03, length_includes_head=True, color="blue")
        ax.arrow(0, 0, wa[0], wa[1], head_width=0.03, length_includes_head=True, color="red")
        ax.arrow(0, 0, wb[0], wb[1], head_width=0.03, length_includes_head=True, color="green")

        ax.text(xe[0] * 1.08, xe[1] * 1.08, "x", color="blue", fontsize=12)
        ax.text(1.05, 0.0, "W_A", color="red", fontsize=12)
        ax.text(wb[0] * 1.08, wb[1] * 1.08, "W_B", color="green", fontsize=12)

        # is the separation strong or weak?
        if tB - tA >= 10:
            verdict = "strong"
        else:
            verdict = "weak"

        print("theta_A =", tA, " theta_B =", tB, "-> closer to A, separation is", verdict)

        ax.set_title("theta_A = " + str(tA) + ", theta_B = " + str(tB)
                     + " (gap = " + str(tB - tA) + " deg)")
        ax.text(0.0, -0.2, "separation: " + verdict, fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlim(-0.4, 1.3)
        ax.set_ylim(-0.3, 1.3)
        ax.grid(alpha=0.3)

    fig.savefig("plots/part1/part2_angular_decision.png", dpi=150)
    plt.close(fig)

    print()
    print("Plots saved in: plots/part1")