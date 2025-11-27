
import numpy as np
import matplotlib.pyplot as plt


class NewtonRaphsonSolver:
    """
    Class for solving f(x) = 0 
    """

    def __init__(self, func, x0, tol=1e-8, max_iter=20, h=1e-6, name=""):
        """
        Parameters
        ----------
        func : callable
            Function f(x) whose root we want to find.
        x0 : float
            Initial guess.
        tol : float
            Tolerance for stopping criterion |x_{k+1} - x_k|.
        max_iter : int
            Maximum number of iterations.
        h : float
            Step size used for numerical differentiation.
        name : str
            Optional name for printing (equation / initial guess).
        """
        self.func = func
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.h = h
        self.name = name
        # history will store dictionaries: {"iter": k, "x": x, "f": f(x), "df": f'(x)}
        self.history = []

    # ---------------------------------------------
    # Numerical derivative: central difference
    # f'(x) ≈ [f(x + h) - f(x - h)] / (2h)
    # ---------------------------------------------
    def numerical_derivative(self, x):
        return (self.func(x + self.h) - self.func(x - self.h)) / (2 * self.h)

    # ---------------------------------------------
    # Newton–Raphson iteration
    # x_{k+1} = x_k - f(x_k) / f'(x_k)
    # ---------------------------------------------
    def solve(self):
        """
        Run the Newton-Raphson iteration.

        Returns
        -------
        root : float
            Approximated root.
        """
        x = self.x0
        self.history.clear()

        for k in range(self.max_iter):
            fx = self.func(x)
            dfx = self.numerical_derivative(x)

            # Save current iteration data
            self.history.append({"iter": k, "x": x, "f": fx, "df": dfx})

            if abs(dfx) < 1e-12:
                print("Warning: derivative is close to zero. Stopping iteration.")
                break

            x_new = x - fx / dfx

            # Check convergence
            if abs(x_new - x) < self.tol:
                x = x_new
                fx = self.func(x)
                dfx = self.numerical_derivative(x)
                # Save final step
                self.history.append({"iter": k + 1, "x": x, "f": fx, "df": dfx})
                break

            x = x_new

        return x

    # ---------------------------------------------
    # Helper: get all x_k values as a NumPy array
    # ---------------------------------------------
    def trajectory(self):
        return np.array([row["x"] for row in self.history])

    # ---------------------------------------------
    # Print iteration table: k, x_k, f(x_k), df(x_k)
    # ---------------------------------------------
    def print_table(self):
        if self.name:
            print(f"\nIteration table for {self.name}")
        else:
            print("\nIteration table")

        # Header
        print(f"{'iter':>4} {'x_k':>15} {'f(x_k)':>15} {'df(x_k)':>15}")
        print("-" * 55)

        # Rows
        for row in self.history:
            print(
                f"{row['iter']:4d} "
                f"{row['x']:15.10f} "
                f"{row['f']:15.10e} "
                f"{row['df']:15.10e}"
            )


# ==========================================================
# 1) First equation: f1(x) = x^2 - 100
# ==========================================================
def f1(x):
    return x**2 - 100


solver1_pos = NewtonRaphsonSolver(
    func=f1, x0=5.0, name="Equation 1: x^2 - 100 = 0, initial x0 = 5"
)
root1_pos = solver1_pos.solve()
traj1_pos = solver1_pos.trajectory()

solver1_neg = NewtonRaphsonSolver(
    func=f1, x0=-5.0, name="Equation 1: x^2 - 100 = 0, initial x0 = -5"
)
root1_neg = solver1_neg.solve()
traj1_neg = solver1_neg.trajectory()

print("=== Equation 1: x^2 - 100 = 0 ===")
print(f"Root from x0 = 5 -> {root1_pos}")
print(f"Root from x0 = -5 -> {root1_neg}")
solver1_pos.print_table()
solver1_neg.print_table()


# ==========================================================
# 2) Second equation: f2(x) = x^2 + 2x - 3
# ==========================================================
def f2(x):
    return x**2 + 2*x - 3


solver2_pos = NewtonRaphsonSolver(
    func=f2, x0=10.0, name="Equation 2: x^2 + 2x - 3 = 0, initial x0 = 10"
)
root2_pos = solver2_pos.solve()
traj2_pos = solver2_pos.trajectory()

solver2_neg = NewtonRaphsonSolver(
    func=f2, x0=-5.0, name="Equation 2: x^2 + 2x - 3 = 0, initial x0 = -5"
)
root2_neg = solver2_neg.solve()
traj2_neg = solver2_neg.trajectory()

print("\n=== Equation 2: x^2 + 2x - 3 = 0 ===")
print(f"Root from x0 = 10 -> {root2_pos}")
print(f"Root from x0 = -5 -> {root2_neg}")
solver2_pos.print_table()
solver2_neg.print_table()


# ==========================================================
# 3) Plot both equations and Newton iteration trajectories
# ==========================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ----- Plot for Equation 1 -----
x_vals1 = np.linspace(-15, 15, 400)
axes[0].plot(x_vals1, f1(x_vals1), label="f(x) = x^2 - 100")
axes[0].axhline(0, linestyle="--", linewidth=1)

axes[0].scatter(traj1_pos, f1(traj1_pos), marker="o", label="Newton (x0 = 5)")
axes[0].plot(traj1_pos, f1(traj1_pos), linestyle="--")

axes[0].scatter(traj1_neg, f1(traj1_neg), marker="s", label="Newton (x0 = -5)")
axes[0].plot(traj1_neg, f1(traj1_neg), linestyle="--")

axes[0].set_title("Equation 1: x^2 - 100 = 0")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].grid(True)
axes[0].legend()

# ----- Plot for Equation 2 -----
x_vals2 = np.linspace(-6, 6, 400)
axes[1].plot(x_vals2, f2(x_vals2), label="f(x) = x^2 + 2x - 3")
axes[1].axhline(0, linestyle="--", linewidth=1)

axes[1].scatter(traj2_pos, f2(traj2_pos), marker="o", label="Newton (x0 = 10)")
axes[1].plot(traj2_pos, f2(traj2_pos), linestyle="--")

axes[1].scatter(traj2_neg, f2(traj2_neg), marker="s", label="Newton (x0 = -5)")
axes[1].plot(traj2_neg, f2(traj2_neg), linestyle="--")

axes[1].set_title("Equation 2: x^2 + 2x - 3 = 0")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
