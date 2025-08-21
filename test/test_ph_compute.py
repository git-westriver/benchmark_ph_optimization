from ph_opt import RipsPH
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from itertools import product

def sample_two_rotated_ellipses(seed: int | None = None):
    """
    Generate two sets of points sampled from the perimeters of two randomly placed
    and rotated ellipses inside [0,1]^2, then merge them.

    Returns
    -------
    points : np.ndarray, shape (n1+n2, 2)
        Coordinates of the merged point cloud.
    labels : np.ndarray, shape (n1+n2,)
        Labels for each point (0: first ellipse, 1: second ellipse).
    params : dict
        Parameters used to generate the ellipses (n1, n2, a1, b1, a2, b2, centers X, Y, angles).
    """
    rng = np.random.default_rng(seed)

    # n1, n2 ~ Uniform integers between 10 and 50
    n1 = int(rng.integers(10, 51))  # upper bound is exclusive
    n2 = int(rng.integers(10, 51))

    # 0 <= a,b <= 1 with 1/4 < b/a < 4
    def sample_ab():
        a = rng.uniform(0.1, 0.5) 
        lower_bound = max(0.1, a / 4)
        upper_bound = min(1.0, a * 4)
        b = rng.uniform(lower_bound, upper_bound)
        return a, b

    a1, b1 = sample_ab()
    a2, b2 = sample_ab()

    # Centers X, Y ~ Uniform([0,1]^2)
    X = rng.uniform(0.0, 1.0, size=2)
    Y = rng.uniform(0.0, 1.0, size=2)

    # Random rotation angles
    theta1 = float(rng.uniform(0.0, 2.0 * np.pi))
    theta2 = float(rng.uniform(0.0, 2.0 * np.pi))

    def ellipse_points(n: int, center: np.ndarray, a: float, b: float, angle: float) -> np.ndarray:
        """
        Generate n points on the perimeter of an ellipse:
        Parametric form (a cos t, b sin t) rotated by angle and shifted by center.
        Note: points are uniform in parameter t, not in arc length.
        """
        t = rng.uniform(0.0, 2.0 * np.pi, size=n)
        cosA, sinA = np.cos(angle), np.sin(angle)
        # Rotation matrix
        R = np.array([[cosA, -sinA],
                      [sinA,  cosA]])
        pts_local = np.stack([a * np.cos(t), b * np.sin(t)], axis=1)
        pts = pts_local @ R.T
        pts += center
        return pts

    # n1 points on ellipse 1
    pts1 = ellipse_points(n1, X, a1, b1, theta1)
    # n2 points on ellipse 2
    pts2 = ellipse_points(n2, Y, a2, b2, theta2)

    # Merge
    points = np.vstack([pts1, pts2])
    labels = np.concatenate([np.zeros(n1, dtype=int), np.ones(n2, dtype=int)])

    # Add noise uniformly from [-noise_level, noise_level]
    noise_level = 0.05
    noise = rng.uniform(-noise_level, noise_level, size=points.shape)
    points = points + noise

    params = {
        "n1": n1, "n2": n2,
        "a1": float(a1), "b1": float(b1),
        "a2": float(a2), "b2": float(b2),
        "X": X.tolist(), "Y": Y.tolist(),
        "theta1": theta1, "theta2": theta2,
    }
    return points, labels, params

def plot_and_save(points, labels, filename):
    """
    Plot the points and save the figure.
    """
    plt.figure(figsize=(6,6))
    plt.scatter(points[labels==0,0], points[labels==0,1], c="blue", label="Ellipse 1", alpha=0.7)
    plt.scatter(points[labels==1,0], points[labels==1,1], c="red", label="Ellipse 2", alpha=0.7)
    plt.legend()
    plt.title("Sampled Points from Two Rotated Ellipses")
    plt.axis("equal")
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    for idx in tqdm(range(200)):
        # --- Generate ---
        points, labels, params = sample_two_rotated_ellipses(seed=idx)

        # --- Plot and save ---
        if idx < 10:
            if not os.path.exists("tmp/ph_compute"):
                os.makedirs("tmp/ph_compute")
            plot_and_save(points, labels, f"tmp/ph_compute/ellipses_{idx}.png")

        # --- get correct PH ---
        correct_ph = RipsPH(points, maxdim=1)
        correct_barcode_0 = correct_ph.get_barcode(dim=0).sort()
        correct_barcode_1 = correct_ph.get_barcode(dim=1).sort()
        assert correct_ph.giotto_dgm is not None
        
        # --- get testee PH (left) ---
        for enclosing_opt, emergent_opt, clearing_opt in product([False, True], repeat=3):
            testee_ph = RipsPH(points, maxdim=1)
            testee_ph.compute_ph(enclosing_opt=enclosing_opt, 
                                 emergent_opt=emergent_opt, 
                                 clearing_opt=clearing_opt)
            testee_barcode_0 = testee_ph.get_barcode(dim=0).sort()
            testee_barcode_1 = testee_ph.get_barcode(dim=1).sort()

            # check if barcodes match
            assert np.array_equal(correct_barcode_0, testee_barcode_0), \
                f"Mismatch in barcode 0 for idx={idx}, opts={enclosing_opt, emergent_opt, clearing_opt}"
            assert np.array_equal(correct_barcode_1, testee_barcode_1), \
                f"Mismatch in barcode 1 for idx={idx}, opts={enclosing_opt, emergent_opt, clearing_opt}"
        
        # --- get testee PH (right) ---
        for enclosing_opt, emergent_opt in product([False, True], repeat=2):
            testee_ph = RipsPH(points, maxdim=1)
            testee_ph.compute_ph_right(enclosing_opt=enclosing_opt, 
                                       emergent_opt=emergent_opt)
            testee_barcode_0 = testee_ph.get_barcode(dim=0).sort()
            testee_barcode_1 = testee_ph.get_barcode(dim=1).sort()

            # check if barcodes match
            assert np.array_equal(correct_barcode_0, testee_barcode_0), \
                f"Mismatch in barcode 0 for idx={idx}, opts={enclosing_opt, emergent_opt, clearing_opt}"
            assert np.array_equal(correct_barcode_1, testee_barcode_1), \
                f"Mismatch in barcode 1 for idx={idx}, opts={enclosing_opt, emergent_opt, clearing_opt}"

    print("All tests passed successfully!")