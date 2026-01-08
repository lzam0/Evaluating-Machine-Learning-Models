from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "extracted_features" / "hand_landmarks_sanitised.csv"

skip = 1
def plot():
    data = np.loadtxt(INPUT, delimiter=",", skiprows=skip, dtype=str)
    labels = data[:, 0]
    coords = data[:, 1:].astype(float).reshape(-1, 21, 3)

    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]

    i = 0  # pick any sample

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    hand = coords[i]

    for a, b in connections:
        ax.plot(
            [hand[a,0], hand[b,0]],
            [hand[a,1], hand[b,1]],
            [hand[a,2], hand[b,2]],
            "k-"
        )

    ax.scatter(hand[:,0], hand[:,1], hand[:,2])
    ax.set_title(labels[i])
    ax.set_box_aspect((1,1,1))
    plt.show()

while True:
    if input(str()) == "n":
        skip += 1
        plot()