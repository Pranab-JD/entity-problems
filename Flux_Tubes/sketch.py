"""
Created on Sat Apr 18 12:43:13 2026

@author: Pranab JD, ChatGPT
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

def add_curved_B(ax, x0, y0, r, color="C4", flip=False):
    if not flip:
        # clockwise
        arc1 = Arc((x0,y0),1.4*r,1.4*r,theta1=120,theta2=180,color=color,lw=2)
        arc2 = Arc((x0,y0),1.4*r,1.4*r,theta1=300,theta2=360,color=color,lw=2)
        ang_r, ang_l, sign = 60, 240, 1
    else:
        # anticlockwise
        arc1 = Arc((x0,y0),1.4*r,1.4*r,theta1=120,theta2=180,color=color,lw=2)
        arc2 = Arc((x0,y0),1.4*r,1.4*r,theta1=300,theta2=360,color=color,lw=2)
        ang_r, ang_l, sign = 60, 240, -1

    ax.add_patch(arc1)
    ax.add_patch(arc2)

    # right arrow
    ang=np.deg2rad(ang_r); xh=x0+0.7*r*np.cos(ang); yh=y0+0.7*r*np.sin(ang); xt=xh-sign*0.25*np.sin(ang); yt=yh+sign*0.2*np.cos(ang)
    ax.annotate("",(xh,yh),(xt,yt),arrowprops=dict(arrowstyle="->",color=color,lw=2))

    # left arrow
    ang=np.deg2rad(ang_l); xh=x0+0.7*r*np.cos(ang); yh=y0+0.7*r*np.sin(ang); xt=xh-sign*0.25*np.sin(ang); yt=yh+sign*0.2*np.cos(ang)
    ax.annotate("",(xh,yh),(xt,yt),arrowprops=dict(arrowstyle="->",color=color,lw=2))

### =============================================== ###

fig = plt.figure(figsize=(10, 6), dpi=300)
ax = fig.add_subplot(111)

# Domain
Lx, Ly = 10, 6
ax.set(xlim=(0, Lx), ylim=(-Ly/2, Ly/2), aspect=1)

for spine in ax.spines.values():
    spine.set_color("C2")

# Flux Tubes
r = 1.2
x0 = Lx/2
y1 = -1.5
y2 = 1.5

theta = np.linspace(0, 2*np.pi, 200)

# Tube 1
ax.plot(x0 + r*np.cos(theta), y1 + r*np.sin(theta), color="red")
ax.text(x0 + 2.6, y1 - r + 0.3, r"Tube radius = $R_J$", ha="center", va="top", color="C0")

# Tube 2
ax.plot(x0 + r*np.cos(theta), y2 + r*np.sin(theta), color="blue")

# Magnetic Field (azimuthal)
add_curved_B(ax, x0, y1, r, color="C4", flip=False)
add_curved_B(ax, x0, y2, r, color="C4", flip=True)

ax.text(x0 + 2.2, 1.2, r"$B_\theta(r) \propto J_0(\alpha r)$", ha="left", va="center", color="C4")

# Bz floor (c_param)
ax.plot(x0 + 2., 1.525, marker=r'$\odot$', markersize=18, color="C3")
ax.text(x0 + 2.2, 1.4, r"$B_z \sim \sqrt{J_0^2 + c_{\mathrm{param}}}$", ha="left", va="bottom", color="C3")

# Currents
ax.plot(x0, y1, marker=r'$\otimes$', markersize=18, color="C5")       # lower tube: Jz > 0 (out of plane)
ax.plot(x0, y2, marker=r'$\odot$', markersize=18, color="C5")     # upper tube: Jz < 0 (into plane)

ax.text(x0 + 0.4, y1, r"$J_z$", ha="left", va="center", color="C5")
ax.text(x0 + 0.4, y2, r"$-J_z$",ha="left", va="center", color="C5")

# E = -v x B
ax.annotate("", (x0 - 1.5, 2.1), (x0 - 1.5, 1.0), arrowprops=dict(arrowstyle="<|-", color="C6"))
ax.annotate("", (x0 - 1.5, -2.0), (x0 - 1.5, -1.0), arrowprops=dict(arrowstyle="<|-", color="C6"))
ax.text(x0 - 2, y1, r"$\beta_{\mathrm{kick}}$", ha="left", va="center", color="C6")
ax.text(x0 - 2, y2, r"$\beta_{\mathrm{kick}}$", ha="left", va="center", color="C6")

ax.text(x0 - 3, 0, r"$\mathbf{E} = -\mathbf{v}\times\mathbf{B}$", ha="left", va="center", color="C6")

ax.text(0.25, 2.5, "Boundaries X: Open/Absorb", ha="left", va="center", color="green")
ax.text(0.25, 2.2, "Boundaries Y: Open/Absorb", ha="left", va="center", color="green")
ax.text(0.25, 1.9, "Boundaries Z: Periodic", ha="left", va="center", color="green")

# XYZ axes (bottom-right corner)
origin = (0.8, -Ly/2 + 0.8)

# X axis
ax.annotate("", (origin[0] + 0.8, origin[1]), origin, arrowprops=dict(arrowstyle="->", color="k"))
ax.text(origin[0] + 0.9, origin[1], r"$X$", ha="left", va="center")

# Y axis
ax.annotate("", (origin[0], origin[1] + 0.8), origin, arrowprops=dict(arrowstyle="->", color="k"))
ax.text(origin[0], origin[1] + 0.9, r"$Y$", ha="center", va="bottom")

# Z axis (out of plane)
ax.plot(origin[0], origin[1], marker=r'$\odot$', markersize=8, color='k')
ax.text(origin[0] - 0.2, origin[1] - 0.2, r"$Z$", ha="right", va="top")

ax.set(xticks=[], yticks=[])
plt.savefig("sketch.png", bbox_inches="tight")