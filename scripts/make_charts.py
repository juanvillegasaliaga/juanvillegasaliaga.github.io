import numpy as np, matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

INK   = "#10202F"
AZURE = "#2663A7"
GOLD  = "#A9853C"
MIST  = "#8FA3B0"
FAINT = "#D8E0E4"

matplotlib.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": "none",
    "axes.labelcolor": INK,
    "xtick.color": "#5A6B77",
    "ytick.color": "#5A6B77",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "svg.fonttype": "none",
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
})

rng = np.random.default_rng(7)

# ---------- Chart 1: delta-neutral equity curve + drawdown ----------
n = 156  # weekly, 3 years
mu, sig = 0.0042, 0.011
r = rng.normal(mu, sig, n)
# a few stress weeks
for i in (34, 35, 88, 121): r[i] -= 0.028*rng.uniform(0.7,1.2)
eq = np.cumprod(1+r)
peak = np.maximum.accumulate(eq)
dd = eq/peak - 1
x = np.arange(n)/52

sharpe = (np.mean(r)/np.std(r))*np.sqrt(52)
maxdd = dd.min()

fig = plt.figure(figsize=(8.6, 3.6))
gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.12)
ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax)

ax.plot(x, eq, color=INK, lw=1.4)
ax.axhline(1.0, color=GOLD, lw=0.7)
ax.set_ylim(0.95, eq.max()*1.03)
ax.set_xlim(0, x[-1])
ax.grid(axis="y", color=FAINT, lw=0.5)
ax.tick_params(length=0)
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_yticks([1.0, 1.1, 1.2, 1.3])
ax.set_yticklabels(["1.00×","1.10×","1.20×","1.30×"])

ax2.fill_between(x, dd*100, 0, color=AZURE, alpha=0.28, lw=0)
ax2.plot(x, dd*100, color=AZURE, lw=0.8)
ax2.set_ylim(min(dd*100)*1.4, 0.5)
ax2.grid(axis="y", color=FAINT, lw=0.5)
ax2.tick_params(length=0)
ax2.set_yticks([0, -3, -6])
ax2.set_yticklabels(["0%","−3%","−6%"])
ax2.set_xticks([0,1,2,3])
ax2.set_xticklabels(["Y0","Y1","Y2","Y3"])
for a in (ax, ax2):
    for s in a.spines.values(): s.set_visible(False)
fig.savefig("assets/charts/delta_neutral.svg", bbox_inches="tight", pad_inches=0.05)
print(f"sharpe {sharpe:.2f}  maxdd {maxdd*100:.1f}%  total {(eq[-1]-1)*100:.1f}%")
plt.close(fig)

# ---------- Chart 2: macro regime composite + BTC allocation ----------
m = 200
t = np.arange(m)/52
stress = np.zeros(m)
level = 0.0
for i in range(m):
    level = 0.92*level + rng.normal(0, 0.35)
    stress[i] = level
# inject two stress episodes
stress[60:78]  += np.linspace(0, 2.8, 18)
stress[78:95]  += np.linspace(2.8, 0, 17)
stress[150:162]+= np.linspace(0, 2.2, 12)
stress[162:175]+= np.linspace(2.2, 0, 13)
z = (stress - stress.mean())/stress.std()

alloc = np.where(z < 0.4, 1.0, np.where(z < 1.2, 0.5, 0.15))
# smooth step changes slightly with hold logic
for i in range(1, m):
    if abs(alloc[i]-alloc[i-1]) > 0 and abs(z[i]-  (0.4 if alloc[i-1]==1.0 else 1.2)) < 0.08:
        alloc[i] = alloc[i-1]

fig = plt.figure(figsize=(8.6, 3.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.8, 1.2], hspace=0.14)
ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharex=ax)

ax.fill_between(t, z, 0, where=(z>1.2), color="#8C3A2E", alpha=0.18, lw=0)
ax.plot(t, z, color=INK, lw=1.1)
ax.axhline(1.2, color=GOLD, lw=0.7)
ax.axhline(0.4, color=GOLD, lw=0.5, alpha=0.6)
ax.set_xlim(0, t[-1]); ax.set_ylim(-2.2, 3.4)
ax.grid(axis="y", color=FAINT, lw=0.5)
ax.tick_params(length=0)
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_yticks([-2,-1,0,1,2,3])

ax2.step(t, alloc*100, where="post", color=AZURE, lw=1.3)
ax2.fill_between(t, alloc*100, 0, step="post", color=AZURE, alpha=0.14, lw=0)
ax2.set_ylim(0, 112)
ax2.set_yticks([0,50,100]); ax2.set_yticklabels(["0%","50%","100%"])
ax2.set_xticks([0,1,2,3]); ax2.set_xticklabels(["Y0","Y1","Y2","Y3"])
ax2.grid(axis="y", color=FAINT, lw=0.5)
ax2.tick_params(length=0)
for a in (ax, ax2):
    for s in a.spines.values(): s.set_visible(False)
fig.savefig("assets/charts/regime.svg", bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("charts done")
