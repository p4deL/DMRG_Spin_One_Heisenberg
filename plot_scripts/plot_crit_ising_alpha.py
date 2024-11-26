import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.rcParams['text.usetex'] = True

fs = 18
fs2 = 12

def hex_to_RGB(hex_str):
  """ #FFFFFF -> [255,255,255]"""
  #Pass 16 to the integer function for change of base
  return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

colors = get_color_gradient("#ff006e", "#3a86ff", 5)


def plot_heisenberg_chain():

    fig = plt.figure(figsize = (5.51,5.51))
    #matplotlib.figure.SubplotParams(left=0.0,right=1.0,bottom=0.5,top=1.0)

    gs = GridSpec(4, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax1b = fig.add_subplot(gs[2, 0])
    ax2b = fig.add_subplot(gs[2, 1])
    ax3b = fig.add_subplot(gs[3, 0])
    

    #ax1.set_position([0.11,0.65,0.85,0.3],which='both')
    ax1.set_position([0.11,0.65,0.82,0.3],which='both')
    ax1b.set_position([0.94,0.65,0.02,0.3],which='both')
    ax1b.set_yticks([])
    ax1b.set_xticks([])
    #ax1b.set_xticklabels([1000000],labels=['$\\infty$'])
    #ax1b.set_xticklabels([1000000])
    #ax1b.set_xticklabels([999999,'$\\infty$'])
    ax1b.text(999998, -0.0645, '$\\infty$')
    ax1.spines['right'].set_visible(False)
    ax1b.spines['left'].set_visible(False)

    kwa = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1)
    d = .005  # how big to make the diagonal lines in axes coordinatess
    s=5
    ax1.plot((1+d, 1-d), (+d*s, -d*s), **kwa)
    ax1.plot((1+d, 1-d), (1+d*s, 1-d*s), **kwa)

    kwa = dict(transform=ax1b.transAxes, color='k', clip_on=False, lw=1)
    d = 0.215
    s = 0.125
    ax1b.plot((+d, -d), (+d*s, -d*s), **kwa)
    ax1b.plot((+d, -d), (1+d*s, 1-d*s), **kwa)
    
    #ax2.set_position([0.11,0.20,0.32,0.35],which='both')
    ax2.set_position([0.11,0.20,0.32,0.35],which='both')
    ax2b.set_position([0.44,0.20,0.02,0.35],which='both')
    ax2b.set_yticks([])
    ax2b.set_xticks([])
    ax2b.text(999998, 0.449, '$\\infty$')
    ax2.spines['right'].set_visible(False)
    ax2b.spines['left'].set_visible(False)
    

    kwa = dict(transform=ax2.transAxes, color='k', clip_on=False, lw=1)
    d = 0.015
    s = 1.8
    ax2.plot((1+d, 1-d), (+d*s, -d*s), **kwa)
    ax2.plot((1+d, 1-d), (1+d*s, 1-d*s), **kwa)

    kwa = dict(transform=ax2b.transAxes, color='k', clip_on=False, lw=1)
    d = 0.215
    s = 0.125
    ax2b.plot((+d, -d), (+d*s, -d*s), **kwa)
    ax2b.plot((+d, -d), (1+d*s, 1-d*s), **kwa)
    
    
    #ax3.set_position([0.64,0.20,0.32,0.35],which='both')
    ax3.set_position([0.61,0.20,0.32,0.35],which='both')
    ax3b.set_position([0.94,0.20,0.02,0.35],which='both')
    ax3b.set_yticks([])
    ax3b.set_xticks([])
    ax3b.text(999998, 0.446, '$\\infty$')
    ax3.spines['right'].set_visible(False)
    ax3b.spines['left'].set_visible(False)

    kwa = dict(transform=ax3.transAxes, color='k', clip_on=False, lw=1)
    d = 0.015
    s = 1.8
    ax3.plot((1+d, 1-d), (+d*s, -d*s), **kwa)
    ax3.plot((1+d, 1-d), (1+d*s, 1-d*s), **kwa)

    kwa = dict(transform=ax3b.transAxes, color='k', clip_on=False, lw=1)
    d = 0.215
    s = 0.125
    ax3b.plot((+d, -d), (+d*s, -d*s), **kwa)
    ax3b.plot((+d, -d), (1+d*s, 1-d*s), **kwa)


    #koziol2021      = ('red', 'x')
    #koziol2021_only = ('lightcoral', 'x')
    #fey2016         = ('green', '+')
    #koffel2012      = ('cornflowerblue', '.')
    #sun2017         = ('orange','d',2.0)
    #vodola2016      = ('violet','.',1.0)

    theta0_qpt = (colors[0], '+-', 7, 1)
    theta2pi16_qpt = (colors[1], 'x-', 5, 1)
    theta4pi16_qpt = (colors[2], '.-', 6, 1)
    theta6pi16_qpt = (colors[3], 'D-', 3.5, 1)
    theta8pi16_qpt = (colors[4], 's-', 3.5, 1)

    theta0 = (colors[0], '+', 7)
    theta2pi16 = (colors[1], 'x', 5)
    theta4pi16 = (colors[2], '.', 6)
    theta6pi16 = (colors[3], 'D', 3.0)
    theta8pi16 = (colors[4], 's', 3.0)

    gap_theta0 = np.loadtxt('data/xxz_bilayer/xxz_bilayer_theta0_gap.csv',skiprows=1,delimiter=',')
    gap_theta2pi16 = np.loadtxt('data/xxz_bilayer/xxz_bilayer_theta2pi16_gap.csv',skiprows=1,delimiter=',')
    gap_theta4pi16 = np.loadtxt('data/xxz_bilayer/xxz_bilayer_theta4pi16_gap.csv',skiprows=1,delimiter=',')
    gap_theta6pi16 = np.loadtxt('data/xxz_bilayer/xxz_bilayer_theta6pi16_gap.csv',skiprows=1,delimiter=',')
    gap_theta8pi16 = np.loadtxt('data/xxz_bilayer/xxz_bilayer_theta8pi16_gap.csv',skiprows=1,delimiter=',')
    
    sw_theta0 = np.loadtxt('data/xxz_bilayer/xxz_bilayer_theta0_sw.csv',skiprows=1,delimiter=',')
    sw_theta2pi16 = np.loadtxt('data/xxz_bilayer/xxz_bilayer_theta2pi16_sw.csv',skiprows=1,delimiter=',')
    sw_theta4pi16 = np.loadtxt('data/xxz_bilayer/xxz_bilayer_theta4pi16_sw.csv',skiprows=1,delimiter=',')
    sw_theta6pi16 = np.loadtxt('data/xxz_bilayer/xxz_bilayer_theta6pi16_sw.csv',skiprows=1,delimiter=',')
    sw_theta8pi16 = np.loadtxt('data/xxz_bilayer/xxz_bilayer_theta8pi16_sw.csv',skiprows=1,delimiter=',')


    ax1.errorbar(gap_theta0[:,0], gap_theta0[:,1],yerr=gap_theta0[:,2],color=theta0_qpt[0],fmt=theta0_qpt[1],ms=theta0_qpt[2],lw=theta0_qpt[3])
    ax1.errorbar(gap_theta2pi16[:,0], gap_theta2pi16[:,1],yerr=gap_theta2pi16[:,2],color=theta2pi16_qpt[0],fmt=theta2pi16_qpt[1],ms=theta2pi16_qpt[2],lw=theta2pi16_qpt[3])
    ax1.errorbar(gap_theta4pi16[:,0], gap_theta4pi16[:,1],yerr=gap_theta4pi16[:,2],color=theta4pi16_qpt[0],fmt=theta4pi16_qpt[1],ms=theta4pi16_qpt[2],lw=theta4pi16_qpt[3])
    ax1.errorbar(gap_theta6pi16[:,0], gap_theta6pi16[:,1],yerr=gap_theta6pi16[:,2],color=theta6pi16_qpt[0],fmt=theta6pi16_qpt[1],ms=theta6pi16_qpt[2],lw=theta6pi16_qpt[3])
    ax1.errorbar(gap_theta8pi16[:,0], gap_theta8pi16[:,1],yerr=gap_theta8pi16[:,2],color=theta8pi16_qpt[0],fmt=theta8pi16_qpt[1],ms=theta8pi16_qpt[2],lw=theta8pi16_qpt[3])
    
    ax1.set_xlabel('$\\sigma$', fontsize=fs)
    ax1.set_ylabel('$\\lambda_c$', fontsize=fs)
    ax1.set_xlim([0.,6.2])
    ax1.set_ylim([0.,0.6])
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.1))
    
    ax1b.errorbar(gap_theta0[-3:,0], gap_theta0[-3:,1],yerr=gap_theta0[-3:,2],color=theta0[0],ms=theta0_qpt[2],lw=theta0_qpt[3],fmt=theta0[1])
    ax1b.errorbar(gap_theta2pi16[-3:,0], gap_theta2pi16[-3:,1],yerr=gap_theta2pi16[-3:,2],color=theta2pi16[0],ms=theta2pi16_qpt[2],lw=theta2pi16_qpt[3],fmt=theta2pi16[1])
    ax1b.errorbar(gap_theta4pi16[-3:,0], gap_theta4pi16[-3:,1],yerr=gap_theta4pi16[-3:,2],color=theta4pi16[0],ms=theta4pi16_qpt[2],lw=theta4pi16_qpt[3],fmt=theta4pi16[1])
    ax1b.errorbar(gap_theta6pi16[-3:,0], gap_theta6pi16[-3:,1],yerr=gap_theta6pi16[-3:,2],color=theta6pi16[0],ms=theta6pi16_qpt[2],lw=theta6pi16_qpt[3],fmt=theta6pi16[1])
    ax1b.errorbar(gap_theta8pi16[-3:,0], gap_theta8pi16[-3:,1],yerr=gap_theta8pi16[-3:,2],color=theta8pi16[0],ms=theta8pi16_qpt[2],lw=theta8pi16_qpt[3],fmt=theta8pi16[1])
    
    ax1b.set_xlim([999998,1000000])
    ax1b.set_ylim([0.,0.6])


    ax1.text(0.05, 0.8, "antiferromagnetic phases", fontsize=10, color='black', transform=ax1.transAxes)
    ax1.text(0.75, 0.2, "rung-singlet phase", fontsize=10, color='black', transform=ax1.transAxes)

    ax2.errorbar(gap_theta0[:-1,0], gap_theta0[:-1,3],yerr=gap_theta0[:-1,4],color=theta0[0],fmt=theta0[1], ms=theta0[2], label='$\\theta=0$\\,\scriptsize{(XY)}')
    ax2.errorbar(gap_theta2pi16[:-1,0], gap_theta2pi16[:-1,3],yerr=gap_theta2pi16[:-1,4],color=theta2pi16[0],fmt=theta2pi16[1], ms=theta2pi16[2], label='$\\theta=\\frac{\pi}{8}$')
    ax2.errorbar(gap_theta4pi16[:-1,0], gap_theta4pi16[:-1,3],yerr=gap_theta4pi16[:-1,4],color=theta4pi16[0],fmt=theta4pi16[1], ms=theta4pi16[2], label='$\\theta=\\frac{\pi}{4}$\\,\scriptsize{(Heisenberg)}')
    ax2.errorbar(gap_theta6pi16[:-1,0], gap_theta6pi16[:-1,3],yerr=gap_theta6pi16[:-1,4],color=theta6pi16[0],fmt=theta6pi16[1], ms=theta6pi16[2], label='$\\theta=\\frac{3\pi}{8}$')
    ax2.errorbar(gap_theta8pi16[:-1,0], gap_theta8pi16[:-1,3],yerr=gap_theta8pi16[:-1,4],color=theta8pi16[0],fmt=theta8pi16[1], ms=theta8pi16[2], label='$\\theta=\\frac{\pi}{2}$\\,\scriptsize{(Ising)}')
    ax2.set_xlabel('$\\sigma$', fontsize=fs)
    ax2.set_ylabel('$z\\nu$', fontsize=fs)
    ax2.set_xlim([0.,6.2])
    ax2.set_ylim([0.48,0.78])
    ax2.xaxis.set_major_locator(MultipleLocator(1))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.05))
    
    ax2b.errorbar(gap_theta0[-1,0], gap_theta0[-1,3],yerr=gap_theta0[-1,4],color=theta0[0],fmt=theta0[1], ms=theta0[2])
    ax2b.errorbar(gap_theta2pi16[-1,0], gap_theta2pi16[-1,3],yerr=gap_theta2pi16[-1,4],color=theta2pi16[0],fmt=theta2pi16[1], ms=theta2pi16[2])
    ax2b.errorbar(gap_theta4pi16[-1,0], gap_theta4pi16[-1,3],yerr=gap_theta4pi16[-1,4],color=theta4pi16[0],fmt=theta4pi16[1], ms=theta4pi16[2])
    ax2b.errorbar(gap_theta6pi16[-1,0], gap_theta6pi16[-1,3],yerr=gap_theta6pi16[-1,4],color=theta6pi16[0],fmt=theta6pi16[1], ms=theta6pi16[2])
    ax2b.errorbar(gap_theta8pi16[-1,0], gap_theta8pi16[-1,3],yerr=gap_theta8pi16[-1,4],color=theta8pi16[0],fmt=theta8pi16[1], ms=theta8pi16[2])
    
    ax2b.set_ylim([0.48,0.78])
    

    ax3.errorbar(sw_theta0[:-1,0], sw_theta0[:-1,3],yerr=sw_theta0[:-1,4],color=theta0[0],fmt=theta0[1], ms=theta0[2])
    ax3.errorbar(sw_theta2pi16[:-1,0], sw_theta2pi16[:-1,3],yerr=sw_theta2pi16[:-1,4],color=theta2pi16[0],fmt=theta2pi16[1], ms=theta2pi16[2])
    ax3.errorbar(sw_theta4pi16[:-1,0], sw_theta4pi16[:-1,3],yerr=sw_theta4pi16[:-1,4],color=theta4pi16[0],fmt=theta4pi16[1], ms=theta4pi16[2])
    ax3.errorbar(sw_theta6pi16[:-1,0], sw_theta6pi16[:-1,3],yerr=sw_theta6pi16[:-1,4],color=theta6pi16[0],fmt=theta6pi16[1], ms=theta6pi16[2])
    ax3.errorbar(sw_theta8pi16[:-1,0], sw_theta8pi16[:-1,3],yerr=sw_theta8pi16[:-1,4],color=theta8pi16[0],fmt=theta8pi16[1], ms=theta8pi16[2])
    ax3.set_xlabel('$\\sigma$', fontsize=fs)
    ax3.set_ylabel('$(2-z-\\eta)\\nu$', fontsize=fs)
    ax3.set_xlim([0.,6.2])
    ax3.set_ylim([0.47,0.74])
    #ax3.plot(sigmas,2*np.ones(200)-sigmas,color='black',linestyle='--')
    ax3.xaxis.set_major_locator(MultipleLocator(1))
    ax3.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax3.yaxis.set_major_locator(MultipleLocator(0.1))
    ax3.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax3b.errorbar(sw_theta0[-1,0], sw_theta0[-1,3],yerr=sw_theta0[-1,4],color=theta0[0],fmt=theta0[1], ms=theta0[2])
    ax3b.errorbar(sw_theta2pi16[-1,0], sw_theta2pi16[-1,3],yerr=sw_theta2pi16[-1,4],color=theta2pi16[0],fmt=theta2pi16[1], ms=theta2pi16[2])
    ax3b.errorbar(sw_theta4pi16[-1,0], sw_theta4pi16[-1,3],yerr=sw_theta4pi16[-1,4],color=theta4pi16[0],fmt=theta4pi16[1], ms=theta4pi16[2])
    ax3b.errorbar(sw_theta6pi16[-1,0], sw_theta6pi16[-1,3],yerr=sw_theta6pi16[-1,4],color=theta6pi16[0],fmt=theta6pi16[1], ms=theta6pi16[2])
    ax3b.errorbar(sw_theta8pi16[-1,0], sw_theta8pi16[-1,3],yerr=sw_theta8pi16[-1,4],color=theta8pi16[0],fmt=theta8pi16[1], ms=theta8pi16[2])

    ax3b.set_ylim([0.47,0.74])

    # plot lines
    xmf, ymf = [0.0, 1.3], [0.5, 0.5]
    ax2.plot(xmf, ymf, c='gray', linewidth=2)

    xnn_xx, ynn_xx = [1.965, 6.5], [0.67155, 0.67155]
    ax2.plot(xnn_xx, ynn_xx, c=colors[0], alpha=0.5, linewidth=2)
    ax2.text(5.1, 0.67955, r'$O(2)$', c=theta0[0], alpha=0.5, fontsize=8)
    xnn_xx, ynn_xx = [999998, 1000000], [0.67155, 0.67155]
    ax2b.plot(xnn_xx, ynn_xx, c=colors[0], alpha=0.5, linewidth=2)

    xnn_heisenberg, ynn_heisenberg = [1.965, 6.5], [0.7117, 0.7117]
    ax2.plot(xnn_heisenberg, ynn_heisenberg, c=colors[2], alpha=0.5, linewidth=2)
    ax2.text(5.1, 0.7197, r'$O(3)$', c=colors[2], alpha=0.5, fontsize=8)
    xnn_heisenberg, ynn_heisenberg = [999998, 1000000], [0.7117, 0.7117]
    ax2b.plot(xnn_heisenberg, ynn_heisenberg, c=colors[2], alpha=0.5, linewidth=2)

    xnn_ising, ynn_ising = [1.965, 6.5], [0.629971, 0.629971]
    ax2.plot(xnn_ising, ynn_ising, c=colors[4], alpha=0.5, linewidth=2)
    ax2.text(5.35, 0.634971, r'$Z_2$', c=colors[4], alpha=0.5, fontsize=8)
    xnn_ising, ynn_ising = [999998, 1000000], [0.629971, 0.629971]
    ax2b.plot(xnn_ising, ynn_ising, c=colors[4], alpha=0.5, linewidth=2)

    ax2.axvspan(0.0, 1.333, facecolor='#00b4d8', alpha=0.2)
    ax2.text(0.015, 0.5, "LRMF", fontsize=8, color='gray', transform=ax2.transAxes)
    ax2.axvspan(1.96, 6.2, facecolor='#00b4d8', alpha=0.2)
    ax2.text(0.6, 0.25, "NN", fontsize=8, color='gray', transform=ax2.transAxes)
    ax2b.axvspan(999998, 1000000, facecolor='#00b4d8', alpha=0.2)
    
    
    xmf, ymf = [0.0, 1.3], [0.5, 0.5]
    ax3.plot(xmf, ymf, c='gray', linewidth=2)

    xnn_xx, ynn_xx = [1.965, 6.5], [0.6460, 0.6460]
    ax3.plot(xnn_xx, ynn_xx, c=colors[0], alpha=0.5, linewidth=2)
    ax3.text(5.1, 0.6540, r'$O(2)$', c=theta0[0], alpha=0.5, fontsize=8)
    xnn_xx, ynn_xx = [999998, 1000000], [0.6460, 0.6460]
    ax3b.plot(xnn_xx, ynn_xx, c=colors[0], alpha=0.5, linewidth=2)

    xnn_heisenberg, ynn_heisenberg = [1.965, 6.5], [0.6847, 0.6847]
    ax3.plot(xnn_heisenberg, ynn_heisenberg, c=colors[2], alpha=0.5, linewidth=2)
    ax3.text(5.1, 0.6927, r'$O(3)$', c=colors[2], alpha=0.5, fontsize=8)
    xnn_heisenberg, ynn_heisenberg = [999998, 1000000], [0.6847, 0.6847]
    ax3b.plot(xnn_heisenberg, ynn_heisenberg, c=colors[2], alpha=0.5, linewidth=2)

    xnn_ising, ynn_ising = [1.965, 6.5], [0.607104, 0.607104]
    ax3.plot(xnn_ising, ynn_ising, c=colors[4], alpha=0.5, linewidth=2)
    ax3.text(5.35, 0.612104, r'$Z_2$', c=colors[4], alpha=0.5, fontsize=8)
    xnn_ising, ynn_ising = [999998, 1000000], [0.607104, 0.607104]
    ax3b.plot(xnn_ising, ynn_ising, c=colors[4], alpha=0.5, linewidth=2)

    ax3.axvspan(0.0, 1.333, facecolor='#00b4d8', alpha=0.2)
    ax3.text(0.015, 0.5, "LRMF", fontsize=8, color='gray', transform=ax3.transAxes)
    ax3.axvspan(1.96, 6.2, facecolor='#00b4d8', alpha=0.2)
    ax3.text(0.6, 0.25, "NN", fontsize=8, color='gray', transform=ax3.transAxes)
    ax3b.axvspan(999998, 1000000, facecolor='#00b4d8', alpha=0.2)


    #plt.tight_layout()
    fig.legend(loc='lower center',ncol=5, handletextpad=-0.2, columnspacing=0.5, fontsize=fs2)
    fig.savefig("output/xxz_bilayer_all_sigma.pdf")


if __name__ == '__main__':
    plot_heisenberg_chain()


