import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, norm
from mode_truc import inv_truncate, truncate
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew, kurtosis
from matplotlib.animation import FuncAnimation, PillowWriter


def ifftnroll(K, q1_k_t, max_imag_lim=1e-8):
    # ifft to real space
    q1_t = np.fft.ifft2(q1_k_t, axes=(0,1))
    
    # check imaginary part
    max_imag_abs = np.max(np.abs(np.imag(q1_t)))
    if max_imag_abs > max_imag_lim:
        raise Exception("get significant imaginary parts, check ifft2")
    else:
        q1_t = np.real(q1_t)
    
    # shift domain from [0,2pi) to [-pi,pi)
    q1_t = np.roll(q1_t, shift=K//2, axis=0) 
    q1_t = np.roll(q1_t, shift=K//2, axis=1) 

    return q1_t


# @jit(nopython=True)
def loop_ifft2_var(K, KX, KY, xgrid, ygrid, var_real, var_imag, var):
    for jx in range(K):
        for jy in range(K):
            exp_cos = np.cos(KX[None,:] * xgrid[jx] + KY[None,:] * ygrid[jy])
            exp_sin = np.sin(KX[None,:] * xgrid[jx] + KY[None,:] * ygrid[jy])
            var_ = (exp_cos**2 @ var_real[:,None] + exp_sin**2 @ var_imag[:,None])[0,0] / K**4
            var[jy, jx] = np.real(var_)
    return var


def ifft2_var(K, var_real, var_imag, r_cut, style='circle'):
    '''
    var = 1/K^4 * (\sum_k{var_real_k * cos^2(kx) + var_imag_k * sin^2(kx)})
    '''
    xgrid = np.linspace(-np.pi,np.pi, K, endpoint=False)
    ygrid = np.linspace(-np.pi,np.pi, K, endpoint=False)
    Kx = np.fft.fftfreq(K) * K
    Ky = np.fft.fftfreq(K) * K
    KX, KY = np.meshgrid(Kx, Ky)
    KX = truncate(KX, r_cut, style)
    KY = truncate(KY, r_cut, style)
    var = np.zeros((K, K))

    var = loop_ifft2_var(K, KX, KY, xgrid, ygrid, var_real, var_imag, var)
    
    return var


def psi2q(psi1_k, psi2_k, kd, h_k):
    K = psi1_k.shape[0]
    
    # transpose for proper broadcasting
    psi1_k = np.transpose(psi1_k, axes=(2,0,1))
    psi2_k = np.transpose(psi2_k, axes=(2,0,1))
    
    Kx = np.fft.fftfreq(K) * K
    Ky = np.fft.fftfreq(K) * K
    KX, KY = np.meshgrid(Kx, Ky)   
    K_squared = KX**2 + KY**2
    q1_k = -(K_squared + kd**2/2) * psi1_k + kd**2/2 * psi2_k
    q2_k = -(K_squared + kd**2/2) * psi2_k + kd**2/2 * psi1_k + h_k

    # transpose back to shape (K, K, N)
    q1_k = np.transpose(q1_k, axes=(1,2,0))
    q2_k = np.transpose(q2_k, axes=(1,2,0))

    return q1_k, q2_k


def plot_contour_fields(q1, q2, title, colorlim=None):
	fig, axes = plt.subplots(1, 2, figsize=(7, 3))

	K = q1.shape[0]
	x = np.linspace(-np.pi, np.pi, K)
	y = np.linspace(-np.pi, np.pi, K)
	X, Y = np.meshgrid(x, y)

	if colorlim is None:
		vmax1 = np.max(abs(q1))
		vmax2 = np.max(abs(q2))
	else:
		vmax1, vmax2 = colorlim

	levels1 = np.linspace(-vmax1, vmax1, 257)
	levels2 = np.linspace(-vmax2, vmax2, 257)

	contour1 = axes[0].contourf(X, Y, q1, levels=levels1, cmap='seismic')
	axes[0].set_title('upper layer PV: '+title)
	fig.colorbar(contour1, ax=axes[0])

	contour2 = axes[1].contourf(X, Y, q2, levels=levels2, cmap='seismic')
	axes[1].set_title('lower layer PV: '+title)
	fig.colorbar(contour2, ax=axes[1])

	plt.tight_layout()
	

def plot_psi_k_seriespdf(dt, sel0, sel1, ikx, iky, interv, xlim, ylim, xt, yt, psi1_k, psi2_k, labels, colors):
	xaxis = np.arange(sel0*dt, sel1*dt, interv*dt)
	fig = plt.figure(figsize=(16,5))
	widths = [5, 1, 5, 1]
	heights = [2, 2, 2]
	spec = fig.add_gridspec(ncols=4, nrows=3, width_ratios=widths, height_ratios=heights)

	plt.subplots_adjust(wspace=0.35, hspace=0.5)     # Adjust the overall spacing of the figure
	ax1 = fig.add_subplot(spec[0, 0])
	ax2 = fig.add_subplot(spec[1, 0])
	ax3 = fig.add_subplot(spec[2, 0])
	ax4 = fig.add_subplot(spec[0, 1])
	ax5 = fig.add_subplot(spec[1, 1])
	ax6 = fig.add_subplot(spec[2, 1])
	ax11 = fig.add_subplot(spec[0, 2])
	ax22 = fig.add_subplot(spec[1, 2])
	ax33 = fig.add_subplot(spec[2, 2])
	ax44 = fig.add_subplot(spec[0, 3])
	ax55 = fig.add_subplot(spec[1, 3])
	ax66 = fig.add_subplot(spec[2, 3])

	# plot time series
	ax1.plot(xaxis, xt[0,sel0:sel1:interv], 'k',label='x')
	ax1.set_xlim(sel0*dt, sel1*dt)
	ax1.set_ylabel('x')
	ax1.set_title('time series')
	ax1.set_xlim(xlim)

	ax11.plot(xaxis, yt[0,sel0:sel1:interv], 'k')
	ax11.set_xlim(sel0*dt, sel1*dt)
	ax11.set_ylabel('y')
	ax11.set_title('time series')
	ax11.set_xlim(xlim)

	# plot pdf
	samples = xt[0, :]
	kde = gaussian_kde(samples)
	xticks = np.linspace(samples.min(), samples.max(), 100)
	p = kde.evaluate(xticks)
	ax4.plot(xticks, p, 'k')
	mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
	gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
	ax4.plot(xticks, gaussian_pdf, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
	ax4.set_title('PDF')
	ax4.set_yscale('log', base=10) 

	# plot pdf
	samples = yt[0, sel0:sel1]
	kde = gaussian_kde(samples)
	xticks = np.linspace(samples.min(), samples.max(), 100)
	p = kde.evaluate(xticks)
	ax44.plot(xticks, p, 'k')
	mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
	gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
	ax44.plot(xticks, gaussian_pdf, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
	ax44.set_title('PDF')
	ax44.set_yscale('log', base=10) 

	for i, data in enumerate(psi1_k):
		ax2.plot(xaxis, data[iky,ikx,sel0:sel1:interv].real, colors[i], label=labels[i])
		ax22.plot(xaxis, data[iky,ikx,sel0:sel1:interv].imag, colors[i], label=labels[i])

		samples = data[iky, ikx, sel0:sel1].real
		kde = gaussian_kde(samples)
		xticks = np.linspace(samples.min(), samples.max(), 100)
		p = kde.evaluate(xticks)
		ax5.plot(xticks, p, colors[i], label=labels[i])

		samples = data[iky, ikx, sel0:sel1].imag
		kde = gaussian_kde(samples)
		xticks = np.linspace(samples.min(), samples.max(), 100)
		p = kde.evaluate(xticks)
		ax55.plot(xticks, p, colors[i], label=labels[i])

	ax2.set_ylabel(r'$Re(\hat{{\psi}}_{{\psi,({:d},{:d})}})$'.format(ikx, iky))
	ax2.legend(prop={'size': 8})
	ax2.set_xlim(xlim)
	ax22.set_ylabel(r'$Im(\hat{{\psi}}_{{\psi,({:d},{:d})}})$'.format(ikx, iky))
	ax22.legend(prop={'size': 8})
	ax22.set_xlim(xlim)

	samples = psi1_k[0][iky, ikx, sel0:sel1].real
	mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
	gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
	ax5.plot(xticks, gaussian_pdf, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
	ax5.set_yscale('log', base=10) 
	ax5.set_ylim(ylim[0], np.max(p)+ylim[1])

	samples = psi1_k[0][iky, ikx, sel0:sel1].imag
	mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
	gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
	ax55.plot(xticks, gaussian_pdf, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
	ax55.set_yscale('log', base=10) 
	ax55.set_ylim(ylim[0], np.max(p)+ylim[1])

	for i, data in enumerate(psi2_k):
		ax3.plot(xaxis, data[iky,ikx,sel0:sel1:interv].real, colors[i], label=labels[i])
		ax33.plot(xaxis, data[iky,ikx,sel0:sel1:interv].imag, colors[i], label=labels[i])

		samples = data[iky, ikx, sel0:sel1].real
		kde = gaussian_kde(samples)
		xticks = np.linspace(samples.min(), samples.max(), 100)
		p = kde.evaluate(xticks)
		ax6.plot(xticks, p, colors[i], label=labels[i])

		samples = data[iky, ikx, sel0:sel1].imag
		kde = gaussian_kde(samples)
		xticks = np.linspace(samples.min(), samples.max(), 100)
		p = kde.evaluate(xticks)
		ax66.plot(xticks, p, colors[i], label=labels[i])

	ax3.set_ylabel(r'$Re(\hat{{\psi}}_{{\tau,({:d},{:d})}})$'.format(ikx, iky))
	ax3.set_xlabel('t')
	ax3.set_xlim(xlim)
	ax33.set_ylabel(r'$Im(\hat{{\psi}}_{{\tau,({:d},{:d})}})$'.format(ikx, iky))
	ax33.set_xlabel('t')
	ax33.set_xlim(xlim)

	samples = psi2_k[0][iky, ikx, sel0:sel1].real
	mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
	gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
	ax6.plot(xticks, gaussian_pdf, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
	ax6.set_ylim(ylim[0], np.max(p)+ylim[1])
	ax6.set_yscale('log', base=10) 

	samples = psi2_k[0][iky, ikx, sel0:sel1].imag
	mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
	gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
	ax66.plot(xticks, gaussian_pdf, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
	ax66.set_ylim(ylim[0], np.max(p)+ylim[1])
	ax66.set_yscale('log', base=10) 
	plt.tight_layout()


def plot_psi1_k_seriespdf(dt, sel0, sel1, ikx, iky, interv, xlim, ylim, xt, yt, psi1_k, psi2_k, labels, colors):
    xaxis = np.arange(sel0*dt, sel1*dt, interv*dt)
    fig = plt.figure(figsize=(16,3.2))
    widths = [5, 1, 5, 1]
    heights = [2, 2]
    spec = fig.add_gridspec(ncols=4, nrows=2, width_ratios=widths, height_ratios=heights)
    
    plt.subplots_adjust(wspace=0.35, hspace=0.5)     # Adjust the overall spacing of the figure
    # ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 0])
    ax3 = fig.add_subplot(spec[1, 0])
    # ax4 = fig.add_subplot(spec[0, 1])
    ax5 = fig.add_subplot(spec[0, 1])
    ax6 = fig.add_subplot(spec[1, 1])
    # ax11 = fig.add_subplot(spec[0, 2])
    ax22 = fig.add_subplot(spec[0, 2])
    ax33 = fig.add_subplot(spec[1, 2])
    # ax44 = fig.add_subplot(spec[0, 3])
    ax55 = fig.add_subplot(spec[0, 3])
    ax66 = fig.add_subplot(spec[1, 3])
    
    # # plot time series
    # ax1.plot(xaxis, xt[0,sel0:sel1:interv], 'k',label='x')
    # ax1.set_xlim(sel0*dt, sel1*dt)
    # ax1.set_ylabel('x')
    # ax1.set_title('time series')
    # ax1.set_xlim(xlim)
    
    # ax11.plot(xaxis, yt[0,sel0:sel1:interv], 'k')
    # ax11.set_xlim(sel0*dt, sel1*dt)
    # ax11.set_ylabel('y')
    # ax11.set_title('time series')
    # ax11.set_xlim(xlim)
    
    # # plot pdf
    # samples = xt[0, :]
    # kde = gaussian_kde(samples)
    # xticks = np.linspace(samples.min(), samples.max(), 100)
    # p = kde.evaluate(xticks)
    # ax4.plot(xticks, p, 'k')
    # mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
    # gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
    # ax4.plot(xticks, gaussian_pdf, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
    # ax4.set_title('PDF')
    # ax4.set_yscale('log', base=10) 
    
    # # plot pdf
    # samples = yt[0, sel0:sel1]
    # kde = gaussian_kde(samples)
    # xticks = np.linspace(samples.min(), samples.max(), 100)
    # p = kde.evaluate(xticks)
    # ax44.plot(xticks, p, 'k')
    # mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
    # gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
    # ax44.plot(xticks, gaussian_pdf, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
    # ax44.set_title('PDF')
    # ax44.set_yscale('log', base=10) 
    
    for i, data in enumerate(psi1_k):
        ax2.plot(xaxis, data[iky,ikx,sel0:sel1:interv].real, colors[i], label=labels[i])
        ax22.plot(xaxis, data[iky,ikx,sel0:sel1:interv].imag, colors[i], label=labels[i])
    
        samples = data[iky, ikx, sel0:sel1].real
        kde = gaussian_kde(samples)
        xticks = np.linspace(samples.min(), samples.max(), 100)
        p = kde.evaluate(xticks)
        ax5.plot(p, xticks, colors[i], label=labels[i])
    
        samples = data[iky, ikx, sel0:sel1].imag
        kde = gaussian_kde(samples)
        xticks = np.linspace(samples.min(), samples.max(), 100)
        p = kde.evaluate(xticks)
        ax55.plot(p, xticks, colors[i], label=labels[i])
    
    ax2.set_ylabel(r'$Re(\hat{{\psi}}_{{1,({:d},{:d})}})$'.format(ikx, iky))
    ax2.set_xlim(xlim)
    ax22.set_ylabel(r'$Im(\hat{{\psi}}_{{1,({:d},{:d})}})$'.format(ikx, iky))
    ax22.set_xlim(xlim)
    
    samples = psi1_k[0][iky, ikx, sel0:sel1].real
    mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
    gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
    ax5.plot(gaussian_pdf, xticks, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
    ax5.set_xscale('log', base=10) 
    ax5.set_xlim(ylim[0], np.max(p)+ylim[1])
    
    samples = psi1_k[0][iky, ikx, sel0:sel1].imag
    mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
    gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
    ax55.plot(gaussian_pdf, xticks, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
    ax55.set_xscale('log', base=10) 
    ax55.set_xlim(ylim[0], np.max(p)+ylim[1])

    for i, data in enumerate(psi2_k):
        ax3.plot(xaxis, data[iky,ikx,sel0:sel1:interv].real, colors[i], label=labels[i])
        ax33.plot(xaxis, data[iky,ikx,sel0:sel1:interv].imag, colors[i], label=labels[i])
    
        samples = data[iky, ikx, sel0:sel1].real
        kde = gaussian_kde(samples)
        xticks = np.linspace(samples.min(), samples.max(), 100)
        p = kde.evaluate(xticks)
        ax6.plot(p, xticks, colors[i], label=labels[i])
    
        samples = data[iky, ikx, sel0:sel1].imag
        kde = gaussian_kde(samples)
        xticks = np.linspace(samples.min(), samples.max(), 100)
        p = kde.evaluate(xticks)
        ax66.plot(p, xticks, colors[i], label=labels[i])

    ax3.set_ylabel(r'$Re(\hat{{\psi}}_{{2,({:d},{:d})}})$'.format(ikx, iky))
    ax3.set_xlabel('t')
    ax3.set_xlim(xlim)
    ax33.set_ylabel(r'$Im(\hat{{\psi}}_{{2,({:d},{:d})}})$'.format(ikx, iky))
    ax33.set_xlabel('t')
    ax33.set_xlim(xlim)
    
    samples = psi2_k[0][iky, ikx, sel0:sel1].real
    mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
    gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
    ax6.plot(gaussian_pdf, xticks, 'k--', label='Gaussian fit')  # Dashed line for Gaussian
    ax6.set_xlim(ylim[0], np.max(p)+ylim[1])
    ax6.set_xscale('log', base=10) 
    
    samples = psi2_k[0][iky, ikx, sel0:sel1].imag
    mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
    gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
    ax66.plot(gaussian_pdf, xticks, 'k--', label='Gaussian fit')  # Dashed line for Gaussian
    ax66.set_xlim(ylim[0], np.max(p)+ylim[1])
    ax66.set_xscale('log', base=10) 

    ax2.set_title('Time series')
    ax22.set_title('Time series')
    ax5.set_title('log PDF')
    ax55.set_title('log PDF')

    plt.legend(prop={'size': 8}, bbox_to_anchor=(1, 1))
    plt.tight_layout()


def plot_layer_seriespdf(dt, sel0, sel1, ikx, iky, interv, xlim, ylim, psi1, psi2, labels, colors):
    xaxis = np.arange(sel0*dt, sel1*dt, interv*dt)
    
    fig = plt.figure(figsize=(9,3.5))
    widths = [5, 1]
    heights = [1, 1]
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
    
    plt.subplots_adjust(wspace=0.35, hspace=0.5)     # Adjust the overall spacing of the figure
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0])
    ax3 = fig.add_subplot(spec[0, 1])
    ax4 = fig.add_subplot(spec[1, 1])
    
    # plot time series
    for i, data in enumerate(psi1):
        ax1.plot(xaxis, data[iky,ikx,sel0:sel1:interv], colors[i], label=labels[i])
    
        samples = data[iky, ikx, sel0:sel1]
        kde = gaussian_kde(samples)
        xticks = np.linspace(samples.min(), samples.max(), 100)
        p = kde.evaluate(xticks)
        ax3.plot(p, xticks, colors[i], label=labels[i])
    
    ax1.set_ylabel(r'$\psi_{{1,({:d},{:d})}}$'.format(ikx, iky))
    ax1.set_xlim(xlim)
    ax1.set_title('Time series')
    
    samples = psi1[0][iky, ikx, sel0:sel1]
    mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
    gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
    ax3.plot(gaussian_pdf, xticks, 'k--', label='Gaussian fit')  # Dashed line for Gaussian
    ax3.set_xscale('log', base=10) 
    ax3.set_title('log PDF')
    
    # plot time series
    for i, data in enumerate(psi2):
        ax2.plot(xaxis, data[iky,ikx,sel0:sel1:interv], colors[i], label=labels[i])
    
        samples = data[iky, ikx, sel0:sel1]
        kde = gaussian_kde(samples)
        xticks = np.linspace(samples.min(), samples.max(), 100)
        p = kde.evaluate(xticks)
        ax4.plot(p, xticks, colors[i], label=labels[i])
    
    ax2.set_ylabel(r'$\psi_{{2,({:d},{:d})}}$'.format(ikx, iky))
    ax2.set_xlim(xlim)
    ax2.set_xlabel('t')
    
    samples = psi2[0][iky, ikx, sel0:sel1]
    mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
    gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
    ax4.plot(gaussian_pdf, xticks, 'k--', label='Gaussian fit')  # Dashed line for Gaussian
    ax4.set_xscale('log', base=10) 
    ax4.set_xlim(ylim[0], np.max(p)+ylim[1])
    
    plt.legend(prop={'size': 8}, bbox_to_anchor=(1, 1))
    plt.tight_layout()


def plot_rmses(dt, sel0, sel1, s_rate, interv, xlim, data1, data2, labels, colors):
	xaxis = np.arange(sel0*(s_rate)*dt, sel1*(s_rate)*dt, interv*s_rate*dt)

	fig = plt.figure(figsize=(8,4))
	widths = [7]
	heights = [1, 1]
	spec = fig.add_gridspec(ncols=1, nrows=2, width_ratios=widths, height_ratios=heights)

	plt.subplots_adjust(wspace=0.35, hspace=0.5)     # Adjust the overall spacing of the figure
	ax1 = fig.add_subplot(spec[0, 0])
	ax2 = fig.add_subplot(spec[1, 0])

	# plot time series
	for i, data in enumerate(data1):
		ax1.plot(xaxis, data[sel0:sel1:interv], colors[i], label=labels[i])

	ax1.set_xlim(sel0*(s_rate)*dt, sel1*(s_rate)*dt)
	ax1.set_ylabel(r'$RMSE\ \psi_1$')
	ax1.set_xlim(xlim)

	# plot time series
	for i, data in enumerate(data2):
		ax2.plot(xaxis, data[sel0:sel1:interv], colors[i], label=labels[i])

	ax2.set_xlim(sel0*(s_rate)*dt, sel1*(s_rate)*dt)
	ax2.set_ylabel(r'$RMSE\ \psi_2$')
	ax2.set_xlabel('t')
	ax2.set_xlim(xlim)
	ax2.legend(prop={'size': 8})
	plt.tight_layout()


def plot_mog(ix, iy, N_s, K, psi2_pos_cg, psi2_pos_lsm, R_psi2_pos_cg, R_psi2_pos_lsm, psi2_t, xlim, smoothing_factor=None, figsize=(4.5,4), smoother='gaussian_filter1d'):    
    # Means and variances for the Gaussian components
    means = psi2_pos_cg[:, iy, ix]
    variances = R_psi2_pos_cg[:, iy, ix]
    std_devs = np.sqrt(variances)
    lsm_mean = psi2_pos_lsm[iy,ix]
    lsm_std = np.sqrt(R_psi2_pos_lsm[iy,ix])
    truth = psi2_t[iy, ix]

    # Initialize the mixture PDF
    x = np.linspace(xlim[0], xlim[1], 1000)
    lsm_pdf = norm.pdf(x, lsm_mean, lsm_std)
    mixture_pdf = np.zeros_like(x)
    mixture_mean = np.mean(means)
    mixture_variance = np.mean(variances) + np.var(means)
    mixture_std_dev = np.sqrt(mixture_variance)
    
    for i in range(N_s):
        # Compute the PDF for the current Gaussian component
        pdf = norm.pdf(x, means[i], std_devs[i])
        mixture_pdf += pdf / N_s  # Equal weights for each component
        # # Plot the individual Gaussian component
        # plt.plot(x, pdf, '--', label=f'Gaussian {i+1}: mean={means[i]:.2f}, std={std_devs[i]:.2f}')
    
    # Apply a smoothing kernel (Gaussian smoothing)
    if smoother == 'gaussian_filter1d':
        mixture_pdf = gaussian_filter1d(mixture_pdf, sigma=smoothing_factor)
    if smoother == 'gaussian_kde':
        mixture_pdf_ = gaussian_kde(x, bw_method=smoothing_factor, weights=mixture_pdf)
        mixture_pdf = mixture_pdf_(x)

    hgt = max(np.max(mixture_pdf), np.max(lsm_pdf))
    
    fig = plt.figure(figsize=figsize)
    truthline = np.linspace(0,hgt,num=2)
    plt.plot(np.array([truth, truth]), truthline,'k',linewidth=2, label='truth')
    plt.plot(x, lsm_pdf, label='one-step', color='red', linewidth=2)
    plt.plot(x, mixture_pdf, label='multi-step', color='blue', linewidth=2)
    # fitted_pdf = norm.pdf(x, mixture_mean, mixture_std_dev)
    # plt.plot(x, fitted_pdf, 'b--', label=f'Fitted Gaussian', linewidth=2)
    plt.title('x={:.2f},y={:.2f}'.format(-np.pi+ix/K*np.pi*2, -np.pi+iy/K*np.pi*2))
    plt.xlabel(r'$\psi_2$')
    plt.ylabel('PDF')
    plt.xlim(xlim)
    plt.legend(prop={'size': 9})



def plot_mog_k(ix, iy, N_s, K, psi2_pos_cg, psi2_pos_lsm, R_psi2_pos_cg, R_psi2_pos_lsm, psi2_t, xlim, smoothing_sigma=10, figsize=(4.5,4)):    
    # Means and variances for the Gaussian components
    means = psi2_pos_cg[:, iy, ix]
    variances = R_psi2_pos_cg[:, iy, ix]
    std_devs = np.sqrt(variances)
    lsm_mean = psi2_pos_lsm[iy,ix]
    lsm_std = np.sqrt(R_psi2_pos_lsm[iy,ix])
    truth = psi2_t[iy, ix]

    # Initialize the mixture PDF
    x = np.linspace(xlim[0], xlim[1], 1000)
    lsm_pdf = norm.pdf(x, lsm_mean, lsm_std)
    mixture_pdf = np.zeros_like(x)
    mixture_mean = np.mean(means)
    mixture_variance = np.mean(variances) + np.var(means)
    mixture_std_dev = np.sqrt(mixture_variance)
    
    for i in range(N_s):
        # Compute the PDF for the current Gaussian component
        pdf = norm.pdf(x, means[i], std_devs[i])
        mixture_pdf += pdf / N_s  # Equal weights for each component
        # # Plot the individual Gaussian component
        # plt.plot(x, pdf, '--', label=f'Gaussian {i+1}: mean={means[i]:.2f}, std={std_devs[i]:.2f}')
    
    # Apply a smoothing kernel (Gaussian smoothing)
    mixture_pdf = gaussian_filter1d(mixture_pdf, sigma=smoothing_sigma)

    hgt = max(np.max(mixture_pdf), np.max(lsm_pdf))
    
    fig = plt.figure(figsize=figsize)
    truthline = np.linspace(0,hgt,num=2)
    plt.plot(np.array([truth, truth]), truthline,'k',linewidth=2, label='truth')
    plt.plot(x, lsm_pdf, label=f'one-step', color='red', linewidth=2)
    plt.plot(x, mixture_pdf, label='multi-step', color='blue', linewidth=2)
    fitted_pdf = norm.pdf(x, mixture_mean, mixture_std_dev)
    plt.plot(x, fitted_pdf, 'b--', label=f'Fitted Gaussian', linewidth=2)
    plt.title('kx={:d},ky={:d}'.format(ix, iy))
    plt.xlabel(r'$\psi_2$')
    plt.ylabel('PDF')
    plt.xlim(xlim)
    plt.legend(prop={'size': 9})


def plot_mog_animation(N_s, K, psi2_pos_cg, psi2_pos_lsm, R_psi2_pos_cg, R_psi2_pos_lsm, psi2_t, xlim, smoothing_sigma=10, figsize=(4.5, 4), gif_name="mog_animation.gif"):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.linspace(xlim[0], xlim[1], 1000)

    def update(frame):
        # Calculate indices
        iy, ix = divmod(frame, 128)
        
        # Clear the previous frame
        ax.clear()
        
        # Means and variances for the Gaussian components
        means = psi2_pos_cg[:, iy, ix]
        variances = R_psi2_pos_cg[:, iy, ix]
        std_devs = np.sqrt(variances)
        lsm_mean = psi2_pos_lsm[iy, ix]
        lsm_std = np.sqrt(R_psi2_pos_lsm[iy, ix])
        truth = psi2_t[iy, ix]

        # LSM PDF
        lsm_pdf = norm.pdf(x, lsm_mean, lsm_std)

        # Initialize the mixture PDF
        mixture_pdf = np.zeros_like(x)
        for i in range(N_s):
            pdf = norm.pdf(x, means[i], std_devs[i])
            mixture_pdf += pdf / N_s  # Equal weights for each component

        # Apply smoothing
        mixture_pdf = gaussian_filter1d(mixture_pdf, sigma=smoothing_sigma)
        
        # Fitted Gaussian PDF
        mixture_mean = np.mean(means)
        mixture_variance = np.mean(variances) + np.var(means)
        mixture_std_dev = np.sqrt(mixture_variance)
        fitted_pdf = norm.pdf(x, mixture_mean, mixture_std_dev)

        # Plotting
        hgt = max(np.max(mixture_pdf), np.max(lsm_pdf))
        truthline = np.linspace(0, hgt, num=2)
        ax.plot([truth, truth], truthline, 'k', label='truth')
        ax.plot(x, lsm_pdf, label='one-step', color='red')
        ax.plot(x, mixture_pdf, label='multi-step', color='blue')
        ax.plot(x, fitted_pdf, 'b--', label='Fitted Gaussian')
        ax.set_title(f"ix={ix:d}, iy={iy:d}")
        ax.set_xlim(xlim)
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=128 * 128, repeat=False)
    writer = PillowWriter(fps=10)
    anim.save(gif_name, writer=writer)
    plt.close(fig)


def scatterplot(ix, iy, K, psi2_pos_cg, R_psi2_pos_cg, figsize=(5,5)):
    means = psi2_pos_cg[:, iy, ix]
    stds = np.sqrt(R_psi2_pos_cg[:, iy, ix])
    plt.figure(figsize=(5,5))
    plt.scatter(means, stds)
    plt.title('x={:.2f},y={:.2f}'.format(-np.pi+ix/K*np.pi*2, -np.pi+iy/K*np.pi*2))
    plt.xlabel('means')
    plt.ylabel('stds')


def calculate_skewness_kurtosis(ix, iy, N_s, K, psi2_pos_cg, R_psi2_pos_cg, num_samples=10000):
    # Means and variances for the Gaussian components
    means = psi2_pos_cg[:, iy, ix]
    variances = R_psi2_pos_cg[:, iy, ix]
    std_devs = np.sqrt(variances)
    
    # Sample data from the mixture distribution
    samples = []
    for i in range(N_s):
        # Generate samples for each Gaussian component
        component_samples = np.random.normal(means[i], std_devs[i], num_samples // N_s)
        samples.extend(component_samples)

    # Convert samples list to array for calculations
    samples = np.array(samples)

    # Calculate skewness and kurtosis from the sampled data
    skewness_value = skew(samples)
    kurtosis_value = kurtosis(samples)

    return skewness_value, kurtosis_value
