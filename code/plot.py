import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, norm


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


def plot_contour_fields(q1, q2, title):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    
    K = q1.shape[0]
    x = np.linspace(-np.pi, np.pi, K)
    y = np.linspace(-np.pi, np.pi, K)
    X, Y = np.meshgrid(x, y)
    
    contour1 = axes[0].contourf(X, Y, q1, levels=128, cmap='seismic')
    axes[0].set_title('upper layer PV: '+title)
    fig.colorbar(contour1, ax=axes[0])
    
    contour2 = axes[1].contourf(X, Y, q2, levels=128, cmap='seismic')
    axes[1].set_title('lower layer PV: '+title)
    fig.colorbar(contour2, ax=axes[1])
    
    plt.tight_layout()


def plot_psi_k_seriespdf(dt, sel0, sel1, ikx, iky, interv, xlim, ylim, xt, yt, psi1_k, psi2_k, labels):
	colors = ['k', 'r', 'c', 'b']
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

	ax2.set_ylabel(r'$Re(\hat{{\psi}}_{{\psi,({:d},{:d})}})$'.format(iky, ikx))
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


def plot_layer_seriespdf(dt, sel0, sel1, ikx, iky, interv, xlim, ylim, psi1, psi2, labels):
	colors = ['k', 'r', 'c', 'b']
	xaxis = np.arange(sel0*dt, sel1*dt, interv*dt)

	fig = plt.figure(figsize=(10,4))
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
		ax3.plot(xticks, p, colors[i], label=labels[i])

	ax1.set_ylabel(r'$\psi_{{1,({:d},{:d})}}$'.format(ikx, iky))
	ax1.set_xlim(xlim)

	samples = psi1[0][iky, ikx, sel0:sel1]
	mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
	gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
	ax3.plot(xticks, gaussian_pdf, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
	ax3.set_yscale('log', base=10) 
	ax3.set_title('PDF')

	# plot time series
	for i, data in enumerate(psi2):
		ax2.plot(xaxis, data[iky,ikx,sel0:sel1:interv], colors[i], label=labels[i])

		samples = data[iky, ikx, sel0:sel1]
		kde = gaussian_kde(samples)
		xticks = np.linspace(samples.min(), samples.max(), 100)
		p = kde.evaluate(xticks)
		ax4.plot(xticks, p, colors[i], label=labels[i])

	ax2.set_ylabel(r'$\psi_{{2,({:d},{:d})}}$'.format(ikx, iky))
	ax2.set_xlim(xlim)
	ax2.legend(prop={'size': 8})

	samples = psi2[0][iky, ikx, sel0:sel1]
	mean, std = samples.mean(), samples.std() # Fit a Gaussian to the same data
	gaussian_pdf = norm.pdf(xticks, mean, std)  # Calculate the Gaussian PDF
	ax4.plot(xticks, gaussian_pdf, 'k--', label='Fitted Gaussian')  # Dashed line for Gaussian
	ax4.set_yscale('log', base=10) 
	ax4.set_title('PDF')
	ax4.set_ylim(ylim[0], np.max(p)+ylim[1])

