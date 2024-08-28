import numpy as np

def double_ft(A, w, q_array, dt=0.05):
    '''
    Goal perform double fourier transform on the time-dependent connected correlation function.

    Input: A
    '''
    num_steps, L = A.shape
    # eta = - np.log(0.5) / (num_steps * dt)
    # exp_jwt = np.exp( 1j * (w + 1j * eta) * np.arange(num_steps) * dt )
    eta = - np.log(0.1) / ((num_steps * dt)**2)
    # exp_jwt = np.cos(w * np.arange(num_steps) * dt )
    exp_jwt = np.exp( 1j * w * np.arange(num_steps) * dt ) * np.exp(- eta * ((np.arange(num_steps) * dt)**2))
    S_jw = exp_jwt.dot(A) * 2 * np.pi / num_steps
    S_jw = S_jw.reshape([L])

    # S_qw = np.zeros([L], dtype=np.complex)
    # for idx_x in range(L):
    #     qx = 2*np.pi * idx_x / L
    #     exp_jqx = np.exp( -1j * qx * ( np.arange(L) - L//2))
    #     S_qw[idx_x] = exp_jqx.dot(np.real(S_jw))

    q_xi = np.einsum('k,j->kj', q_array, np.arange(L) - L//2)
    exp_jqx = np.exp( -1j * q_xi)
    S_qw = exp_jqx.dot(S_jw)

    S_qw = (1. / L) * S_qw
    return S_qw

def get_S_qw(Stx, dt, omega=(-1, 20), dw=0.05):
    dw = np.amin([dw, dt])
    num_steps, L = Stx.shape

    S_array = []
    w_array = []
    num_omega = int((omega[1]-omega[0])/dw) + 1

    q_list = []
    q_list = [2*np.pi * idx_x / L for idx_x in range(L)]

    q_array = np.array(q_list)

    for idx_w in range(num_omega):
        w = omega[0] + dw * idx_w
        S_array.append(double_ft(Stx, w, q_array, dt))
        w_array.append(w)


    return np.array(S_array), q_array, np.array(w_array)

def get_spectral_function2(Ctx, dt):
    _, L = Ctx.shape
    print('Compute Fourier transform')
    Swk, momenta, freqs = get_S_qw(Ctx, dt=dt)
    momenta = ((momenta + np.pi) % (2*np.pi) ) - np.pi
    momenta = np.concatenate([momenta[(L+1)//2:], momenta[:(L+1)//2]], axis=0)
    Swk = np.concatenate([Swk[:, (L+1)//2:], Swk[:, :(L+1)//2]], axis=1)
    print('finished')

    return np.array(Swk), momenta, freqs


def fourier_space(x_series):
    """ Calculates the FFT of a spatial series of values. """
    import numpy as np
    ft = np.fft.fft(x_series)
    n = len(x_series)
    momenta = 2*np.pi * np.fft.fftfreq(n, 1)

    # order momenta in increasing order
    momenta = np.fft.fftshift(momenta)

    # shift results accordingly
    Ck = np.fft.fftshift(ft)

    # extend the results to the whole Brillouin zone (right border included)
    momenta = np.append(momenta, -momenta[0])
    Ck = np.append(Ck, Ck[0])

    return momenta, Ck

def fourier_time(t_series, dt, sigma = 0.4):
    """ Calculates the FFT of a time series, applying a Gaussian window function. """

    # Gaussian window function
    n = len(t_series)
    gauss = [np.exp(-1/2.*(i/(sigma * n))**2) for i in np.arange(n)]
    gauss = 1.
    input_series = gauss * t_series

    # Fourier transform
    ft = np.fft.fft(input_series)
    freqs = np.fft.fftfreq(n, dt) * 2 * np.pi

    # order frequencies in increasing order
    end = np.argmin(freqs)
    freqs = np.append(freqs[end:], freqs[:end])
    # shift results accordingly
    ftShifted = np.append(ft[end:], ft[:end])

    # Take into account the additional minus sign in the time FT
    if len(ftShifted)%2 == 0:
        ftShifted = np.append(ftShifted, ftShifted[0])
        ftShifted = ftShifted[::-1]
        ftShifted = ftShifted[:-1]
    else:
        ftShifted = ftShifted[::-1]

    return freqs, ftShifted

def get_correlation_function(L, J, g, h, dt, t_max, chi_max=30, eps=1e-10):
    E0, psi_0, model = get_groundstate(L, J, g, h, chi_max)

    print('-'*70)
    print('Compute dynamic correlations')
    Ctx, entanglement = compute_dynamic_correlations(psi_0, model, E0, dt, t_max, chi_max, eps)
    return Ctx, entanglement

## The np.fft implementation
def get_spectral_function1(Ctx, dt):
    # Rearrange corrs such that position 0 corresponds to the perturbed site
    # (distance 0 to perturbation)
    _, L = Ctx.shape
    xi = L//2
    c_temp = np.zeros(Ctx.shape, dtype=complex)
    c_temp[:, :L-xi] = Ctx[:, xi:]
    c_temp[:, L-xi:] = Ctx[:, :xi]
    Ctx = c_temp

    print('Compute Fourier transform')
    # Fourier transform in space
    corrs_tk = np.zeros((Ctx.shape[0], Ctx.shape[1]+1), dtype=complex)
    for i in np.arange(Ctx.shape[0]):
        momenta, Ck = fourier_space(Ctx[i,:])
        corrs_tk[i, :] = Ck

    # Fourier transform in time
    Swk = np.zeros(corrs_tk.shape, dtype=complex)
    for k in np.arange(corrs_tk.shape[1]):
        freqs, Sw = fourier_time(corrs_tk[:, k], dt)
        Swk[:, k] = Sw
    print('finished')

    return np.array(Swk), momenta, freqs
