import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.sparse.linalg

import scipy.sparse
kron = scipy.sparse.kron
eye = scipy.sparse.eye

X = np.array([[0., 1.], [1., 0.]])
Z = np.array([[1., 0.], [0., -1.]])
Y = np.array([[0., -1.j], [1.j, 0.]])
dense_X, dense_Y, dense_Z = X.copy(), Y.copy(), Z.copy()

X = scipy.sparse.csr_array(X)
Y = scipy.sparse.csr_array(Y)
Z = scipy.sparse.csr_array(Z)

def AFH(L,):
    """
    Convention: +XX + YY + ZZ
    """
    # H = np.zeros([2**L, 2**L])
    H = scipy.sparse.csr_matrix((2**L, 2**L), dtype=np.float64)
    for Op in [X, Y, Z]:
        for i in range(L-1):
            h = eye(1)
            for j in range(0, i):
                h = kron(h, eye(2))

            h = kron(h, Op)
            h = kron(h, Op)
            for j in range(i+2, L):
                h = kron(h, eye(2))

            H = H + h

    return H


def TFI(L, g):
    """
    Convention: -ZZ + -gX
    """
    # H = np.zeros([2**L, 2**L])
    H = scipy.sparse.csr_matrix((2**L, 2**L), dtype=np.float64)
    print("constructing H_TFI", end="\r")
    for i in range(L-1):
        print("constructing H_TFI ZZ term for i=%d" % i, end="\r")
        h = eye(1)
        for j in range(0, i):
            h = kron(h, eye(2))

        h = kron(h, Z)
        h = kron(h, Z)
        for j in range(i+2, L):
            h = kron(h, eye(2))

        H = H + (-h)

    h = Z
    for i in range(L-2):
        h = kron(h, eye(2))

    h = kron(h, Z)
    H = H + (-h)

    for i in range(L):
        print("constructing H_TFI X term for i=%d" % i, end="\r")
        h = eye(1)
        for j in range(0, i):
            h = kron(h, eye(2))

        h = kron(h, X)
        for j in range(i+1, L):
            h = kron(h, eye(2))

        H = H + (-g * h)

    return H

def H_z(L):
    """
    H = -Z
    """
    print("constructing H_z", end="\r")
    # H = np.zeros([2**L, 2**L])
    H = scipy.sparse.csr_matrix((2**L, 2**L), dtype=np.float64)
    for i in range(L):
        print("constructing Z term for i=%d" % i, end="\r")
        h = eye(1)
        for j in range(0, i):
            h = kron(h, eye(2))

        h = kron(h, Z)
        for j in range(i+1, L):
            h = kron(h, eye(2))

        H = H + (-h)

    return H

def H_x(L):
    """
    H = -X
    """
    print("constructing H_x", end="\r")
    # H = np.zeros([2**L, 2**L])
    H = scipy.sparse.csr_matrix((2**L, 2**L), dtype=np.float64)
    for i in range(L):
        print("constructing X term for i=%d" % i, end="\r")
        h = eye(1)
        for j in range(0, i):
            h = kron(h, eye(2))

        h = kron(h, X)
        for j in range(i+1, L):
            h = kron(h, eye(2))

        H = H + (-h)

    return H

def get_C(vec, Op1, Op2, L):
    # Compute <vec | Op1_r Op2_{mid} | vec> for all r
    vec = vec.copy()
    C = []
    Op2_vec = np.tensordot(Op2,
                           vec.reshape([2**(L//2), 2, 2**((L-1)//2)]),
                           [[1], [1]])
    Op2_vec = Op2_vec.transpose([1, 0, 2]).flatten()

    Op2_exp_val = vec.conj().dot(Op2_vec)
    for r in range(L):
        Op1_vec = np.tensordot(Op1,
                               vec.reshape([2**(r), 2, 2**(L-1-r)]),
                               [[1], [1]])
        Op1_vec = Op1_vec.transpose([1, 0, 2]).flatten()
        Op1_exp_val = vec.conj().dot(Op1_vec)

        Op1_Op2_vec = np.tensordot(Op1,
                                   Op2_vec.reshape([2**(r), 2, 2**(L-1-r)]),
                                   [[1], [1]])
        Op1_Op2_vec = Op1_Op2_vec.transpose([1, 0, 2]).flatten()
        exp_val = vec.conj().dot(Op1_Op2_vec)
        # print("r = ", r, "exp=", exp_val, "exp--=", exp_val - Op1_exp_val * Op2_exp_val)
        # C.append(exp_val)
        C.append(exp_val - Op1_exp_val * Op2_exp_val)


    return C

def get_ETC(vec, H, dt, steps, Op1, Op2, L):
    ETC = []
    ETC.append(get_C(vec, Op1, Op2, L))

    # U = scipy.linalg.expm(-1.j * dt * H)
    for idx in range(steps):
        # vec = U.dot(vec)
        vec = scipy.sparse.linalg.expm_multiply(-1.j * dt * H, vec)

        ETC.append(get_C(vec, Op1, Op2, L))

    ETC = np.array(ETC)
    return ETC


def get_UTC(vec, H, dt, steps, Op1, Op2, L):
    UTC = []
    UTC.append(get_C(vec, Op1, Op2, L))

    Op2_vec = np.tensordot(Op2,
                           vec.reshape([2**(L//2), 2, 2**((L-1)//2)]),
                           [[1], [1]])
    Op2_vec = Op2_vec.transpose([1, 0, 2]).flatten()

    # U = scipy.linalg.expm(-1.j * dt * H)
    for idx in range(steps):
        # vec = U.dot(vec)
        vec = scipy.sparse.linalg.expm_multiply(-1.j * dt * H, vec)

        # Op2_vec = U.dot(Op2_vec)
        Op2_vec = scipy.sparse.linalg.expm_multiply(-1.j * dt * H, Op2_vec)

        C = []
        for r in range(L):
            Op1_Op2_vec = np.tensordot(Op1,
                                       Op2_vec.reshape([2**(r), 2, 2**(L-1-r)]),
                                       [[1], [1]])
            Op1_Op2_vec = Op1_Op2_vec.transpose([1, 0, 2]).flatten()
            exp_val = vec.conj().dot(Op1_Op2_vec)
            C.append(exp_val)

        UTC.append(C)

    return np.array(UTC)



if __name__ == "__main__":
    L = 11
    H1 = H_x(L)
    H_TFI = TFI(L, 3.)

    E, V = np.linalg.eigh(H1)
    print("E = ", E)
    E_min_arg = np.argmin(E)
    GS = V[:, E_min_arg]
    init_vec = GS.copy()

    # total_steps = 100
    # for step in range(total_steps):
    #     H = (total_steps - step) / total_steps * H1 + step / total_steps * H_TFI
    #     vec = scipy.sparse.linalg.expm_multiply(-1.j * H, vec)

    E, V = np.linalg.eigh(H_TFI)
    E_min_arg = np.argmin(E)
    GS = V[:, E_min_arg]

    sum_all_basis = np.sum(V, axis=-1)
    sum_all_basis /= np.linalg.norm(sum_all_basis)

    vec = GS + 0.1 * sum_all_basis
    # vec = GS + 0.1 * init_vec
    # vec = np.random.rand(2**L) - 0.5
    # vec = GS + 0.2 * vec
    # vec = init_vec
    vec /= np.linalg.norm(vec)

    print("init vec = ", vec)
    print("E_TFI = ", E)
    overlap = np.abs(vec.dot(V))
    print("overlap with eigenstates= ", overlap)
    plt.semilogy(E, overlap, 'o-')
    plt.show()

    dt = 0.02
    steps = 75
    Op1 = Op2 = Z
    ETC = get_ETC(vec, H_TFI, dt, steps, Op1, Op2, L)
    UTC = get_UTC(vec, H_TFI, dt, steps, Op1, Op2, L)
    for data in [UTC, ETC]:
        plt.imshow(data.real, origin='lower', aspect='auto')
        plt.show()

        import ft
        fig = plt.figure()
        ax = plt.gca()
        Swk_c, momenta_c, freqs_c = ft.get_spectral_function2(data, dt)
        y_lim = (-1, 5)

        # Swk_c = Swk_c.real
        # Swk_c[Swk_c < 0] = 1e-16
        # Swk_c = np.log(Swk_c)

        # len_freqs_c = len(freqs_c)
        # freqs_c = freqs_c[len_freqs_c//2 + 1:]
        # Swk_c = Swk_c[len_freqs_c//2 + 1:]

        # Swk_c = np.abs(Swk_c)

        im = ax.pcolormesh(momenta_c, freqs_c, np.real(Swk_c[:-1, :-1]), cmap=plt.get_cmap('magma'))
        ax.set_xlabel('k')
        ax.set_ylabel(r'$\omega$')
        # ax.set_ylim(y_lim)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()





    # for L in [6, 8]:
    #     E_list = []
    #     g_list = np.arange(0, 2, 0.05)
    #     for g in g_list:
    #         print(f"runnning L={L}, g={g}")
    #         H = TFI(L, g)
    #         E, V = np.linalg.eigh(H)
    #         E_list.append(np.amin(E))
    #         print("E = ", np.amin(E))


    #     # plt.plot(g_list, E_list, 'o-')
    #     dE = np.array(E_list)[1:] - np.array(E_list)[:-1]
    #     ddE = dE[1:] - dE[:-1]
    #     plt.plot(g_list[:-2], ddE, 'x-', label=f'L={L}')

    # plt.legend()
    # plt.show()


