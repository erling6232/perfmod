import numpy as np


def make_annet_conv(aif_value):
    # Construction to pass aif_value to function
    # def myfun_annet_conv(x, tau, d, fa, k21, k12):
    def myfun_annet_conv(x, *b):

        # b = [tau d fa k21 k12]
        # tau = b(1);
        # d   = b(2);
        # fa  = b(3);
        # k21 = b(4);
        # k12 = b(5);
        tau, d, fa, k21, k12 = b

        # ntime = numel(x);
        # ntime = np.prod(x.size).item()

        # delta t
        # dt = (x([2:end end]) - x([1 1:end-1]))/2;
        dt = np.empty_like(x, dtype=np.float32)
        dt[1:] = x[1:] - x[:-1]
        # dt(1) = dt(2);
        dt[0] = dt[1]
        # dt(end) = dt(end-1);

        m = np.mean(aif_value)
        # f1 = interp1(x,aif_value,x + tau,'linear',m);
        f1 = np.interp(x + tau, x, aif_value, left=m, right=m)
        # f1 = aif_value;
        f2 = dt * np.exp(-x / d)
        caprime = np.convolve(f1, f2 / d).reshape(-1, 1)
        # caprime = caprime(1:ntime);

        # hematocrit adjustment
        # From Annet, page 845
        # caprime = caprime/(1 - 0.55);

        f1 = dt * np.exp(-k12 * x)
        f2 = caprime
        ck = k21 * np.convolve(f1, f2).reshape(-1, 1)
        # ck = ck(1:ntime)
        c = fa * caprime + ck

        # res = C - data.ydata;
        return c
    return myfun_annet_conv


def make_annet_matrix(aif_value):
    # Construction to pass aif_value to function
    # def myfun_annet_matrix(x, tau, d, fa, k21, k12):
    def myfun_annet_matrix(x, *b):

        # x = (rowvect(x))';
        # x = x.flatten(1)    # Column vector
        x = x.reshape(-1, 1)  # Column vector
        # aif = (rowvect(aif_value))';
        aif = aif_value.reshape(-1, 1)  # Column vector

        # b = [tau d fa k21 k12]
        # tau = b(1);
        # d   = b(2);
        # fa  = b(3);
        # k21 = b(4);
        # k12 = b(5);
        tau, d, fa, k21, k12 = b

        # ntime = numel(x);
        ntime = np.prod(x.size).item()

        # delta t
        # dt = (x([2:end end]) - x([1 1:end-1]))/2;
        dt = np.empty_like(x, dtype=np.float32)
        dt[1:] = x[1:] - x[:-1]
        # dt(1) = dt(2);
        dt[0] = dt[1]
        # dt(end) = dt(end-1);

        m = np.mean(aif)
        # f1 = interp1(x,aif,x + tau,'linear',m);
        f1 = np.interp(x + tau, x, aif, left=m, right=m)
        # f1 = aif;

        # _m = diag(ones(ntime,1));
        _m = np.eye(ntime)
        for i in range(ntime):
            for j in range(i + 1):
                _dt = x[i] - x[j]
                _m[i, j] = np.exp(-_dt / d) * dt[j]
        caprime = (1 / d) * _m * f1

        # _n = diag(ones(ntime,1));
        _n = np.eye(ntime)
        for i in range(ntime):
            for j in range(i + 1):
                _dt = x[i] - x[j]
                _n[i, j] = np.exp(-_dt * k12) * dt[j]
        ck = k21 * _n * caprime

        c = fa * caprime + ck
        # res = C - data.ydata;
        return c
    return myfun_annet_matrix


def make_sourbron_conv(aif_value):
    # Construction to pass aif_value to function
    # def myfun_sourbron_conv(x, vp, tp, ft, tt):
    def myfun_sourbron_conv(x, *b):
        # NB must have equal time sampling

        # x = x(:);
        # x = x.reshape(-1, 1)
        # aif = aif_value(:);
        # aif = aif_value.reshape(-1, 1)
        aif = aif_value

        # [vp tp ft tt]
        # vp = b(1);
        # tp = b(2);
        # ft = b(3);
        # tt = b(4);
        vp, tp, ft, tt = b

        # ntime = numel(x);
        ntime = x.size
        # dt = (x([2:end end]) - x([1 1:end-1]))/2;
        dt = np.empty_like(x, dtype=np.float32)
        dt[1:] = x[1:] - x[:-1]
        # dt(1) = dt(2);
        dt[0] = dt[1]
        # dt(end) = dt(end-1);

        f2 = dt * np.exp(-x / tp)
        # cp = conv(aif,f2/tp,'same');
        # cp = np.convolve(aif, f2 / tp, mode='same')
        cp = np.convolve(aif, f2 / tp)
        cp = cp[:ntime]
        # cp = cp.reshape(-1, 1)

        f1 = dt * np.exp(-x / tt)
        # ck = ft*conv(f1,cp,'same');
        # ck = ft*np.convolve(f1, cp, mode='same')
        ck = ft * np.convolve(f1, cp)
        ck = ck[:ntime]
        # ck = ck.reshape(-1, 1)
        c = vp * cp + ck
        return c
    return myfun_sourbron_conv


def make_sourbron_loop(aif_value):
    # Construction to pass aif_value to function
    # def myfun_sourbron_loop(x, vp, tp, ft, tt):
    def myfun_sourbron_loop(x, *b):

        # x = x(:);
        # x = x.reshape(-1, 1)
        # aif = aif_value(:);
        # aif = aif_value.reshape(-1, 1)
        aif = aif_value

        # vp = b(1);
        # tp = b(2);
        # ft = b(3);
        # tt = b(4);
        vp, tp, ft, tt = b

        # ntime = numel(x);
        ntime = x.size

        # delta t
        # dt = (x([2:end end]) - x([1 1:end-1]))/2;
        # dt = (x[1:] - x[:-1]) / 2
        dt = np.empty_like(x, dtype=np.float32)
        dt[1:] = x[1:] - x[:-1]
        # dt(1) = dt(2);
        dt[0] = dt[1]
        # dt(end) = dt(end-1);

        # find cp
        cp = np.zeros(ntime)
        for i in range(ntime):
            for j in range(i + 1):
                _dt = x[i] - x[j]
                delta = np.exp(-_dt / tp) * aif[j] * dt[j]
                cp[i] = cp[i] + delta
        cp = (1 / tp) * cp

        # find solution to ODE
        # ck = zeros(ntime,1);
        ck = np.zeros(ntime)
        for i in range(ntime):
            for j in range(i + 1):
                _dt = x[i] - x[j]
                delta = np.exp(-_dt / tt) * cp[j] * dt[j]
                ck[i] = ck[i] + delta

        # combine model
        c = vp * cp + ft * ck
        return c
    return myfun_sourbron_loop


def make_sourbron_matrix(aif_value):
    # Construction to pass aif_value to function
    # def myfun_sourbron_matrix(x, vp, tp, ft, tt):
    def myfun_sourbron_matrix(x, *b):

        # x = x(:);
        x = x.reshape(-1, 1)
        # aif = aif_value(:);
        aif = aif_value.reshape(-1, 1)

        # vp = b(1);
        # tp = b(2);
        # ft = b(3);
        # tt = b(4);
        vp, tp, ft, tt = b

        # ntime = numel(x);
        ntime = np.prod(x.size).item()

        # delta t
        # dt = (x([2:end end]) - x([1 1:end-1]))/2;
        dt = np.empty_like(x, dtype=np.float32)
        dt[1:] = x[1:] - x[:-1]
        # dt(1) = dt(2);
        dt[0] = dt[1]
        # dt(end) = dt(end-1);

        # find cp
        # _m = diag(ones(ntime,1));
        _m = np.eye(ntime)
        for i in range(ntime):
            for j in range(i + 1):
                # time span for integration
                _dt = x[i] - x[j]
                _m[i, j] = np.exp(-_dt / tp) * dt[j]
        cp = (1 / tp) * _m * aif

        # _n = diag(ones(ntime,1));
        _n = np.eye(ntime)
        for i in range(ntime):
            for j in range(i + 1):
                # time span for integration
                _dt = x[i] - x[j]
                _n[i, j] = np.exp(-_dt / tt) * dt[j]
        ck = ft * _n * cp

        # combine model
        c = vp * cp + ck
        return c
    return myfun_sourbron_matrix


def make_sourbron_numint(aif_value):
    # Construction to pass aif_value to function
    # def myfun_sourbron_numint(x, vp, tp, ft, tt):
    def myfun_sourbron_numint(x, *b):
        # NB only works for even delta t!!!

        # x = x(:);
        x = x.reshape(-1, 1)
        # aif = aif_value(:);
        aif = aif_value.reshape(-1, 1)

        # vp = b(1);
        # tp = b(2);
        # ft = b(3);
        # tt = b(4);
        vp, tp, ft, tt = b

        # ntime = numel(x);
        ntime = np.prod(x.size).item()

        #
        # Numerical integration matlab
        #

        # concentration plasma space cp
        cp = (1 / tp) * np.ones(ntime)
        for i in range(1, ntime):
            tau = x[1:i]
            y = np.exp(-(x[i] - tau) / tp)
            y = y * aif[0:i]
            cp[i, 0] = cp[i, 0] * np.trapz(y, tau)
        cp[0] = 0

        # concentration tubular space
        ck = ft * np.ones(ntime)
        for i in range(1, ntime):
            tau = x[0:i]
            y = np.exp(-(x[i] - tau) / tt)
            y = y * cp[0:i]
            ck[i, 0] = ck[i, 0] * np.trapz(y, tau)
        ck[0] = 0

        # combine model
        c = vp * cp + ck
        return c
    return myfun_sourbron_numint
