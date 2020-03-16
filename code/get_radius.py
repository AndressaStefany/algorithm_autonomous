def get_radius(fac, p, m, k_win):
    a = p^(1/(2^(1/2)))
    b = (1-(1/(k_win+1)))^m
    return fac*a*1.0/b