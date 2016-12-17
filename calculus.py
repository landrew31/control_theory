import math
import numpy as np
from scipy import optimize, linalg
from matplotlib import pyplot


THETA = 31


class Solver:
    def  __init__(self, **kwargs):
        self.init_data = kwargs
        self.k = kwargs.get('k')
        self.tau = kwargs.get('tau')
        self.t1 = kwargs.get('t1')
        self.t2 = kwargs.get('t2')
        self.t3 = kwargs.get('t3')
        self.epses = [0.01 * i for i in range(1, 6)]

    def __str__(self):

        str_object = 'Clause 3.1:\n'
        for i, t0 in enumerate(self.t0):
            str_object = '{}T0_{}: {}\n\n'.format(str_object, i, t0)

        str_object = '{}Clause 3.2:\n'.format(str_object)
        str_object = '{0}Lambda: {1:.3f}\n'.format(str_object, self.lambd_0)
        str_object = '{0}Kp_0: {1:.3f}\n'.format(str_object, self.kp_0)
        str_object = '{0}Kp_1: {1:.3f}\n'.format(str_object, self.kp_1)
        str_object = '{0}Omega 0: {1}\n'.format(str_object, self.xes_for_first)
        str_object = '{0}Omega 1: {1}\n'.format(str_object, self.xes_for_second)
        str_object = '{0}T_0_0 djuri: {1}\n'.format(str_object, self.t0_0_dj)
        str_object = '{0}T_0_1 djuri: {1}\n\n'.format(str_object, self.t0_1_dj)

        str_object = '{}Clause 3.3:\n'.format(str_object)
        str_object = '{0}T: {1}\n'.format(str_object, self.t)
        str_object = '{0}b: {1}\n'.format(str_object, self.b)
        str_object = '{0}Nu: {1}\n'.format(str_object, self.nu)
        str_object = '{0}q: {1}\n'.format(str_object, self.q)
        str_object = '{0}T_0 dyn: {1}\n\n'.format(str_object, self.t0_dyn)

        str_object = '{}Clause 4:\n'.format(str_object)
        str_object = '{0}d: {1}\n'.format(str_object, self.n_4_d)
        str_object = '{0}Ti: {1}\n'.format(str_object, self.ti)
        str_object = '{0}lambda: {1}\n'.format(str_object, self.lambdas)
        str_object = '{0}Kp: {1}\n\n'.format(str_object, self.kpes)

        str_object = '{}Clause 5 (first column):\n'.format(str_object)
        str_object = '{}{}\n'.format(str_object, repr(self.clause_5_first_column))
        
        str_object = '{}Clause 5 (second column):\n'.format(str_object)
        str_object = '{}{}\n'.format(str_object, repr(self.clause_5_second_column))

        str_object = '{}Clause 8:\n'.format(str_object)
        str_object = '{0}A0: {1}\n'.format(str_object, self.A_0)
        str_object = '{0}A1: {1}\n'.format(str_object, self.A_1)

        return str_object

    class Calculation5:
        def __init__(self, **kwargs):
            self.t3 = kwargs['t3']
            self.t1 = kwargs['t1']
            self.t2 = kwargs['t2']
            self.tau = kwargs['tau']
            self.k = kwargs['k']

        def calculate_5(self, with_t3=True):
            ARG = 2.62

            if with_t3:
                t3 = self.t3
            else:
                t3 = 0

            def equation_for_omega(x, t1=self.t1, t2=self.t2, t3=t3, tau=self.tau):
                return np.arctan(x * t1) + np.arctan(x * t2) + np.arctan(x * t3) + x * tau

            def get_A(omega, k=self.k, t1=self.t1, t2=self.t2, t3=t3):
                return k / math.sqrt(
                    (1 + (omega * t1) ** 2) *
                    (1 + (omega * t2) ** 2) *
                    (1 + (omega * t3) ** 2)
                )

            def equation_for_dubleve(x, t_0_opt):
                return equation_for_omega(x) + x * t_0_opt / 2

            def get_A_dubleve(dubleve, t_0_opt):
                return get_A(dubleve) * math.sin(dubleve * t_0_opt / 2) / (dubleve * t_0_opt / 2)

            self.omega_fi_n = float(list(optimize.fsolve(
                lambda x: equation_for_omega(x) - ARG,
                (0.01,)
            ))[0])

            self.omega_fi_n_n = self.omega_fi_n / math.sqrt(2)
            self.omega_fi_n_v = self.omega_fi_n * math.sqrt(2)
            self.A_omega_fi_n = get_A(self.omega_fi_n)
            self.fi_n_fi = equation_for_omega(self.omega_fi_n_v) - equation_for_omega(self.omega_fi_n_n)
            self.fi_n_A = get_A(self.omega_fi_n_v) / get_A(self.omega_fi_n_n)
            self.t_nep_i_opt = (4.061 * np.power(self.fi_n_A, -0.3387) * math.pow(self.fi_n_fi, 0.2075)) / self.omega_fi_n
            self.k_nep_p_opt = (
                (1 + 1.189 * np.power(self.fi_n_A, 0.7139) * np.power((1.852 - self.fi_n_fi), 0.8643))
                /
                (2 * self.A_omega_fi_n)
            )
            self.t_0_opt = (0.5742 * np.power(self.fi_n_A, 0.5742) * np.power(self.fi_n_fi, 0.9394)) / self.omega_fi_n
            self.dubleve_fi = float(list(optimize.fsolve(
                lambda x: equation_for_dubleve(x, self.t_0_opt) - ARG,
                (0.1,)
            ))[0])
            self.dubleve_fi_n_n = self.dubleve_fi / math.sqrt(2)
            self.dubleve_fi_n_v = self.dubleve_fi * math.sqrt(2)
            self.A_dubleve_fi = get_A_dubleve(self.dubleve_fi, self.t_0_opt)
            self.fi_A = get_A_dubleve(self.dubleve_fi_n_v, self.t_0_opt) / get_A_dubleve(self.dubleve_fi_n_n, self.t_0_opt)
            self.fi_fi = equation_for_dubleve(self.dubleve_fi_n_v, self.t_0_opt) - equation_for_dubleve(self.dubleve_fi_n_n, self.t_0_opt)
            self.t_i_opt = (4.061 * np.power(self.fi_A, -0.3387) * math.pow(self.fi_fi, 0.8843)) / self.dubleve_fi
            self.k_p_opt = (
                (1 + 1.189 * math.pow(self.fi_A, 0.7139) * math.pow(1.852 - self.fi_fi, 0.8643))
                /
                (2 * self.A_dubleve_fi)
            )

            return self

        def __repr__(self):
            str_object = ''

            str_object = '{0}self.omega_fi_n: {1}\n'.format(str_object, self.omega_fi_n)
            str_object = '{0}self.omega_fi_n_n: {1}\n'.format(str_object, self.omega_fi_n_n)
            str_object = '{0}self.omega_fi_n_v: {1}\n'.format(str_object, self.omega_fi_n_v)
            str_object = '{0}self.A_omega_fi_n: {1}\n'.format(str_object, self.A_omega_fi_n)
            str_object = '{0}fi_n_fi: {1}\n'.format(str_object, self.fi_n_fi)
            str_object = '{0}fi_n_A: {1}\n'.format(str_object, self.fi_n_A)
            str_object = '{0}t_nep_i_opt: {1}\n'.format(str_object, self.t_nep_i_opt)
            str_object = '{0}k_nep_p_opt: {1}\n'.format(str_object, self.k_nep_p_opt)
            str_object = '{0}t_0_opt: {1}\n'.format(str_object, self.t_0_opt)
            str_object = '{0}dubleve_fi: {1}\n'.format(str_object, self.dubleve_fi)
            str_object = '{0}dubleve_fi_n_n: {1}\n'.format(str_object, self.dubleve_fi_n_n)
            str_object = '{0}dubleve_fi_n_vKp: {1}\n'.format(str_object, self.dubleve_fi_n_v)
            str_object = '{0}A_dubleve_fi: {1}\n'.format(str_object, self.A_dubleve_fi)
            str_object = '{0}fi_A: {1}\n'.format(str_object, self.fi_A)
            str_object = '{0}fi_fi: {1}\n'.format(str_object, self.fi_fi)
            str_object = '{0}t_i_opt: {1}\n'.format(str_object, self.t_i_opt)
            str_object = '{0}k_p_opt: {1}\n'.format(str_object, self.k_p_opt)

            return str_object

    def execute_calculations(self):
        self.calculate_3_1()
        self.calculate_3_2_start()
        self.calculate_3_2_end()
        self.calculate_3_3()
        self.calculate_4()
        self.clause_5_first_column = self.Calculation5(**self.init_data)
        self.clause_5_second_column = self.Calculation5(**self.init_data)
        self.clause_5_first_column.calculate_5(False)
        self.clause_5_second_column.calculate_5(True)
        self.calculate_6()
        self.calculate_8()
        return self

    def ploted(self):
        self.plot_t0_eps()
        return self

    def calculate_3_1(self):
        deltas = [
            self.k / self.t1,
            self.k / (self.t1 + self.t2)
        ]

        def get_t0(delta):
            return [
                '{0:.3f}'.format(
                    eps / delta
                ) for eps in self.epses
            ]

        self.t0 = [get_t0(delta) for delta in deltas]

    def calculate_3_2_start(self):
        self.lambd_0 = 1 / self.t1
        self.kp_0 = (
            self.lambd_0 * self.t1
            /
            (self.k * (1 + self.lambd_0 * self.tau))
        )
        self.kp_1 = 1 / self.k

    def calculate_3_2_end(self):
        def equation_for_w0_1(x, eps, t1, tau, k, kp_0):
            return (np.absolute(1 / (
                1 + (t1 * 1j * x * (math.e **(tau * 1j * x))) / (k * kp_0)
            )) - eps)

        def equation_for_w0_2(x, eps, t1):
            return (1 / math.sqrt(1 + t1 * t1 * x * x) - eps)

        self.xes_for_first = [
            '{0:.4f}'.format(float(list(optimize.fsolve(
                lambda x: equation_for_w0_1(x, eps, self.t1, self.tau, self.k, self.kp_0),
                (0.1,)
            ))[0])) for eps in self.epses
        ]

        self.xes_for_second = [
            '{0:.4f}'.format(float(list(optimize.fsolve(
                lambda x: equation_for_w0_2(x, eps, self.t1),
                (0.00001,)
            ))[0])) for eps in self.epses
        ]

        self.t0_0_dj = ['{0:.4f}'.format(math.pi / float(omega)) for omega in self.xes_for_first]

        self.t0_1_dj = ['{0:.4f}'.format(math.pi / float(omega)) for omega in self.xes_for_second]

    def calculate_3_3(self):

        def dyn_for_q(x, k, b, nu):
            return (
                k * math.sqrt(
                    (1 + b * b * x * x) /
                    (1 - 2 * x * x + x * x * x * x + 4 * x * x * nu * nu)
                ) - 1 / THETA
            )

        def dyn_for_qq(x, k, b, nu):
            return (
                    k * (1 + b * x * 1j) /
                    (1 - x * x + 2 * x * 1j * nu)
                 - 1 / THETA
            )

        self.t = math.sqrt(self.t2 * self.t3)
        self.b = self.t1 / self.t
        self.nu = (self.t2 + self.t3) / (2 * self.t)
        self.q = '{0:.4f}'.format(float(list(optimize.fsolve(
            lambda x: dyn_for_q(x, self.k, self.b, self.nu),
            (0.1,)
        ))[0]))
        self.t0_dyn = math.pi * self.t / abs(float(self.q))

    def calculate_4(self):
        t0 = float(self.t0[0][2])
        exp_mult =  (math.exp(t0 / self.t1) - 1)
        self.n_4_d = round(self.tau / t0)
        self.ti = t0 / exp_mult

        def get_kp(lamb):
            multiplier = (1 - math.exp(- lamb * t0))
            return (
                multiplier /
                (self.k * exp_mult * (1 + self.n_4_d * multiplier))
            )

        coefs = [1, 1.5, 2, 3]
        self.lambdas = [1 / coef / self.t1 for coef in coefs]
        self.kpes = [get_kp(lamb) for lamb in self.lambdas]

    def calculate_8(self):
        def get_A0(k_p, t_0, t_i=self.ti, t_d=0):
            return (
                k_p * (1 + t_0 / t_i + t_d / t_0)
            )

        def get_A1(k_p, t_0, t_d=0):
            return (
                - k_p * (1 + 2 * t_d / t_0)
            )

        def get_A2(k_p, t_0, t_d=0):
            return (
                k_p *  t_d / t_0
            )

        self.kp_8 = self.kpes
        self.kp_8.append(self.clause_5_first_column.k_p_opt)
        self.kp_8.append(self.clause_5_second_column.k_p_opt)
        self.t0_8 = [float(self.t0_0_dj[2])] * 4
        print(self.t0_8)
        self.t0_8.append(self.clause_5_second_column.t_0_opt)
        self.t0_8.append(self.clause_5_first_column.t_0_opt)

        self.A_0 = []
        self.A_1 = []
        for kp, t0_8 in zip(self.kp_8, self.t0_8):
            print(kp, t0_8)
            self.A_0.append(get_A0(kp, t0_8))
            self.A_1.append(get_A1(kp, t0_8))

    def plot_t0_eps(self):
        fig, ax1 = pyplot.subplots()
        ax1.plot(self.epses, self.t0_0_dj)
        ax1.plot(self.epses, self.t0_1_dj)
        pyplot.show()


init_data = {
    'k': 3.72,
    't1': 28,
    't2': 16,
    't3': 7,
    'tau': 54
}
print(
    Solver(
        **init_data
    ).execute_calculations(
    ).ploted(

    )
)
