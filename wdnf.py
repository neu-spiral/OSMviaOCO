import math
import numpy as np
from scipy.special import comb
import sys


def merge(t1, t2):
    """
    Merge two tuples (in this context, keys) into a sorted tuple by taking
    the set union of them.
    """
    if type(t1) is int and type(t2) is int and t1 == t2:  # if else conditions may be redundant if all keys are tuples
        key = t1
    elif type(t1) is int and type(t2) is int and t1 != t2:
        key = tuple(sorted([t1, t2]))
    elif type(t1) is int and type(t2) is tuple:
        key = tuple(sorted({t1}.union(set(t2))))
    elif type(t1) is tuple and type(t2) is int:
        key = tuple(sorted(set(t1).union({t2})))
    else:
        key = tuple(sorted(set(t1).union(set(t2))))
    return key


class WDNF:
    """
    A class implementing a polynomial in Weighted Disjunctive Normal Form
    (WDNF) consisting of monomials with (a) negative or positive literals and
    (b) integer terms.
    """

    def __init__(self, coefficients={}, sign=-1):
        """
        Coefficients is a dictionary containing tuples with indexes of the
        set elements as keys and coefficients as values. Sign denotes whether
        the WDNF formed with negative literals or positive literals.
        e.g: WDNF({(1, 3): 2.0, (2, 4): 10.0, (3, 4): 3.0}) =
        2.0(1-x_1)(1-x_3) + 10.0(1-x_2)(1-x_4) + 3.0(1-x_3)(1-x_4)
        """
        self.coefficients = coefficients
        self.sign = sign

    def find_dependencies(self):
        dependencies = {}
        for key in self.coefficients:
            try:
                for var in key:
        # dependencies = {var: dependencies[var] + key if var in dependencies else key for key in self.coefficients
        #                 for var in key}
                    if var in dependencies:
                        dependencies[var] + key
                    else:
                        dependencies[var] = key
            except (TypeError, KeyError):
                if key in dependencies:
                    dependencies[key].append(key)
                else:
                    dependencies[key] = [key]
        #dependencies[key] = dependencies[key] + [key] if key in dependencies else [key]
        # print(dependencies)
        return dependencies

    def __call__(self, x):
        """
        Given a dictionary x, evaluate WDNF(x) at the values x.
        """
        sum_so_far = 0.0
        for key in self.coefficients:
            prod = self.coefficients[key]  # beta
            # monomials = [1.0 - x[var] if self.sign == -1 else x[var] for var in key]
            monomials = []
            try:
                for var in key:
                    if self.sign == -1:
                        monomials.append(1.0 - x[var])
                    else:
                        monomials.append(x[var])
            except (TypeError, KeyError):
                if self.sign == -1:
                    monomials.append(1.0 - x[key])
                else:
                    monomials.append(x[key])
            prod *= np.prod(monomials)
            sum_so_far += prod
        return sum_so_far

    def __add__(self, other):
        """
        Add two polynomials in WDNF and return the resulting WDNF
        """
        assert self.sign == other.sign, 'Two WDNF polynomials of different signs cannot be added!'
        new_coefficients = self.coefficients.copy()  # empty dict for empty WDNF
        if not other.coefficients:
            return self
        elif not self.coefficients:
            return other
        else:
            additions = {key: new_coefficients[key] + other.coefficients[key] if key in self.coefficients.keys()
                         else other.coefficients[key] for key in other.coefficients}
            new_coefficients.update(additions)
        #    for key in other.coefficients:
        #        new_coefficients[key] = new_coefficients[key]+other.coefficients[key] \
        #                                if key in self.coefficients.keys() else other.coefficients[key]
        return WDNF(new_coefficients, self.sign)

    def __radd__(self, other):
        """
        useful for sum()
        :param other:
        :return:
        """
        return self

    def __mul__(self, other):
        """ Multiply two polynomials in WDNF and return the resulting WDNF
        """
        assert self.sign == other.sign, 'Two WDNF polynomials of different signs cannot be multiplied!'
        new_coefficients = dict()
        for key1 in self.coefficients:
            for key2 in other.coefficients:
                new_key = merge(key1, key2)
        # the code segment below should work but it doesn't. I couldn't figure it out why.
        # new_coefficients = {merge(key1, key2): (new_coefficients[merge(key1, key2)] + (self.coefficients[key1] *
        #                                                                                other.coefficients[key2]))
        #                     if merge(key1, key2) in new_coefficients else (self.coefficients[key1] *
        #                                                                    other.coefficients[key2])
        #                     for key1 in self.coefficients for key2 in other.coefficients}
                if new_key in new_coefficients:
                    new_coefficients[new_key] += self.coefficients[key1] * other.coefficients[key2]
                else:
                    new_coefficients[new_key] = self.coefficients[key1] * other.coefficients[key2]
        return WDNF(new_coefficients, self.sign)

    def __rmul__(self, scalar):
        """ Multiplies the coefficients of a WDNF function with a scalar
        """
        # new_coefficients = self.coefficients.copy()
        new_coefficients = {key: self.coefficients[key] * scalar for key in self.coefficients}
        # for key in self.coefficients:
        #     new_coefficients[key] = self.coefficients[key] * scalar
        return WDNF(new_coefficients, self.sign)

    def __pow__(self, k):
        """Calculates the kth power of a WDNF function and returns the result.
        k must be greater than or equal to 0.
        """
        if k == 0:
            return WDNF({(): 1}, self.sign)
        else:
            # power_wdnf = self
            # for i in range(2, k + 1):
            #     power_wdnf *= self
            copycats = [self] * k
            return np.prod(copycats)

    def evaluate(self, x, func):  # might be redundant
        return func(self(x))


class Poly(object):
    """A class for defining univariate polynomials with the largest degree and
    the coefficients list of size (largest degree + 1) where coefficients are
    stored as [coef_0 coef_1 ... coef_n]
    """

    def __init__(self, degree, poly_coef):
        """e.g: poly(n, [a0 a1 ... an]) defines
        f(x) = a0 + a1*x + ... + an*(x^n)
        """
        assert len(poly_coef) == (degree + 1), 'Size of the coefficients list does not match with the degree!'
        self.poly_coef = poly_coef
        self.degree = degree

    def __add__(self, other):
        """Adds two univariate polynomials and returns the sum as another Poly
        object.
        """
        if self.degree >= other.degree:
            poly_coef = list(np.array(self.poly_coef) + np.array(other.poly_coef + [0] * (self.degree - other.degree)))
        else:
            return other + self
        return Poly(self.degree, poly_coef)

    def __sub__(self, other):
        """Subtracts two univariate polynomials and returns the difference as
        another Poly object.
        """
        return self + ((-1) * other)

    def __mul__(self, other):
        """Multiplies two polynomials and return the product as another Poly
        object.
        """
        degree = self.degree + other.degree
        poly_coef = [0] * (degree + 1)
        for i in range(len(self.poly_coef)):
            for j in range(len(other.poly_coef)):
                poly_coef[i + j] += self.poly_coef[i] * other.poly_coef[j]
        return Poly(degree, poly_coef)

    def __rmul__(self, scalar):
        """Multiplies a polynomial with a scalar.
        """
        return Poly(self.degree, list(np.array(self.poly_coef) * scalar))

    def compose(self, my_wdnf):
        """ Given a one-dimensional polynomial function f with degree k and coefficients
        stored in coef_list, computes f(self) and returns the result in WDNF.
        """
        wdnf_so_far = WDNF({(): 1}, my_wdnf.sign)
        result = self.poly_coef[0] * wdnf_so_far
        wdnf_so_far = my_wdnf
        result += self.poly_coef[1] * wdnf_so_far
        for i in range(2, self.degree + 1):
            wdnf_so_far *= my_wdnf
            result += self.poly_coef[i] * wdnf_so_far
        return result

    def __call__(self, x):
        """Calculates f(x) for a given x.
        """
        output = 0.0
        for i in range(self.degree + 1):
            output += self.poly_coef[i] * (x ** i)
        return output


class Taylor(Poly):
    """ A class computing the Taylor expansion of a function"""

    def __init__(self, degree, derivatives, center):
        """Given the calculated derivatives at the center, initializes Taylor
        expansion of a function in the standard polynomial form by expanding the
        terms using binomial expansion.
        """
        # if center == 0:
        #    poly.__init__(self, degree, derivatives)
        # else:
        poly_coef = [0.0] * (degree + 1)
        for i in range(degree + 1):
            # print(i)
            for j in range(i, degree + 1):
                # print(j)
                if j - i > 0:
                    poly_coef[i] += derivatives[j] * comb(j, i, True) / math.factorial(j) * (-center) ** (j - i)
                    # print(poly_coef[i])
                else:
                    poly_coef[i] += derivatives[j] * comb(j, i, True) / math.factorial(j)
        super(Taylor, self).__init__(degree, poly_coef)


if __name__ == "__main__":
    wdnf0 = WDNF({(): 1}, -1)

    wdnf1 = WDNF({(2, ): 1, (3, ): 1}, -1)
    wdnf2 = WDNF({(1, 2): 4.0, (1, 3): 5.0}, -1)
    # wdnf3 = wdnf1 * wdnf2
    wdnf4 = wdnf0 + (-1) * wdnf1
    wdnf_list = [wdnf1, wdnf2]
    sys.stderr.write("wdnf_list: " + str() + "\n")

    # wdnf5 = 4 * wdnf1
    # wdnf6 = wdnf1**2
    # wdnf7 = wdnf0 + wdnf1
    # x = {1:1, 2:1, 3:1, 4:0}
    # print(wdnf1(x))

    # poly1 = Poly(2, [3, 4, 0])
    # poly2 = Poly(2, [8, 1, 1])
    # poly3 = poly2 + poly1
    # wdnf4 = poly2.compose(wdnf1)

    # myTaylor = Taylor(8, [1, 1, 1, 1, 1, 1, 1, 1, 1], 0)
    # myTaylor.expand()

    # new_wdnf1 = myTaylor.compose(wdnf1)
    # print(new_wdnf1.coefficients)
