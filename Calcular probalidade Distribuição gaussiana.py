from numpy import *
import math
# covariance matrix
sigma = matrix([[100, 50],
           [50, 233.3]])
# mean vector
mu = array([80, 203.3])

# input
x = array([100, 225])

def pdf_multivariate_gauss(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        print(det)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = (1.0/ ( math.pow((2*pi),float(size)/2))) * (1.0/ math.pow(det,1.0/2))
        x_mu = matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

print(pdf_multivariate_gauss(x, mu, sigma))



def pdf_univariant_gauss(x,u,o):
    return (1 / (sqrt(2 * pi) * o)) * exp(((-1 / (2 * pow(o,2))) * math.pow(x - u, 2)))


print(pdf_univariant_gauss(100,93.3,66.58))

print(pdf_univariant_gauss(225,156.6,5.77))

print(pdf_univariant_gauss(100,80,10))

print(pdf_univariant_gauss(225,203.3,15.275))