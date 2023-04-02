"""
graphScattering.py  Graph scattering transform models
Functions:
monicCubicFunction: computes a function with a monic polynomial with cubic
    interpolation
HannKernel: computes the values of the Hann kernel function
Wavelets (filter banks):
monicCubicWavelet: computes the filter bank for the wavelets constructed
    of two monic polynomials interpolated with a cubic one
tightHannWavelet: computes the filter bank for the tight wavelets constructed
    using a Hann kernel function and warping
diffusionWavelets: computes the filter bank for the diffusion wavelets
Models:
graphScatteringTransform: base class for the computation of the graph scattering
    transform coefficients
DiffusionScattering: diffusion scattering transform
MonicCubic: graph scattering transform using monic cubic polynomial wavelets
TightHann: graph scattering transform using a tight frame with Hann wavelets
GeometricScattering: geometric scattering transform
"""

import numpy as np


zeroTolerance = 1e-9 # Values below this number are considered zero.
infiniteNumber = 1e12 # infinity equals this number

def diffusionWavelets(J, T):
    """
    diffusionWavelets: computes the filter bank for the diffusion wavelets
    See R. R. Coifman and M. Maggioni, “Diffusion wavelets,” Appl. Comput.
        Harmonic Anal., vol. 21, no. 1, pp. 53–94, July 2006.
    Alternatively, see eq. (6) of F. Gama, A. Ribeiro, and J. Bruna, “Diffusion
        Scattering Transforms on Graphs,” in 7th Int. Conf. Learning
        Representations. New Orleans, LA: Assoc. Comput. Linguistics,
        6-9 May 2019, pp. 1–12.
    Input:
        J (int): number of scales
        T (np.array): lazy diffusion operator
    Output:
        H (np.array): of shape J x N x N contains the J matrices corresponding
            to all the filter scales
    """
    # J is the number of scales, and we do waveletgs from 0 to J-1, so it always
    # needs to be at least 1: I need at last one scale
    assert J > 0
    N = T.shape[0] # Number of nodes
    assert T.shape[1] == N # Check it's a square matrix
    I = np.eye(N) # Identity matrix
    H = (I - T).reshape(1, N, N) # 1 x N x N
    for j in range(1,J):
        thisPower = 2 ** (j-1) # 2^(j-1)
        powerT = np.linalg.matrix_power(T, thisPower) # T^{2^{j-1}}
        thisH = powerT @ (I - powerT) # T^{2^{j-1}} * (I - T^{2^{j-1}})
        H = np.concatenate((H, thisH.reshape(1,N,N)), axis = 0)
    return H

class GraphScatteringTransform:
    """
    graphScatteringTransform: base class for the computation of the graph
        scattering transform coefficients
    Initialization:
    Input:
        numScales (int): number of wavelet scales (size of the filter bank)
        numLayers (int): number of layers
        adjacencyMatrix (np.array): of shape N x N
    Output:
        Creates graph scattering transform base handler
    Methods:
        Phi = .computeTransform(x): computes the graph scattering coefficients
            of input x (where x is a np.array of shape B x F x N, with B the
            batch size, F the number of node features, and N the number of
            nodes)
    """

    # We use this as base class to then specify the wavelet and the self.U
    # All of them use np.abs() as noinlinearity. I could make this generic
    # afterward as well, but not for now.

    def __init__(self, numScales, numLayers, adjacencyMatrix):

        self.J = numScales
        self.L = numLayers
        self.W = adjacencyMatrix.copy()
        self.N = self.W.shape[0]
        assert self.W.shape[1] == self.N
        self.U = None
        self.H = None

    def computeTransform(self, x):
        # Check the dimension of x: batchSize x numberFeatures x numberNodes
        assert len(x.shape) == 3
        B = x.shape[0] # batchSize
        F = x.shape[1] # numberFeatures
        assert x.shape[2] == self.N
        # Start creating the output
        #   Add the dimensions for B and F in low-pass operator U
        U = self.U.reshape([1, self.N, 1]) # 1 x N x 1
        #   Compute the first coefficient
        Phi = x @ U # B x F x 1
        rhoHx = x.reshape(B, 1, F, self.N) # B x 1 x F x N
        # Reshape U once again, because from now on, it will have to multiply
        # J elements (we want it to be 1 x J x N x 1)
        U = U.reshape(1, 1, self.N, 1) # 1 x 1 x N x 1
        U = np.tile(U, [1, self.J, 1, 1])
        # Now, we move to the rest of the layers
        for l in range(1,self.L): # l = 1,2,...,L
            nextRhoHx = np.empty([B, 0, F, self.N])
            for j in range(self.J ** (l-1)): # j = 0,...,l-1
                thisX = rhoHx[:,j,:,:] # B x J x F x N
                thisHx = thisX.reshape(B, 1, F, self.N) \
                            @ self.H.reshape(1, self.J, self.N, self.N)
                    # B x J x F x N
                thisRhoHx = np.abs(thisHx) # B x J x F x N
                nextRhoHx = np.concatenate((nextRhoHx, thisRhoHx), axis = 1)

                phi_j = thisRhoHx @ U # B x J x F x 1
                phi_j = phi_j.squeeze(3) # B x J x F
                phi_j = phi_j.transpose(0, 2, 1) # B x F x J
                Phi = np.concatenate((Phi, phi_j), axis = 2) # Keeps adding the
                    # coefficients
            rhoHx = nextRhoHx.copy()

        return Phi

class DiffusionScattering(GraphScatteringTransform):
    """
    DiffusionScattering: diffusion scattering transform
    Initialization:
    Input:
        numScales (int): number of wavelet scales (size of the filter bank)
        numLayers (int): number of layers
        adjacencyMatrix (np.array): of shape N x N
    Output:
        Instantiates the class for the diffusion scattering transform
    Methods:
        Phi = .computeTransform(x): computes the diffusion scattering
            coefficients of input x (np.array of shape B x F x N, with B the
            batch size, F the number of node features, and N the number of
            nodes)
    """

    def __init__(self, numScales, numLayers, adjacencyMatrix):
        super().__init__(numScales, numLayers, adjacencyMatrix)
        d = np.sum(self.W, axis = 1)
        killIndices = np.nonzero(d < zeroTolerance)[0] # Nodes with zero
            # degree or negative degree (there shouldn't be any since (i) the
            # graph is connected -all nonzero degrees- and (ii) all edge
            # weights are supposed to be positive)
        dReady = d.copy()
        dReady[killIndices] = 1.
        # Apply sqrt and invert without fear of getting nans or stuff
        dSqrtInv = 1./np.sqrt(dReady)
        # Put back zeros in those numbers that had been failing
        dSqrtInv[killIndices] = 0.
        # Inverse diagonal squareroot matrix
        DsqrtInv = np.diag(dSqrtInv)
        # Normalized adjacency
        A = DsqrtInv.dot(self.W.dot(DsqrtInv))
        # Lazy diffusion
        self.T = 1/2*(np.eye(self.N) + A)
        # Low-pass average operator
        self.U = d/np.linalg.norm(d, 1)
        # Construct wavelets
        self.H = diffusionWavelets(self.J, self.T)


class GeometricScattering:
    # Because the geometric scattering computes Q coefficients every time the
    # "low pass" operator is applied, it cannot just inherit from the
    # GraphScatteringTransform class as all the other GSTs. However, the
    # basis of the code in the .computeTransform() method is essentially the
    # same, only with the application of "U" adapted to output the Q
    # moments, instead of just a single value
    # TODO: Add a ".computeMoments()" method on the "GraphScatteringTransform"
    # class so that we can unify all methods. In the already existing methods,
    # this would just be multiplying by U.
    """
    GeometricScattering: geometric scattering transform
    Initialization:
    Input:
        numScales (int): number of wavelet scales (size of the filter bank)
        numLayers (int): number of layers
        numMoments (int): number of moments to compute invariants
        adjacencyMatrix (np.array): of shape N x N
    Output:
        Instantiates the class for the geometric scattering transform
    Methods:
        Phi = .computeTransform(x): computes the diffusion scattering
            coefficients of input x (np.array of shape B x F x N, with B the
            batch size, F the number of node features, and N the number of
            nodes)
    """

    def __init__(self, numScales, numLayers, numMoments, adjacencyMatrix):

        self.J = numScales
        self.L = numLayers
        assert numMoments > 0
        self.Q = numMoments
        self.W = adjacencyMatrix.copy()
        self.N = self.W.shape[0]
        assert self.W.shape[1] == self.N

        d = np.sum(self.W, axis = 1)
        killIndices = np.nonzero(d < zeroTolerance)[0] # Nodes with zero
            # degree or negative degree (there shouldn't be any since (i) the
            # graph is connected -all nonzero degrees- and (ii) all edge
            # weights are supposed to be positive)
        dReady = d.copy()
        dReady[killIndices] = 1.
        # Apply sqrt and invert without fear of getting nans or stuff
        dInv = 1./dReady
        # Put back zeros in those numbers that had been failing
        dInv[killIndices] = 0.
        # Inverse diagonal squareroot matrix
        Dinv = np.diag(dInv)
        # Lazy diffusion random walk
        self.P = 1/2*(np.eye(self.N) + self.W.dot(Dinv))
        # Construct wavelets
        self.H = diffusionWavelets(self.J, self.P)
        #   Note that the diffusion wavelets coefficients don't change. What
        #   changes is the matrix (now it's the lazy diffusion random walk
        #   instead of the lazy diffusion adjacency), but nothing else.

    def computeMoments(self, x):

        # The input is B x J x F x N and the output has to be B x J x F x Q
        # (J is the number of scales we have up to here, it doesn't matter)
        assert len(x.shape) == 4
        assert x.shape[3] == self.N

        # Because we have checked that Q is at least 1, we know that the first
        # order moment we will always be here, so we just compute it
        Sx = np.sum(x, axis = 3) # B x J x F
        # Add the dim, because on that dim we will concatenate the values of Q
        Sx = np.expand_dims(Sx, 3) # B x J x F x 1
        # Now, for all other values of Q that we haven't used yet
        for q in range(2, self.Q+1): # q = 2, ..., Q
            # Compute the qth moment and add up
            thisMoment = np.sum(x ** q, axis = 3) # B x J x F
            # Add the extra dimension
            thisMoment = np.expand_dims(thisMoment, 3) # B x J x F x 1
            # Concatenate to the already existing Sx
            Sx = np.concatenate((Sx, thisMoment), axis = 3) # B x J x F x q

        # When we're done, we return this
        return Sx # B x J x F x Q

    def computeTransform(self, x):
        # Check the dimension of x: batchSize x numberFeatures x numberNodes
        assert len(x.shape) == 3
        B = x.shape[0] # batchSize
        F = x.shape[1] # numberFeatures
        assert x.shape[2] == self.N
        # Start creating the output
        #   Compute the first coefficients
        Phi = self.computeMoments(np.expand_dims(x, 1)) # B x 1 x F x Q
        Phi = Phi.squeeze(1) # B x F x Q
        # Reshape x to account for the increasing J dimension that we will have
        rhoHx = x.reshape(B, 1, F, self.N) # B x 1 x F x N
        # Now, we move to the rest of the layers
        for l in range(1,self.L): # l = 1,2,...,L
            nextRhoHx = np.empty([B, 0, F, self.N])
            for j in range(self.J ** (l-1)): # j = 0,...,l-1
                thisX = rhoHx[:,j,:,:] # B x 1 x F x N
                thisHx = thisX.reshape(B, 1, F, self.N) \
                            @ self.H.reshape(1, self.J, self.N, self.N)
                    # B x J x F x N
                thisRhoHx = np.abs(thisHx) # B x J x F x N
                nextRhoHx = np.concatenate((nextRhoHx, thisRhoHx), axis = 1)

                phi_j = self.computeMoments(thisRhoHx) # B x J x F x Q
                phi_j = phi_j.transpose(0, 2, 1, 3) # B x F x J x Q
                phi_j = phi_j.reshape(B, F, self.J * self.Q)
                Phi = np.concatenate((Phi, phi_j), axis = 2) # Keeps adding the
                    # coefficients
            rhoHx = nextRhoHx.copy()

        return Phi