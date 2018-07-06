# -*- coding: utf-8 -*-

import numpy as np
from numpy import pi
from numpy.linalg import norm, det

EPSILON = 1e-9

class MassLoadingPlate:
    def __init__(self, primitive_lattice_vectors,
                 primitive_reciprocal_lattice_vectors,
                 inclusion_positions, mass, num_modes):

        self.primitive_lattice_vectors = np.array(primitive_lattice_vectors)
        self.primitive_reciprocal_lattice_vectors = primitive_reciprocal_lattice_vectors
        self.mass = mass
        self.inclusion_positions = inclusion_positions
        self.num_modes = num_modes
        self.dimemsion = len(primitive_lattice_vectors)

        # Check if the primitive lattice vectors are compatible to the
        # primitive reciprocal lattice vectors
        D = np.dot(primitive_lattice_vectors,
                   np.transpose(primitive_reciprocal_lattice_vectors))

        if np.any(abs(D - 2*pi * np.identity(self.dimemsion)) > EPISLON):
            raise Exception("The primitive lattice vectors are incompatible with the reciprocal counterparts.")

        # Check if the inculsions are located inside the primitive unit cell
        D = np.dot(primitive_lattice_vectors,
                   np.transpose(inclusion_positions)))

        if np.any( D < 0) or np.any( D > 1):
            raise Exception("Inclusions must be inside the fundemental primitive unit cell.")


    def get_primitive_cell_volume(self):
        return abs(det(self.primitive_lattice_vectors))

    def get_reciprocal_lattice_vectors(self):
        # Generate integer indices for the reciprocal lattice vectors
        indices = [[]]
        for n in range(self.dimemsion):
            next_indices = []
            for i in indices:
                for j in range(self.num_modes):
                    next_indices.append(i+[j])
            indices = next_indices

        reciprocal_lattice_vectors = []
        for i in indices:
            # b_i dot with the i
            pass




    def get_frequency(self, bloch_wavevector):
        kappa = bloch_wavevector
        G = self.get_reciprocal_lattice_vectors()

        K = np.asmatrix( np.diag( norm(G-kappa, axis=1)**4 ) )
        P = np.exp(1j * np.dot(self.inclusion_positions, np.transpose(G-kappa)))

        A = np.block
